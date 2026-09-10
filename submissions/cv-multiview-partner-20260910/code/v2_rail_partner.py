"""冻结v2完整轨道特征链适配，证据绑定输入和版本，原生标签显式映射。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import numpy as np
from report_contract import ContractError, atomic_json, digest_file, digest_json, read_json, run_lock

DIMENSIONS = [4096, 1152, 1024, 1024, 1536, 1152]


class V2RailPartner:
    def __init__(self, legacy_root, directory, label_mapping, *, timeout=1200):
        self.root, self.directory = Path(legacy_root).resolve(), Path(directory).resolve()
        self.mapping = dict(label_mapping)
        self.timeout = timeout
        names = ['src/common.py', 'src/extract_features.py', 'src/qwen_adapter.py',
                 'adapters/qwen_local_provider.py', 'config/deploy_config.json',
                 'models/MODEL_MANIFEST.json', 'models/classifier/expq_lr.npz']
        self.version = {'source': {n: digest_file(self.root/n) for n in names},
                        'mapping_sha256': digest_json(self.mapping), 'dimensions': DIMENSIONS,
                        'adapter_sha256': digest_file(__file__)}

    def inspect(self, row, category):
        if category != '轨道':
            return {'status': 'unavailable', 'result': {'sample_id': row['sample_id']},
                    'reason': '旧链仅适用于轨道，未执行桥梁默认分类', 'model_calls': 0}
        return self.inspect_many([{**row, 'questionCategory': category}])[row['sample_id']]

    def inspect_many(self, rows):
        if not rows or len({r['sample_id'] for r in rows}) != len(rows):
            raise ContractError('旧链输入为空或案件重复')
        if len({Path(r['path']).name for r in rows}) != len(rows):
            raise ContractError('旧链批次文件名重复')
        for row in rows:
            if row.get('questionCategory') != '轨道':
                raise ContractError('旧链批次必须逐项登记为轨道')
            if digest_file(row['path']) != row['image_sha256']:
                raise ContractError('旧链输入摘要错误')
        identity = {'version': self.version, 'inputs': [{k: r[k] for k in ('sample_id','image_sha256')} for r in rows]}
        folder = self.directory/digest_json(identity)
        with run_lock(folder/'run.lock'):
            receipt = folder/'receipt.json'
            features = folder/'features.npz'
            reused = False
            if receipt.exists():
                data = read_json(receipt)
                if data['identity'] != identity or digest_file(features) != data['features_sha256']:
                    raise ContractError('旧链缓存身份或特征摘要不一致')
                reused = True
            else:
                state_path = folder/'attempts.json'
                state = read_json(state_path) if state_path.exists() else {'attempts': 0}
                if state['attempts'] >= 2:
                    raise ContractError('旧链特征提取尝试耗尽')
                state['attempts'] += 1; atomic_json(state_path, state)
                manifest = folder/'manifest.jsonl'
                manifest.write_text('\n'.join(json.dumps({'filename':Path(r['path']).name,
                    'image_path':str(Path(r['path']).resolve()),'questionCategory':'轨道'},ensure_ascii=False) for r in rows)+'\n')
                env = os.environ.copy()
                env.update(QWEN_PROVIDER_MODULE='qwen_local_provider', QWEN_MODEL='Qwen3-VL-Embedding-8B',
                    QWEN_LOCAL_MODEL_DIR=str(self.root/'models/transformers/qwen3_vl_embedding'),
                    QWEN_LOCAL_OFFLOAD_DIR=str(folder/'offload'), QWEN_LOCAL_GPU_GIB='40', QWEN_LOCAL_CPU_GIB='8',
                    QWEN_LOCAL_BATCH_SIZE='1', CQAIP_MODELS_DIR=str(self.root/'models'),
                    OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    PYTHONPATH=os.pathsep.join([str(self.root/'.runtime/qwen_deps'),str(self.root/'adapters'),env.get('PYTHONPATH','')]))
                started=time.time()
                with (folder/f'extraction-{state["attempts"]}.log').open('xb') as log:
                    proc=subprocess.run([sys.executable,'-B',str(self.root/'src/extract_features.py'),
                        '--manifest',str(manifest),'--output',str(features),'--config',str(self.root/'config/deploy_config.json'),
                        '--device','cuda','--batch-images','2'],env=env,stdout=log,stderr=log,timeout=self.timeout)
                if proc.returncode:
                    raise RuntimeError(f'旧链提取失败，退出码{proc.returncode}，日志留平台')
                data={'identity':identity,'features_sha256':digest_file(features),
                      'extraction_seconds':time.time()-started,'feature_path':str(features),
                      'logical_backbone_passes':6,'actual_forward_calls':None,
                      'cost_limit':'六分支含三视图及微批，实际forward次数尚未计数'}
                atomic_json(receipt,data)
            with np.load(features,allow_pickle=False) as raw:
                filenames=raw['filenames'].astype(str).tolist()
                xbase=np.asarray(raw['xbase'],dtype=np.float32)
                ft=np.asarray(raw['expq_ft'],dtype=np.float32)
            if filenames != [Path(r['path']).name for r in rows] or xbase.shape!=(len(rows),8832) or ft.shape!=(len(rows),1152):
                raise ContractError('旧链特征维度、文件名或顺序错误')
            if not np.isfinite(xbase).all() or not np.isfinite(ft).all():
                raise ContractError('旧链特征含非有限值')
            norm=np.linalg.norm(ft,axis=-1,keepdims=True)
            if np.any(norm<=1e-12):raise ContractError('旧链expQ零特征')
            x=np.hstack([xbase,ft/norm]).astype(np.float32)
            with np.load(self.root/'models/classifier/expq_lr.npz',allow_pickle=False) as model:
                if int(model['feature_dim'])!=9984:raise ContractError('旧分类器维度不符')
                labels=model['class_names'].astype(str)
                logits=x.astype(np.float64)@model['coef'].astype(np.float64).T+model['intercept'].astype(np.float64)
            logits-=logits.max(axis=1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(axis=1,keepdims=True)
            result={}
            for i,row in enumerate(rows):
                candidates=[{'native_label':str(labels[j]),'label':self.mapping.get(str(labels[j])),
                             'mapping_status':'mapped' if self.mapping.get(str(labels[j])) else 'unmapped',
                             'uncalibrated_score':float(prob[i,j])} for j in np.argsort(-prob[i])]
                result[row['sample_id']]={'status':'ok' if all(c['label'] for c in candidates) else 'partial',
                    'result':{'sample_id':row['sample_id'],'outputs':{'frozen_v2':{'candidates':candidates,
                        'checkpoint_sha256':self.version['source']['models/classifier/expq_lr.npz'],
                        'preprocessing_sha256':digest_json(self.version)}}},
                    'input_sha256':row['image_sha256'],'source_family':'v2_shared_convnext_and_multibackbone',
                    'receipt':str(receipt),'cache_reused':reused,'extraction_seconds':0 if reused else data['extraction_seconds'],
                    'model_calls':0 if reused else None,'limitations':['原生分数未校准；不自动授予改判权限']}
            atomic_json(folder/'partner_responses.json',result)
            return result
