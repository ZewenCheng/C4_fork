"""线上完整专家证据组装；只从图像预测构造历史评级输入，不读取线上标签。"""
import os,sys,json,time
from pathlib import Path
import numpy as np,torch,joblib
from torch.nn import functional as F
from download_assets import ROOT,dump,sha
from train_fusion import Fusion
from train_legacy_head import LegacyHead
from train_ordinal import OrdinalExpert
from train_tabular import facts,CatBoostClassifier
from query_standard import query_standard
from report_contract import select_region_candidates

ONLINE=ROOT/'candidate_online_20260908'
OPTS=['adamw','sgd_momentum']

def checked(folder,key='checkpoint_sha256'):
    p=folder/'last.pt';digest=sha(p)
    assert digest==json.loads((folder/'complete.json').read_text())[key]
    return p,digest

def candidates(scores,labels):
    values,indices=torch.as_tensor(scores).topk(5)
    return [{'label':labels[int(i)],'label_id':int(i),'uncalibrated_score':float(v)} for v,i in zip(values,indices)]

def main():
    torch.set_num_threads(4)
    rows=json.loads((ONLINE/'input_manifest.json').read_text());labels=json.loads((ROOT/'data/vocabulary.json').read_text())['raw_classes']
    out=ONLINE/'packets';out.mkdir(exist_ok=True)
    fusion={};legacy={};assets={};enc=json.loads((ROOT/'checkpoints/ordinal_history/encoding.json').read_text());ordinal={}
    for opt in OPTS:
        folder=ROOT/'checkpoints/semantic_fusion'/opt;cfg=json.loads((folder/'configuration.json').read_text());p,d=checked(folder)
        model=Fusion(cfg['widths'],cfg['classes']);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);fusion[opt]=model.eval();assets['fusion_'+opt]=d
        # 保留训练来源核验；线上特征另外验证其实际基底与适配权重，不冒充训练特征。
        for rel,digest in cfg['sources'].items():assert sha(ROOT/rel/'source.json')==digest
        cfg=json.loads((ROOT/'checkpoints/legacy_convnext_head/config.json').read_text());p,d=checked(ROOT/'checkpoints/legacy_convnext_head'/opt)
        model=LegacyHead(cfg['width'],cfg['classes']);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);legacy[opt]=model.eval();assets['legacy_'+opt]=d
        p,d=checked(ROOT/'checkpoints/ordinal_history'/opt);model=OrdinalExpert([len(x)+1 for x in enc['categorical_maps']]);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model']);ordinal[opt]=model.eval();assets['ordinal_'+opt]=d
        for family,prefix in [('hplus_distilled_features','dinov3_hplus_distilled_'),('hplus_original_features','dinov3_hplus_'),('wemm_features','wemm9b_')]:
            _,d=checked(ROOT/'checkpoints'/(prefix+opt));source=json.loads((ONLINE/family/opt/'source.json').read_text());assert source['checkpoint_sha256']==d;assets[prefix+opt]=d
    legacy_source=json.loads((ONLINE/'legacy_convnext_features/source.json').read_text());assert legacy_source['base_sha256']==sha(Path(legacy_source['base'])/'model.safetensors')
    assets['convnext_base']=legacy_source['base_sha256']
    packets=[];input_facts=[]
    bank=json.loads((ROOT/'checkpoints/anomaly_banks/configuration.json').read_text())
    for i,row in enumerate(rows):
        sid=row['sample_id'];evidence={};events=[]
        def add(tool,result):
            evidence[tool]={'status':'ok','tool':tool,'result':result};events.append({'name':tool,'arguments':{'sample_id':sid},'status':'ok','invoked_by':'online_assembly'})
        heads={};local_heads={};standalone={}
        for opt in OPTS:
            fs=[]
            for family in ['hplus_distilled_features','wemm_features','legacy_convnext_features']:
                folder=ONLINE/family if family=='legacy_convnext_features' else ONLINE/family/opt
                with np.load(folder/(sid+'.npz'),allow_pickle=False) as z:f=torch.from_numpy(z['features'].astype(np.float32))
                assert torch.isfinite(f).all();fs.append(F.normalize(F.normalize(f,dim=-1).mean(0),dim=-1).unsqueeze(0))
                if family=='legacy_convnext_features':lf=F.normalize(f,dim=-1)
            with torch.inference_mode():
                logits,gates=fusion[opt](fs);scores=logits[0].softmax(-1)
                ls=(torch.logsumexp(legacy[opt](lf),0)-np.log(3)).softmax(-1)
            heads[opt]={'checkpoint_sha256':assets['fusion_'+opt],'candidates':candidates(scores,labels),'expert_weights':gates[0].tolist()}
            local_heads[opt]={'checkpoint_sha256':assets['legacy_'+opt],'candidates':candidates(ls,labels)}
            for family in ['hplus_distilled_features','hplus_original_features']:
                with np.load(ONLINE/family/opt/(sid+'.npz'),allow_pickle=False) as z:sc=z['scores']
                assert np.isfinite(sc).all();standalone[family+'_'+opt]=candidates(sc,labels)
        add('inspect_semantics',{'sample_id':sid,'outputs':heads});add('inspect_local',{'sample_id':sid,'outputs':local_heads});add('inspect_hplus',{'sample_id':sid,'outputs':standalone})
        for tool,family,prefix in [('inspect_regions','rtdetr_holdout','rtdetr_'),('inspect_masks','sam_holdout','sam3_'),('inspect_grounding','grounding_holdout','grounding_dino_')]:
            outputs={}
            for opt in OPTS:
                folder=ONLINE/family/opt;p=folder/(sid+'.json');packet=json.loads(p.read_text());_,digest=checked(ROOT/'checkpoints'/(prefix+opt));assert packet['sample_id']==sid and packet['checkpoint_sha256']==digest
                assert json.loads((folder/'complete.json').read_text())['checkpoint_sha256']==digest;assets[prefix+opt]=digest
                if tool=='inspect_masks':
                    assert sha(folder/packet['mask_file'])==packet['mask_sha256']
                    with np.load(folder/packet['mask_file'],allow_pickle=False) as z:m=z['masks']
                    assert m.shape==(len(packet['candidates']),8192) and m.dtype==np.uint8
                    for c in packet['candidates']:assert int(np.unpackbits(m[c['mask_index']]).sum())==c['mask_grid_area_pixels']
                for c in packet['candidates']:
                    box=c['box_normalized_xyxy'];assert len(box)==4 and all(np.isfinite(v) and 0<=v<=1 for v in box) and box[0]<=box[2] and box[1]<=box[3]
                packet['total_candidates']=len(packet['candidates'])
                packet['candidates']=select_region_candidates(packet['candidates'],heads['adamw']['candidates'][0]['label'],limit=5)
                packet['selection_policy']='report-contract-v1:类型对应与查询覆盖优先，预算5，不改变分类或候选分数'
                outputs[opt]=packet
            add(tool,{'sample_id':sid,'outputs':outputs})
        anomaly=json.loads((ONLINE/'anomaly_holdout'/(sid+'.json')).read_text())['result'];assert anomaly['sample_id']==sid
        for name,val in anomaly['scores'].items():assert val['bank_sha256']==bank['banks'][name]['sha256'];assets['anomaly_'+name]=val['bank_sha256']
        add('inspect_anomaly',anomaly)
        label=heads['adamw']['candidates'][0]['label']
        category='桥梁' if '桥梁' in Path(row['path']).parts else '轨道'
        predicted={'questionCategory':category,'defectType':label,'defectLocation':'','defectDescription':''}
        input_facts.append(facts(predicted))
        add('query_standard',query_standard(label,limit=2))
        packets.append({'sample_id':sid,'filename':Path(row['path']).name,'category':category,'predicted_type':label,'evidence':evidence,'events':events})
        dump(ONLINE/'assembly_status.json',{'status':'assembling_visual_evidence','completed':i+1,'total':len(rows),'time':time.time()})
    x=np.asarray(input_facts,dtype=object);predictions={}
    for depth in [6,8]:
        p=ROOT/'checkpoints/tabular_history'/f'catboost_depth{depth}.cbm';assert sha(p)==json.loads((p.parent/f'metrics_depth{depth}.json').read_text())['sha256']
        model=CatBoostClassifier();model.load_model(str(p));predictions['catboost_'+str(depth)]=model.predict(x,thread_count=4).reshape(-1).astype(int);assets['catboost_'+str(depth)]=sha(p)
    cs=torch.tensor([[enc['categorical_maps'][j].get(r[j],0) for j in range(3)] for r in input_facts]);ns=torch.tensor([[np.log1p(max(r[3],0)),*r[4:]] for r in input_facts],dtype=torch.float32)
    ns=(ns-torch.tensor(enc['numeric_mean']))/torch.tensor(enc['numeric_std'])
    for opt in OPTS:
        with torch.inference_mode():predictions['ordinal_'+opt]=(1+(ordinal[opt](cs,ns).sigmoid()>=.5).sum(1)).numpy()
        folder=ROOT/'checkpoints/TabPFN_adapter'/opt;p=folder/'fitted_adapter.cloud_only.joblib';assert sha(p)==json.loads((folder/'whole_holdout_metrics.json').read_text())['fitted_sha256']
        model=joblib.load(p);predictions['tabpfn_'+opt]=model.predict(x).astype(int);assets['tabpfn_'+opt]=sha(p)
        del model
    # 规范资产真实加载；缺少构件面积分母与标定时不调用虚构量值。
    p=ROOT/'checkpoints/normative_area_v1/model.cbm';assert sha(p)==json.loads((p.parent/'complete.json').read_text())['model_sha256'];norm=CatBoostClassifier();norm.load_model(str(p));assets['normative_area']=sha(p)
    rules=json.loads((p.parent/'rulebook.json').read_text())
    for i,packet in enumerate(packets):
        evidence={'historical_predictions':{k:int(v[i]) for k,v in predictions.items()},'input_facts':input_facts[i],'source':'线上图像分类预测；数量与位置缺失，不是标注事实','normative':{'status':'missing_measurement_and_component_scope','model_loaded':True,'model_sha256':assets['normative_area'],'rulebook':rules,'final_grade':None},'limitations':['历史模型在标注事实训练、此处输入模型预测，存在分布变化','TabPFN已知多数类退化，历史预测不当作规范等级']}
        packet['evidence']['assess_history']={'status':'ok','result':evidence};packet['events'].append({'name':'assess_history','arguments':{'sample_id':packet['sample_id']},'status':'ok'})
        dump(out/(packet['sample_id']+'.json'),packet)
    dump(ONLINE/'asset_usage.json',{'assets':assets,'images':len(rows),'normative_area':'已加载，因无实测与构件范围暂不输出等级','teacher':'冻结7B真实异常推理','qwen':'下一阶段真实生成','time':time.time()})
    dump(ONLINE/'assembly_status.json',{'status':'all_online_expert_packets_complete','images':len(rows),'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ONLINE/'assembly_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
