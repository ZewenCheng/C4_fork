"""裁块补证有界子进程与缓存，参数由登记案件限定。"""
from pathlib import Path
import os,signal,subprocess,sys
from threading import Lock
from report_contract import ContractError,read_json,atomic_json,digest_file,digest_json,run_lock


class CropPartner:
    def __init__(self,rows,asset_root,directory):
        self.rows={r['sample_id']:r for r in rows};self.assets=Path(asset_root);self.directory=Path(directory)
        self.worker=Path(__file__).with_name('partner_crop_worker.py')
        self.device_lock=Lock()
        self.version={'worker':digest_file(self.worker),'input_quality':digest_file(self.worker.with_name('input_quality.py')),
                      'checkpoint':digest_file(self.assets/'checkpoints/rtdetr_adamw/last.pt')}

    def __call__(self,args):
        with self.device_lock:
            return self._execute(args)

    def _execute(self,args):
        if set(args)!={'sample_id'} or args['sample_id'] not in self.rows:raise ContractError('裁块案件不合法')
        row=self.rows[args['sample_id']]
        if digest_file(row['path'])!=row['image_sha256']:raise ContractError('裁块输入不一致')
        request={'row':row,'asset_root':str(self.assets),'version':self.version}
        folder=self.directory/digest_json(request)
        with run_lock(folder/'run.lock'):
            output=folder/'response.json';receipt=folder/'complete.json'
            if receipt.exists():
                if read_json(receipt)['sha256']!=digest_file(output):raise ContractError('裁块缓存损坏')
                value=read_json(output);return {**value,'cache_reused':True,'model_calls':0,'new_visual_observation':False}
            state=read_json(folder/'state.json') if (folder/'state.json').exists() else {'attempts':0}
            if state['attempts']>=2:raise ContractError('裁块尝试额度耗尽')
            state['attempts']+=1;atomic_json(folder/'state.json',state);atomic_json(folder/'request.json',request)
            with (folder/f'attempt-{state["attempts"]}.log').open('xb') as log:
                child=subprocess.Popen([sys.executable,'-B',str(self.worker),'--request',str(folder/'request.json'),'--output',str(output)],
                    stdout=log,stderr=log,start_new_session=True)
                try:code=child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('裁块推理超时')
            if code:raise RuntimeError('裁块推理失败，日志留平台')
            value=read_json(output)
            if value.get('input_sha256')!=row['image_sha256'] or value['result']['sample_id']!=row['sample_id']:raise ContractError('裁块响应身份不符')
            atomic_json(receipt,{'sha256':digest_file(output),'version':self.version})
            return {**value,'cache_reused':False}
