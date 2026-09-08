"""公开TabPFN V2历史评级上下文拟合；不把fit误称梯度微调。"""
import os,sys,json,time,hashlib
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
sys.path.insert(0,'/workspace/work/c4-experiments/expert-autotrain-20260906')
from download_assets import ROOT,dump,sha
sys.path.insert(0,str(ROOT/'deps'))
import requests,numpy as np,joblib,torch
from tabpfn import TabPFNClassifier
from train_tabular import facts

def main():
    torch.set_num_threads(2)
    repo='Prior-Labs/TabPFN-v2-clf';rev='f851f2a3c941544733b712d8c0f96dfae9b28862'
    folder=ROOT/'models/TabPFN-v2';folder.mkdir(exist_ok=True)
    file='tabpfn-v2-classifier-v2_default.ckpt';expected='cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'
    for name in ['LICENSE.txt','README.md',file]:
        p=folder/name
        if not p.exists():
            dump(ROOT/'tabpfn_status.json',{'status':'preparing_public_v2_base','file':name,'time':time.time()})
            r=requests.get(f'https://hf-mirror.com/{repo}/resolve/{rev}/{name}',timeout=(15,90));r.raise_for_status()
            if len(r.content)>40_000_000:raise ValueError('模型文件超出核定大小')
            tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(r.content);os.replace(tmp,p)
    if sha(folder/file)!=expected:raise ValueError('公开原始权重摘要不符')
    dump(folder/'source.json',{'repo':repo,'revision':rev,'sha256':expected,'license':'Prior Labs License 1.1','attribution':'Built with PriorLabs-TabPFN','model_name_prefix':'TabPFN','telemetry_disabled':True})
    rows=json.loads((ROOT/'data/manifest.json').read_text());tx=[];ty=[];vx=[];vy=[]
    for row in rows:
        for a in row['annotations']:
            try:y=int(float(a.get('ratingScale(1-5)')))
            except (TypeError,ValueError):continue
            if not 1<=y<=5:continue
            (tx if row['split']=='train' else vx).append(facts(a));(ty if row['split']=='train' else vy).append(y)
    model=TabPFNClassifier(model_path=folder/file,device='cpu',n_estimators=8,n_jobs=2,categorical_features_indices=[0,1,2],random_state=20260906)
    dump(ROOT/'tabpfn_status.json',{'status':'fitting_history_context','rows':len(tx),'gradient_finetuning':False,'time':time.time()})
    model.fit(np.asarray(tx,dtype=object),np.asarray(ty))
    out=ROOT/'checkpoints/TabPFN_history';out.mkdir(exist_ok=True)
    pred=model.predict(np.asarray(vx,dtype=object));tmp=out/'fitted.cloud_only.joblib.tmp';joblib.dump(model,tmp);os.replace(tmp,out/'fitted.cloud_only.joblib')
    dump(out/'complete.json',{'status':'context_fitted_gradient_adaptation_pending','train_rows':len(ty),'holdout_rows':len(vy),'accuracy_given_annotation_facts':float(np.mean(pred==vy)),'mae_given_annotation_facts':float(np.mean(np.abs(pred-np.asarray(vy)))),'checkpoint_sha256':sha(out/'fitted.cloud_only.joblib'),'attribution':'Built with PriorLabs-TabPFN','limitation':'上下文拟合不是梯度训练；标注事实输入不是图片端到端评级；拟合状态含训练事实，限平台保留','time':time.time()})
    dump(ROOT/'tabpfn_status.json',json.loads((out/'complete.json').read_text()))
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'tabpfn_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
