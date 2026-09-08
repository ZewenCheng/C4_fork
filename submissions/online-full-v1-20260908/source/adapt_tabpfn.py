"""TabPFN V2冻结基底低秩适配；训练分区内部上下文/查询轮换，不使用保留集。"""
import os,sys,json,time
os.environ['TABPFN_DISABLE_TELEMETRY']='1';os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
sys.path.insert(0,'/workspace/work/c4-experiments/expert-autotrain-20260906')
from download_assets import ROOT,dump,sha
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,torch
from torch import nn
from torch.nn.utils import parametrize
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from train_tabular import facts

class LowRankWeight(nn.Module):
    def __init__(self,weight):
        super().__init__();self.a=nn.Parameter(torch.randn(4,weight.shape[1])*.01);self.b=nn.Parameter(torch.zeros(weight.shape[0],4))
    def forward(self,weight):return weight+(self.b@self.a)*2

def main():
    torch.set_num_threads(2);seed=20260906
    rows=json.loads((ROOT/'data/manifest.json').read_text());xs=[];ys=[]
    for row in rows:
        if row['split']!='train':continue
        for a in row['annotations']:
            try:y=int(float(a.get('ratingScale(1-5)')))
            except (TypeError,ValueError):continue
            if 1<=y<=5:xs.append(facts(a));ys.append(y)
    x=np.asarray(xs,dtype=object);y=np.asarray(ys)
    root=ROOT/'checkpoints/TabPFN_adapter';root.mkdir(exist_ok=True)
    for optname in ['adamw','sgd_momentum']:
        out=root/optname;out.mkdir(exist_ok=True);last=out/'last.pt'
        if (out/'complete.json').exists():continue
        torch.manual_seed(seed)
        clf=TabPFNClassifier(model_path=ROOT/'models/TabPFN-v2/tabpfn-v2-classifier-v2_default.ckpt',device='cpu',n_estimators=1,n_jobs=1,random_state=seed,inference_precision=torch.float32)
        counter=[0]
        def split(a,b):
            state=seed+counter[0];counter[0]+=1
            return train_test_split(a,b,test_size=.25,random_state=state,stratify=b)
        dataset=clf.get_preprocessed_datasets(x,y,split,max_data_size=None)
        clf.model_.requires_grad_(False);names=[]
        for name,module in list(clf.model_.named_modules()):
            if isinstance(module,nn.Linear):parametrize.register_parametrization(module,'weight',LowRankWeight(module.weight));names.append(name)
        params=[p for p in clf.model_.parameters() if p.requires_grad]
        if not params:raise RuntimeError('未发现可训练适配器')
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01) if optname=='adamw' else torch.optim.SGD(params,lr=.005,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100);start=0
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False);clf.model_.load_state_dict(saved['adapter'],strict=False);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);start=saved['step'];torch.set_rng_state(saved['rng']);dataset.rng.bit_generator.state=saved['dataset_rng'];counter[0]=saved['split_counter']
        dump(out/'config.json',{'rank':4,'scale':2,'modules':names,'base_sha256':sha(ROOT/'models/TabPFN-v2/tabpfn-v2-classifier-v2_default.ckpt'),'train_rows':len(y),'episodes':100,'attribution':'Built with PriorLabs-TabPFN','trainable_parameters':sum(p.numel() for p in params),'heldout_use':'未参与本训练，适配完成后整体评估'})
        begin=time.monotonic()
        for step in range(start,100):
            a,b,c,d,cats,configs=dataset[0]
            clf.fit_from_preprocessed([v.unsqueeze(0) for v in a],[v.unsqueeze(0) for v in c],[cats],[[v] for v in configs])
            clf.model_.train();optimizer.zero_grad(set_to_none=True)
            logits=clf.forward([v.unsqueeze(0) for v in b],use_inference_mode=False,return_logits=True)
            loss=nn.functional.cross_entropy(logits,d.long().unsqueeze(0))
            if not loss.requires_grad or not torch.isfinite(loss):raise RuntimeError('未形成有效梯度损失')
            loss.backward();grad=float(nn.utils.clip_grad_norm_(params,1));optimizer.step();scheduler.step()
            adapter={k:v for k,v in clf.model_.state_dict().items() if '.parametrizations.weight.0.' in k}
            tmp=out/'last.pt.tmp';torch.save({'adapter':adapter,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step+1,'rng':torch.get_rng_state(),'dataset_rng':dataset.rng.bit_generator.state,'split_counter':counter[0]},tmp);os.replace(tmp,last)
            dump(ROOT/'tabpfn_adapter_status.json',{'status':'gradient_adaptation','optimizer':optname,'step':step+1,'episodes':100,'loss':float(loss.detach()),'gradient_norm':grad,'remaining_seconds_current_optimizer':(99-step)*(time.monotonic()-begin)/(step-start+1),'time':time.time()})
        dump(out/'complete.json',{'status':'adapter_trained_overall_inference_pending','sha256':sha(last),'time':time.time()})
    dump(ROOT/'tabpfn_adapter_status.json',{'status':'both_adapters_trained_overall_inference_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'tabpfn_adapter_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
