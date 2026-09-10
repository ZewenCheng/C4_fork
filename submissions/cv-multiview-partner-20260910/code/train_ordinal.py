"""CPU序数评级专家：有序切点、训练集编码、双优化器；历史轨不冒充规范轨。"""
import json,time,random,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from download_assets import ROOT,dump,sha
from train_tabular import facts

class OrdinalExpert(nn.Module):
    def __init__(self,sizes):
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(n, min(32,max(4,int(n**.5)*2))) for n in sizes])
        width=sum(e.embedding_dim for e in self.embeddings)+5
        self.net=nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        self.cut_start=nn.Parameter(torch.tensor(-1.5));self.cut_increments=nn.Parameter(torch.zeros(3))
    def forward(self,cats,nums):
        z=self.net(torch.cat([e(cats[:,i]) for i,e in enumerate(self.embeddings)]+[nums],dim=1))
        cuts=torch.cat([self.cut_start[None],self.cut_start+torch.cumsum(F.softplus(self.cut_increments)+.001,0)])
        return z-cuts[None,:]

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    seed=20260906;rows=json.loads((ROOT/'data/manifest.json').read_text());parts={'train':[],'holdout':[]}
    for row in rows:
        for a in row['annotations']:
            try:y=int(float(a.get('ratingScale(1-5)')))
            except (TypeError,ValueError):continue
            if 1<=y<=5:parts['train' if row['split']=='train' else 'holdout'].append((facts(a),y,1/max(1,len(row['annotations']))))
    mappings=[{v:i+1 for i,v in enumerate(sorted({x[0][j] for x in parts['train']}))} for j in range(3)]
    def encode(items):
        c=torch.tensor([[mappings[j].get(x[j],0) for j in range(3)] for x,_,_ in items],dtype=torch.long)
        n=torch.tensor([[np.log1p(max(x[3],0)),*x[4:]] for x,_,_ in items],dtype=torch.float32)
        return c,n,torch.tensor([y for _,y,_ in items]),torch.tensor([w for _,_,w in items])
    tc,tn,ty,tw=encode(parts['train']);vc,vn,vy,vw=encode(parts['holdout'])
    mean=tn.mean(0);std=tn.std(0).clamp_min(.01);tn=(tn-mean)/std;vn=(vn-mean)/std
    target=(ty[:,None]>torch.arange(1,5)[None,:]).float()
    base=ROOT/'checkpoints/ordinal_history';base.mkdir(exist_ok=True)
    dump(base/'encoding.json',{'categorical_maps':mappings,'numeric_mean':mean.tolist(),'numeric_std':std.tolist(),'classes':[1,2,3,4,5],
        'numeric_features':['log1p_text_length_max_mm','quantity_present','through_phrase','progress_phrase','repair_phrase'],
        'source':'与CatBoost相同的历史标注事实，不是工程实测宽度或规范评级真值','seed':seed})
    for optname in ['adamw','sgd_momentum']:
        out=base/optname;out.mkdir(exist_ok=True);last=out/'last.pt'
        if (out/'complete.json').exists():continue
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=OrdinalExpert([len(m)+1 for m in mappings]);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01) if optname=='adamw' else torch.optim.SGD(model.parameters(),lr=.02,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80);start_epoch=0;steps=0
        if last.exists():
            d=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);scheduler.load_state_dict(d['scheduler']);start_epoch=d['epoch'];steps=d['steps'];torch.set_rng_state(d['torch_rng'])
        begin=time.monotonic()
        for epoch in range(start_epoch,80):
            model.train();order=torch.randperm(len(ty));total=0.
            for offset in range(0,len(order),128):
                ix=order[offset:offset+128];opt.zero_grad(set_to_none=True)
                loss=(F.binary_cross_entropy_with_logits(model(tc[ix],tn[ix]),target[ix],reduction='none').mean(1)*tw[ix]).sum()/tw[ix].sum()
                if not torch.isfinite(loss):raise RuntimeError('序数损失非有限')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1;total+=float(loss.detach())
            scheduler.step();tmp=out/'last.pt.tmp'
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch+1,'steps':steps,'torch_rng':torch.get_rng_state()},tmp);os.replace(tmp,last)
            dump(ROOT/'ordinal_status.json',{'status':'training_history_ordinal','optimizer':optname,'epoch':epoch+1,'epochs':80,'steps':steps,'epoch_loss_sum':total,'time':time.time()})
        model.eval()
        with torch.inference_mode():
            probabilities=torch.sigmoid(model(vc,vn));prediction=1+(probabilities>=.5).sum(1)
        dump(out/'complete.json',{'status':'trained','checkpoint_sha256':sha(last),'train_rows':len(ty),'holdout_rows':len(vy),'accuracy_given_annotation_facts':float((prediction==vy).float().mean()),'mae_given_annotation_facts':float((prediction-vy).abs().float().mean()),'seconds_this_process':time.monotonic()-begin,'limitation':'历史标注事实输入；不是图片端到端或规范符合性成绩'})
    dump(ROOT/'ordinal_status.json',{'status':'both_optimizers_trained_overall_inference_pending','time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'ordinal_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
