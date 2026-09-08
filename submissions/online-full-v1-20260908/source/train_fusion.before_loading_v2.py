"""等待适配特征齐备后训练三专家语义融合头；保留集仅用于最终整体推理。"""
import json,time,os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from download_assets import ROOT,dump,sha

class Fusion(nn.Module):
    def __init__(self,widths,classes):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(w),nn.Linear(w,256),nn.GELU()) for w in widths])
        self.gate=nn.Linear(256*len(widths),len(widths))
        self.head=nn.Sequential(nn.LayerNorm(256),nn.Dropout(.2),nn.Linear(256,classes))
    def forward(self,features):
        z=[p(x) for p,x in zip(self.projections,features)]
        gates=self.gate(torch.cat(z,-1)).softmax(-1)
        return self.head((torch.stack(z,1)*gates.unsqueeze(-1)).sum(1)),gates

def main():
    torch.set_num_threads(2)
    status=ROOT/'fusion_status.json'
    rows=json.loads((ROOT/'data/manifest.json').read_text())
    vocab=json.loads((ROOT/'data/vocabulary.json').read_text())
    target=torch.zeros(len(rows),len(vocab['raw_classes']))
    for i,row in enumerate(rows):target[i,row['label_ids']]=1/len(row['label_ids'])
    train=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train'])
    hold=torch.tensor([i for i,r in enumerate(rows) if r['split']=='holdout'])
    assert len(set(train.tolist())&set(hold.tolist()))==0 and len(train) and len(hold)
    for name in ['adamw','sgd_momentum']:
        out=ROOT/'checkpoints/semantic_fusion'/name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        folders=[ROOT/'hplus_distilled_features'/name,ROOT/'wemm_features'/name,ROOT/'legacy_convnext_features']
        while not all((p/'complete.json').exists() for p in folders[:2]):
            dump(status,{'status':'waiting_hplus_and_wemm_features','optimizer':name,'time':time.time()});time.sleep(60)
        sources={str(p.relative_to(ROOT)):sha(p/'source.json') for p in folders}
        features=[]
        for p in folders:
            vectors=[]
            for row in rows:
                with np.load(p/(row['sample_id']+'.npz')) as d:
                    f=torch.from_numpy(d['features'].astype(np.float32))
                    if f.ndim!=2 or not torch.isfinite(f).all():raise ValueError('特征维度或有限性异常')
                    vectors.append(F.normalize(F.normalize(f,dim=-1).mean(0),dim=-1))
            features.append(torch.stack(vectors))
        widths=[x.shape[1] for x in features]
        config={'widths':widths,'classes':len(vocab['raw_classes']),'sources':sources,'vocabulary_sha256':sha(ROOT/'data/vocabulary.json'),'manifest_sha256':sha(ROOT/'data/manifest.json'),'epochs':100,'train_images':len(train),'holdout_images':len(hold),'pooling':'每视图L2归一化后平均，再L2归一化','target':'原始标签集合均匀软目标，不声称完整病害负标签','holdout_use':'仅训练完成后的整批推理，禁止拟合或挑选轮次'}
        cfg=out/'configuration.json'
        if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('恢复时融合数据来源发生变化')
        dump(cfg,config);torch.manual_seed(20260906)
        model=Fusion(widths,len(vocab['raw_classes']))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01) if name=='adamw' else torch.optim.SGD(model.parameters(),lr=.02,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100)
        last=out/'last.pt';start=0
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);torch.set_rng_state(saved['rng']);start=saved['epoch']
        for epoch in range(start,100):
            model.train();order=train[torch.randperm(len(train))];loss_sum=0.
            for offset in range(0,len(order),64):
                ix=order[offset:offset+64];optimizer.zero_grad(set_to_none=True)
                logits,gates=model([x[ix] for x in features])
                loss=-(target[ix]*F.log_softmax(logits,-1)).sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('融合损失非有限')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();loss_sum+=float(loss.detach())
            scheduler.step();tmp=out/'last.pt.tmp'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'epoch':epoch+1},tmp);os.replace(tmp,last)
            dump(status,{'status':'training_fusion','optimizer':name,'epoch':epoch+1,'epochs':100,'loss_sum':loss_sum,'time':time.time()})
        model.eval();predictions=[];correct=0
        with torch.inference_mode():
            for offset in range(0,len(hold),64):
                ix=hold[offset:offset+64];logits,gates=model([x[ix] for x in features]);scores=logits.softmax(-1)
                correct+=int((target[ix,scores.argmax(-1)]>0).sum())
                for j,i in enumerate(ix.tolist()):predictions.append({'sample_id':rows[i]['sample_id'],'scores':scores[j].tolist(),'expert_weights':gates[j].tolist()})
        dump(out/'holdout_predictions.cloud_only.json',predictions)
        dump(out/'complete.json',{'status':'fusion_trained_whole_holdout_inferred_gateway_pending','checkpoint_sha256':sha(last),'holdout_images':len(hold),'top1_in_annotation_labelset':correct/len(hold),'time':time.time()})
    dump(status,{'status':'both_fusion_heads_complete_gateway_pending','time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'fusion_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
