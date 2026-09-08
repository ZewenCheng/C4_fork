"""H+已有监督适配后接7B冻结教师关系蒸馏；不同特征维度通过关系矩阵对齐。"""
from train_dino import ROOT,Expert,views
import gc,json,math,os,random,time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

SEED=20260906
EPOCHS=5
BATCH=8
def relation_loss(student,teacher):
    s=F.normalize(student.float(),dim=-1);t=F.normalize(teacher.float(),dim=-1)
    ss=s@s.transpose(-2,-1)/.1;tt=t@t.transpose(-2,-1)/.1
    diagonal=torch.eye(ss.shape[-1],device=ss.device,dtype=torch.bool)
    ss=ss.masked_fill(diagonal,-1e4);tt=tt.masked_fill(diagonal,-1e4)
    return F.kl_div(F.log_softmax(ss,dim=-1),F.softmax(tt,dim=-1),reduction='none').sum(-1).mean()
def main():
    status=ROOT/'dino_distillation_status.json';torch.set_num_threads(3)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='train']
    vocab=json.loads((ROOT/'data/vocabulary.json').read_text())
    for optname in ['adamw','sgd_momentum']:
        parent=ROOT/'checkpoints'/('dinov3_hplus_'+optname)
        out=ROOT/'checkpoints'/('dinov3_hplus_distilled_'+optname);out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        while True:
            p=ROOT/'teacher_features_status.json';s=json.loads(p.read_text()) if p.exists() else {}
            if (parent/'complete.json').exists() and s.get('status')=='complete':break
            dump(status,{'status':'waiting_supervised_adapter_and_teacher_features','optimizer':optname,'time':time.time()});time.sleep(60)
        while torch.cuda.mem_get_info()[0]<18*1024**3:
            dump(status,{'status':'waiting_18gib_free_ppu','optimizer':optname,'time':time.time()});time.sleep(60)
        torch.cuda.set_per_process_memory_fraction(.2)
        torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
        base=AutoModel.from_pretrained(ROOT/'models/dinov3-vith16plus-pretrain-lvd1689m',local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        base=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj','o_proj']))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model=Expert(base,len(vocab['raw_classes'])).to('cuda')
        initial=torch.load(parent/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(initial['trainable'],strict=False);del initial
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01) if optname=='adamw' else torch.optim.SGD(params,lr=.002,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS);epoch0=offset0=steps=0;last=out/'last.pt'
        if last.exists():
            s=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(s['trainable'],strict=False);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
            epoch0=s['epoch'];offset0=s['offset'];steps=s['steps'];random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng']);del s
        names={n for n,p in model.named_parameters() if p.requires_grad}
        dump(out/'configuration.json',{'parent_checkpoint_sha256':sha(parent/'last.pt'),'teacher_features_source_sha256':sha(ROOT/'teacher_features/source.json'),
          'data_manifest_sha256':sha(ROOT/'data/manifest.json'),'optimizer':optname,'epochs':EPOCHS,'views':2,'image_batch':BATCH,
          'loss':'0.5原标签分类+全局跨图关系KL+4x4局部关系KL，teacher特征完全冻结','phase':'在20轮标签监督适配后增加5轮教师蒸馏，保留原适配版本供完整组合比较'})
        def save(epoch,offset):
            tmp=out/'last.pt.tmp';torch.save({'trainable':{k:v.detach().cpu() for k,v in model.state_dict().items() if k in names},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'epoch':epoch,'offset':offset,'steps':steps,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp);os.replace(tmp,last)
        mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
        model.train();start=time.monotonic();processed=0
        for epoch in range(epoch0,EPOCHS):
            order=np.random.default_rng(SEED+epoch).permutation(len(rows)).tolist()
            for offset in range(offset0 if epoch==epoch0 else 0,len(order),BATCH):
                batch=[rows[i] for i in order[offset:offset+BATCH]]
                array=np.stack([views(r)[:2] for r in batch]).reshape(-1,512,512,3)
                glob=[];local=[]
                for r in batch:
                    with np.load(ROOT/'teacher_features'/(r['sample_id']+'.npz')) as data:glob.append(data['global_features']);local.append(data['local_features'])
                tg=torch.from_numpy(np.stack(glob).reshape(len(batch)*2,-1)).to('cuda',dtype=torch.float32)
                tl=torch.from_numpy(np.stack(local).reshape(len(batch)*2,16,-1)).to('cuda',dtype=torch.float32)
                target=torch.zeros((len(batch),len(vocab['raw_classes'])),device='cuda')
                for j,r in enumerate(batch):target[j,r['label_ids']]=1/len(r['label_ids'])
                x=torch.from_numpy(array).permute(0,3,1,2).to('cuda',dtype=torch.float32);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    h=model.backbone(pixel_values=(x/255-mean)/std).last_hidden_state
                    f=torch.cat([h[:,0],h[:,5:].mean(1)],-1).float();logits=model.head(f).float().reshape(len(batch),2,-1)
                    pooled=torch.logsumexp(logits,dim=1)-math.log(2)
                    classification=-(target*F.log_softmax(pooled,-1)).sum(-1).mean()
                    sl=h[:,5:].reshape(len(batch)*2,4,8,4,8,-1).mean((2,4)).reshape(len(batch)*2,16,-1)
                    loss=.5*classification+relation_loss(f,tg)+relation_loss(sl,tl)
                if not torch.isfinite(loss):raise RuntimeError('蒸馏损失非有限')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();steps+=1;processed+=len(batch)
                if steps%50==0:save(epoch,offset+len(batch))
                if steps%10==0:
                    rate=processed/max(time.monotonic()-start,1);d={'status':'distilling','optimizer':optname,'epoch':epoch+1,'offset':offset+len(batch),'steps':steps,'loss':float(loss.detach()),
                       'remaining_seconds_current_optimizer':((EPOCHS-epoch-1)*len(rows)+len(rows)-offset-len(batch))/max(rate,1e-6),'time':time.time()};dump(status,d);print(json.dumps(d,ensure_ascii=False),flush=True)
                del h,f,logits,pooled,sl,loss,x,tg,tl
            scheduler.step();save(epoch+1,0)
        dump(out/'complete.json',{'status':'distillation_complete','checkpoint_sha256':sha(last),'overall_inference':'pending','steps':steps,'time':time.time()})
        del model,base,optimizer,params;scheduler=None;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_optimizers_distilled_overall_inference_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'dino_distillation_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
