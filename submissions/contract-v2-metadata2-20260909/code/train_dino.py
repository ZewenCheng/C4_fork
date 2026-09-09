"""DINOv3-H+多视图LoRA训练；两个优化器完整运行，之后整体保留集推理。"""
import os
from pathlib import Path
ROOT=Path('/workspace/work/c4-contract-entry-v3-20260908')
for key,rel in {'HF_HOME':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'cache','TMPDIR':'tmp','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','ALIPPU_CONFIG_PATH':'ppu','HGRTC_CACHE_PATH':'ppu/hgrtc'}.items():
    p=ROOT/'runtime'/rel;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
from concurrent.futures import ThreadPoolExecutor
import json
import math
import random
import gc
import time
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

SEED=20260906
EPOCHS=20
VIEWS=4
SIZE=512

def views(row):
    with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
    width,height=im.size
    full=ImageOps.pad(im,(SIZE,SIZE),color=(124,116,104))
    cw,ch=max(1,int(width*.7)),max(1,int(height*.7))
    crops=[im.crop((0,0,cw,ch)),im.crop((width-cw,height-ch,width,height))]
    variants=[full,ImageOps.mirror(full)]+[x.resize((SIZE,SIZE),Image.Resampling.BICUBIC) for x in crops]
    return np.stack([np.asarray(x,dtype=np.uint8) for x in variants])

class Expert(nn.Module):
    def __init__(self,base,n):
        super().__init__();self.backbone=base
        self.head=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,n))
    def forward(self,x):
        h=self.backbone(pixel_values=x).last_hidden_state
        f=torch.cat([h[:,0],h[:,5:].mean(1)],dim=-1).float()
        return self.head(f),f

def main():
    torch.set_num_threads(4)
    base_dir=ROOT/'models/dinov3-vith16plus-pretrain-lvd1689m'
    status=ROOT/'dino_status.json'
    # 等待实际发布文件完成；不重复下载，不以效果探针决定是否训练。
    while not (base_dir/'model.safetensors').exists() or not (ROOT/'data/manifest.json').exists():
        dump(status,{'status':'waiting_assets_or_manifest','time':time.time()})
        failure=ROOT/'download_status.json'
        if failure.exists() and json.loads(failure.read_text()).get('status')=='failed':raise RuntimeError('下载失败，待恢复')
        time.sleep(30)
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    vocab=json.loads((ROOT/'data/vocabulary.json').read_text())
    train=[r for r in manifest if r['split']=='train'];hold=[r for r in manifest if r['split']=='holdout']
    n=len(vocab['raw_classes']);device='cuda'
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    for opt_name in ['adamw','sgd_momentum']:
        out=ROOT/'checkpoints'/('dinov3_hplus_'+opt_name);out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        base=AutoModel.from_pretrained(base_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        base=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj','o_proj']))
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model=Expert(base,n).to(device)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=(torch.optim.AdamW(params,lr=1e-4,weight_decay=.01) if opt_name=='adamw' else torch.optim.SGD(params,lr=.01,momentum=.9,weight_decay=.0001))
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
        batch=8;start_epoch=0;start_offset=0;steps=0
        ckpt=out/'last.pt'
        if ckpt.exists():
            saved=torch.load(ckpt,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['trainable'],strict=False);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            start_epoch=saved['epoch'];start_offset=saved['offset'];steps=saved['steps'];batch=saved['batch']
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        dump(out/'configuration.json',{'backbone':str(base_dir),'base_manifest_sha256':sha(base_dir/'source_manifest.json'),'vocabulary_sha256':sha(ROOT/'data/vocabulary.json'),
             'data_manifest_sha256':sha(ROOT/'data/manifest.json'),'optimizer':opt_name,'epochs':EPOCHS,'views':VIEWS,'image_size':SIZE,
             'trainable_parameters':sum(p.numel() for p in params),'seed':SEED,'phase':'原始训练标签监督适配；7B蒸馏由独立后续阶段执行','validation':'完整隔离保留集推理，不作训练准入门'})
        def save(epoch,offset):
            tmp=out/'last.pt.tmp'
            torch.save({'trainable':{k:v.detach().cpu() for k,v in model.state_dict().items() if k in dict(model.named_parameters()) and dict(model.named_parameters())[k].requires_grad},
              'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'offset':offset,'steps':steps,'batch':batch,
              'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp)
            os.replace(tmp,ckpt)
        started=time.monotonic();processed=0
        with ThreadPoolExecutor(max_workers=12) as pool:
            for epoch in range(start_epoch,EPOCHS):
                order=np.random.default_rng(SEED+epoch).permutation(len(train)).tolist()
                offset=start_offset if epoch==start_epoch else 0
                model.train()
                while offset<len(order):
                    ids=order[offset:offset+batch];rows=[train[i] for i in ids]
                    pixels=np.stack(list(pool.map(views,rows))).reshape(-1,SIZE,SIZE,3)
                    target=torch.zeros((len(rows),n),device=device)
                    for j,row in enumerate(rows):target[j,row['label_ids']]=1/len(row['label_ids'])
                    try:
                        x=torch.from_numpy(pixels).permute(0,3,1,2).to(device=device,dtype=torch.float32)
                        x=(x/255-mean)/std
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            logits,feat=model(x)
                            logits=logits.float().view(len(rows),VIEWS,n)
                            # 图级监督在视图池化后计算；局部裁剪不强制拥有整图病害。
                            pooled=torch.logsumexp(logits,dim=1)-math.log(VIEWS)
                            loss=-(target*F.log_softmax(pooled,dim=-1)).sum(-1).mean()
                        if not torch.isfinite(loss):raise RuntimeError('训练损失非有限')
                        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
                        steps+=1;offset+=len(rows);processed+=len(rows)
                        torch.cuda.synchronize()
                        elapsed=time.monotonic()-started;rate=processed/max(elapsed,1)
                        progress={'status':'training','optimizer':opt_name,'epoch':epoch+1,'epochs':EPOCHS,'offset':offset,'train_images':len(train),'steps':steps,
                            'loss':float(loss.detach()),'image_batch':batch,'view_batch':batch*VIEWS,'images_per_second':rate,
                            'remaining_seconds_current_optimizer':((EPOCHS-epoch-1)*len(train)+len(train)-offset)/max(rate,1e-6),
                            'ppu_allocated_bytes':torch.cuda.memory_allocated(),'time':time.time()}
                        dump(status,progress)
                        if steps%10==0:print(json.dumps(progress,ensure_ascii=False),flush=True)
                        if steps%50==0:save(epoch,offset)
                        del x,logits,feat,pooled,loss
                    except torch.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True)
                        x=logits=feat=pooled=loss=None
                        gc.collect()
                        torch.cuda.empty_cache()
                        if batch<=1:raise
                        batch=max(1,batch//2)
                        print('本批显存不足，保留同批数据并降低图像batch至',batch,flush=True)
                scheduler.step();save(epoch+1,0)
        model.eval();pred=[]
        with torch.inference_mode(),ThreadPoolExecutor(max_workers=12) as pool:
            for offset in range(0,len(hold),batch):
                rows=hold[offset:offset+batch]
                pixels=np.stack(list(pool.map(views,rows))).reshape(-1,SIZE,SIZE,3)
                x=torch.from_numpy(pixels).permute(0,3,1,2).to(device=device,dtype=torch.float32)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model((x/255-mean)/std)
                scores=(torch.logsumexp(logits.float().view(len(rows),VIEWS,n),dim=1)-math.log(VIEWS)).softmax(-1).cpu().numpy()
                pred.extend({'sample_id':r['sample_id'],'scores':s.tolist(),'label_ids':r['label_ids']} for r,s in zip(rows,scores))
        dump(out/'holdout_predictions.cloud_only.json',pred)
        dump(out/'complete.json',{'status':'complete_label_adaptation','steps':steps,'holdout_images':len(pred),'raw_label_top1_hit':sum(int(np.argmax(r['scores'])) in r['label_ids'] for r in pred)/max(1,len(pred)),
             'checkpoint_sha256':sha(ckpt),'notes':'这是已有原始标签预测指标，非真实全部病害识别率或比赛总分。'})
        del model,base,optimizer,params;scheduler=None;torch.cuda.empty_cache()
    dump(status,{'status':'label_adaptation_complete_teacher_distillation_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'dino_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
