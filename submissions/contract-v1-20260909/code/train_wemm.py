"""WeMM-9B图文双视图LoRA训练；使用已审阅官方embedding方法，教师与Qwen不更新。"""
from prepare_regions import ROOT
import gc,json,os,random,time
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoProcessor
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

SEED=20260906
EPOCHS=3
BATCH=2
def main():
    status=ROOT/'wemm_training_status.json'
    while True:
        p=ROOT/'wemm_download_status.json';d=json.loads(p.read_text()) if p.exists() else {}
        if d.get('status')=='complete':break
        dump(status,{'status':'waiting_verified_wemm_weights','download_status':d.get('status'),'time':time.time()});time.sleep(60)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.48)
    directory=ROOT/'models/WeMM-Embedding-9B'
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='train']
    processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
    processor.tokenizer.padding_side='right'
    embedding_id=processor.tokenizer.convert_tokens_to_ids('<embedding>')
    if embedding_id is None or embedding_id==processor.tokenizer.unk_token_id:raise ValueError('缺少官方embedding token')
    for optname in ['adamw','sgd_momentum']:
        out=ROOT/'checkpoints'/('wemm9b_'+optname);out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        base=AutoModel.from_pretrained(directory,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        base.config.use_cache=False
        model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj','o_proj','in_proj_qkv'])).to('cuda')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01) if optname=='adamw' else torch.optim.SGD(params,lr=.002,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
        epoch0=offset0=steps=0;last=out/'last.pt'
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(saved['trainable'],strict=False)
            optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);epoch0=saved['epoch'];offset0=saved['offset'];steps=saved['steps']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        names={n for n,p in model.named_parameters() if p.requires_grad}
        dump(out/'configuration.json',{'base':str(directory),'base_manifest_sha256':sha(directory/'source_manifest.json'),'reviewed_embedding_code_sha256':sha(directory/'modeling_wemm_embedding.py'),
            'data_manifest_sha256':sha(ROOT/'data/manifest.json'),'optimizer':optname,'epochs':EPOCHS,'image_batch':BATCH,'views':2,'max_image_pixels':512*512,
            'text_limit_chars':1200,'embedding_dimension':4096,'rank':8,'trainable_parameters':sum(p.numel() for p in params),
            'loss':'双向图文多正例对比，相同已有标签不作为相互负例；同图双视图一致性','evaluation':'完整组合推理待独立执行'})
        def save(epoch,offset):
            tmp=out/'last.pt.tmp'
            torch.save({'trainable':{k:v.detach().cpu() for k,v in model.state_dict().items() if k in names},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'epoch':epoch,'offset':offset,'steps':steps,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp)
            os.replace(tmp,last)
        def embed(inputs):
            inputs=inputs.to('cuda')
            end=inputs['attention_mask'].sum(-1)-1
            if not torch.all(inputs['input_ids'][torch.arange(len(end),device='cuda'),end]==embedding_id):raise ValueError('池化位置不是embedding token')
            return model.get_base_model().embedding(**inputs,use_cache=False).float()
        start=time.monotonic();processed=0;model.train()
        for epoch in range(epoch0,EPOCHS):
            order=np.random.default_rng(SEED+epoch).permutation(len(rows)).tolist()
            for offset in range(offset0 if epoch==epoch0 else 0,len(order),BATCH):
                batch=[rows[i] for i in order[offset:offset+BATCH]];images=[];texts=[]
                for row in batch:
                    with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                    im.thumbnail((512,512),Image.Resampling.LANCZOS)
                    images.extend([im,ImageEnhance.Brightness(im).enhance(random.uniform(.85,1.15))])
                    a=row['annotations'][0]
                    description='；'.join(str(a.get(k,'')) for k in ['questionCategory','defectLocation','defectType','defectDescription'])[:1200]
                    texts.append('<|im_start|>user\n'+description+'<|im_end|><embedding>')
                itext=['<|im_start|>user<|vision_start|><|image_pad|><|vision_end|><|im_end|><embedding>']*len(images)
                iinputs=processor(images=images,text=itext,padding=True,return_tensors='pt')
                tinputs=processor(text=texts,padding=True,return_tensors='pt')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    iviews=embed(iinputs).reshape(len(batch),2,-1);ie=F.normalize(iviews.mean(1),dim=-1);te=embed(tinputs)
                    sim=ie@te.T/.07
                    positive=torch.tensor([[bool(set(a['label_ids'])&set(b['label_ids'])) for b in batch] for a in batch],device='cuda',dtype=torch.float32)
                    target=positive/positive.sum(1,keepdim=True)
                    loss=-.5*((target*sim.log_softmax(1)).sum(1).mean()+(target*sim.T.log_softmax(1)).sum(1).mean())
                    loss=loss+.1*(1-F.cosine_similarity(iviews[:,0],iviews[:,1])).mean()
                if not torch.isfinite(loss):raise RuntimeError('WeMM损失非有限')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();steps+=1;processed+=len(batch)
                if steps%25==0:save(epoch,offset+len(batch))
                if steps%5==0:
                    rate=processed/max(time.monotonic()-start,1)
                    state={'status':'training','optimizer':optname,'epoch':epoch+1,'offset':offset+len(batch),'steps':steps,'loss':float(loss.detach()),'images_per_second':rate,
                           'remaining_seconds_current_optimizer':((EPOCHS-epoch-1)*len(rows)+len(rows)-offset-len(batch))/max(rate,1e-6),'time':time.time()}
                    dump(status,state);print(json.dumps(state,ensure_ascii=False),flush=True)
                del iinputs,tinputs,iviews,ie,te,sim,loss
            scheduler.step();save(epoch+1,0)
        dump(out/'complete.json',{'status':'training_complete','steps':steps,'checkpoint_sha256':sha(last),'overall_inference':'pending','time':time.time()})
        del model,base,params,optimizer;scheduler=None;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_optimizers_trained_overall_inference_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'wemm_training_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
