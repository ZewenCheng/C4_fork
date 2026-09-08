"""SAM3解码路径LoRA双优化器训练；仅匹配来源可追踪的软区域候选。"""
from prepare_regions import ROOT
import gc,json,math,os,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import Sam3Model,Sam3Processor
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

EPOCHS=10
SEED=20260906
def main():
    status=ROOT/'sam_training_status.json'
    while True:
        p=ROOT/'regions_status.json';d=json.loads(p.read_text()) if p.exists() else {}
        if d.get('status')=='complete':break
        dump(status,{'status':'waiting_full_training_region_cache','regions_completed':d.get('completed'),'time':time.time()})
        time.sleep(60)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    records=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='train']
    directory=ROOT/'pseudo_regions';items=[]
    for row in records:
        regions=json.loads((directory/(row['sample_id']+'.json')).read_text())
        for query in sorted({r['query'] for r in regions['sam3']}):
            items.append((row,regions,query))
    if not items:raise RuntimeError('无可训练软区域，保留图级训练，不能伪造掩码')
    base_dir=ROOT/'models/sam3';processor=Sam3Processor.from_pretrained(base_dir,local_files_only=True)
    for optname in ['adamw','sgd_momentum']:
        out=ROOT/'checkpoints'/('sam3_'+optname);out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        model=Sam3Model.from_pretrained(base_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        targets=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear) and name.startswith(('geometry_encoder.','detr_encoder.','detr_decoder.','mask_decoder.'))]
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets)).to('cuda')
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01) if optname=='adamw' else torch.optim.SGD(params,lr=.002,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
        epoch0=offset0=steps=0;last=out/'last.pt'
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(saved['trainable'],strict=False)
            optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            epoch0=saved['epoch'];offset0=saved['offset'];steps=saved['steps']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        dump(out/'configuration.json',{'model':str(base_dir),'base_manifest_sha256':sha(base_dir/'source_manifest.json'),'pseudo_source_sha256':sha(directory/'source.json'),
             'optimizer':optname,'epochs':EPOCHS,'train_image_queries':len(items),'lora_target_modules':targets,'rank':8,
             'trainable_parameters':sum(p.numel() for p in params),'loss':'匹配区域的软置信度校准、掩码BCE+Dice、归一化边框L1；未匹配查询不标阴性',
             'augmentation':'亮度与对比度双视图交替，不改变已缓存区域的坐标系','final_evaluation':'组合专家整体推理单独执行'})
        names={n for n,p in model.named_parameters() if p.requires_grad}
        def save(epoch,offset):
            tmp=out/'last.pt.tmp'
            torch.save({'trainable':{k:v.detach().cpu() for k,v in model.state_dict().items() if k in names},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'epoch':epoch,'offset':offset,'steps':steps,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},tmp)
            os.replace(tmp,last)
        model.train();start=time.monotonic();processed=0
        for epoch in range(epoch0,EPOCHS):
            order=np.random.default_rng(SEED+epoch).permutation(len(items)).tolist()
            for offset in range(offset0 if epoch==epoch0 else 0,len(order)):
                row,regions,query=items[order[offset]]
                with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                factor=random.uniform(.85,1.15)
                im=ImageEnhance.Brightness(im).enhance(factor) if (epoch+offset)%2 else ImageEnhance.Contrast(im).enhance(factor)
                selected=[r for r in regions['sam3'] if r['query']==query]
                with np.load(directory/regions['mask_file']) as packed:
                    masks=np.stack([np.unpackbits(packed['masks'][r['mask_index']]).reshape(256,256) for r in selected])
                target=torch.from_numpy(masks).to('cuda',dtype=torch.float32)
                boxes=torch.tensor([r['box_normalized_xyxy'] for r in selected],device='cuda',dtype=torch.float32)
                confidence=torch.tensor([r['score'] for r in selected],device='cuda',dtype=torch.float32)
                inputs=processor(images=im,text=query,return_tensors='pt').to('cuda');optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(**inputs)
                logits=F.interpolate(prediction.pred_masks[0][:,None].float(),size=(256,256),mode='bilinear',align_corners=False)[:,0]
                pflat=logits.sigmoid().flatten(1);tflat=target.flatten(1)
                with torch.no_grad():
                    intersection=pflat@tflat.T;dice=(2*intersection+1)/(pflat.sum(1)[:,None]+tflat.sum(1)[None]+1)
                    cost=1-dice+torch.cdist(prediction.pred_boxes[0].float(),boxes,p=1)
                    idx,jdx=linear_sum_assignment(cost.float().cpu().numpy())
                ii=torch.tensor(idx,device='cuda');jj=torch.tensor(jdx,device='cuda')
                matching=logits[ii];truth=target[jj];weight=confidence[jj]
                bce=F.binary_cross_entropy_with_logits(matching,truth,reduction='none').mean((1,2))
                prob=matching.sigmoid();dice_loss=1-(2*(prob*truth).sum((1,2))+1)/(prob.sum((1,2))+truth.sum((1,2))+1)
                boxloss=(prediction.pred_boxes[0,ii].float()-boxes[jj]).abs().mean(1)
                score=prediction.pred_logits[0,ii].float().sigmoid()
                if prediction.presence_logits is not None:score=score*prediction.presence_logits[0].float().sigmoid().squeeze()
                calibration=F.binary_cross_entropy(score.clamp(1e-6,1-1e-6),weight,reduction='none')
                loss=((bce+dice_loss+boxloss+.2*calibration)*weight).sum()/weight.sum().clamp_min(1e-6)
                if not torch.isfinite(loss):raise RuntimeError('SAM损失非有限')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();steps+=1;processed+=1
                if steps%50==0:save(epoch,offset+1)
                if steps%10==0:
                    rate=processed/max(time.monotonic()-start,1)
                    state={'status':'training','optimizer':optname,'epoch':epoch+1,'offset':offset+1,'queries_per_epoch':len(items),'steps':steps,'loss':float(loss.detach()),
                        'queries_per_second':rate,'remaining_seconds_current_optimizer':((EPOCHS-epoch-1)*len(items)+len(items)-offset-1)/max(rate,1e-6),'time':time.time()}
                    dump(status,state);print(json.dumps(state,ensure_ascii=False),flush=True)
                del prediction,inputs,logits,pflat,tflat,loss,matching,truth,prob,cost
            scheduler.step();save(epoch+1,0)
        dump(out/'complete.json',{'status':'training_complete','steps':steps,'checkpoint_sha256':sha(last),'overall_inference':'pending','time':time.time()})
        del model,optimizer,params;scheduler=None;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_optimizers_trained_overall_inference_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'sam_training_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
