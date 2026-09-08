"""GroundingDINO与RT-DETRv2-X软区域监督适配；双优化器、断点与独立词表。"""
from prepare_regions import ROOT
import argparse,gc,json,os,random,time
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor,GroundingDinoForObjectDetection,RTDetrV2ForObjectDetection
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

SEED=20260906
def xyxy_to_cxcywh(box):
    return [(box[0]+box[2])/2,(box[1]+box[3])/2,box[2]-box[0],box[3]-box[1]]
def pair_iou(a,b):
    aa=torch.cat([a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2],-1)
    bb=torch.cat([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],-1)
    inter=(torch.minimum(aa[:,None,2:],bb[None,:,2:])-torch.maximum(aa[:,None,:2],bb[None,:,:2])).clamp_min(0).prod(-1)
    union=a[:,2:].clamp_min(0).prod(-1)[:,None]+b[:,2:].clamp_min(0).prod(-1)[None,:]-inter
    return inter/union.clamp_min(1e-6)
def candidates(regions,query):
    height,width=regions['original_size_hw'];result=[]
    gd=regions['grounding_dino']
    for box,score,label in zip(gd['boxes_xyxy'],gd['scores'],gd['labels']):
        if label and (query==label or query in label or label in query):
            b=[box[0]/width,box[1]/height,box[2]/width,box[3]/height]
            result.append((b,float(score),'grounding_dino'))
    result.extend((r['box_normalized_xyxy'],float(r['score']),'sam3') for r in regions['sam3'] if r['query']==query)
    kept=[]
    for b,score,source in sorted(result,key=lambda x:-x[1]):
        b=np.clip(b,0,1).tolist()
        if b[2]<=b[0] or b[3]<=b[1]:continue
        # 高重叠重复框合并；不同教师分歧保留为软监督，不按一致性冒称正确。
        duplicate=False
        for old in kept:
            inter=max(0,min(b[2],old[0][2])-max(b[0],old[0][0]))*max(0,min(b[3],old[0][3])-max(b[1],old[0][1]))
            union=(b[2]-b[0])*(b[3]-b[1])+(old[0][2]-old[0][0])*(old[0][3]-old[0][1])-inter
            if inter/max(union,1e-8)>.85:duplicate=True;break
        if not duplicate:kept.append((b,score,source))
    return kept[:12]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--expert',choices=['grounding_dino','rtdetr'],required=True);args=parser.parse_args()
    name=args.expert;status=ROOT/(name+'_training_status.json');epochs=10 if name=='grounding_dino' else 30
    while True:
        p=ROOT/'regions_status.json';d=json.loads(p.read_text()) if p.exists() else {}
        if d.get('status')=='complete':break
        dump(status,{'status':'waiting_full_training_region_cache','regions_completed':d.get('completed'),'time':time.time()});time.sleep(60)
    required_gib=22 if name=='grounding_dino' else 12
    while torch.cuda.mem_get_info()[0]<required_gib*1024**3:
        dump(status,{'status':'waiting_free_ppu','required_gib':required_gib,'time':time.time()});time.sleep(60)
    torch.cuda.set_per_process_memory_fraction(.22 if name=='grounding_dino' else .12);torch.set_num_threads(2)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='train']
    samples=[];queries=set()
    for row in rows:
        r=json.loads((ROOT/'pseudo_regions'/(row['sample_id']+'.json')).read_text())
        grouped={q:candidates(r,q) for q in r['queries']};grouped={q:v for q,v in grouped.items() if v}
        if grouped:samples.append((row,grouped));queries.update(grouped)
    vocab=sorted(queries)
    if not samples:raise RuntimeError('无区域软监督；不伪造检测框')
    items=[(row,{q:v}) for row,g in samples for q,v in g.items()] if name=='grounding_dino' else samples
    base_dir=ROOT/'models'/('grounding-dino-base' if name=='grounding_dino' else 'rtdetr_v2_r101vd')
    processor=AutoProcessor.from_pretrained(base_dir,local_files_only=True)
    batchsize=1 if name=='grounding_dino' else 4
    for optname in ['adamw','sgd_momentum']:
        out=ROOT/'checkpoints'/(name+'_'+optname);out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
        if name=='grounding_dino':
            model=GroundingDinoForObjectDetection.from_pretrained(base_dir,local_files_only=True,disable_custom_kernels=True,attn_implementation='eager')
        else:
            model=RTDetrV2ForObjectDetection.from_pretrained(base_dir,local_files_only=True,num_labels=len(vocab),id2label=dict(enumerate(vocab)),label2id={q:i for i,q in enumerate(vocab)},ignore_mismatched_sizes=True)
        embedding_module=model.model.denoising_class_embed if name=='rtdetr' else model.model.text_backbone.embeddings.word_embeddings
        model.get_input_embeddings=lambda _module=embedding_module:_module
        targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)] if name=='rtdetr' else 'all-linear'
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets)).to('cuda')
        if name=='rtdetr':
            for n,p in model.named_parameters():
                if any(k in n for k in ['class_embed','enc_score_head','denoising_class_embed']):p.requires_grad_(True)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=.01) if optname=='adamw' else torch.optim.SGD(params,lr=.002,momentum=.9,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs);epoch0=offset0=steps=0;last=out/'last.pt'
        if last.exists():
            s=torch.load(last,map_location='cpu',weights_only=False);model.load_state_dict(s['trainable'],strict=False);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
            epoch0=s['epoch'];offset0=s['offset'];steps=s['steps'];random.setstate(s['python_rng']);np.random.set_state(s['numpy_rng']);torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])
        names={n for n,p in model.named_parameters() if p.requires_grad}
        dump(out/'configuration.json',{'base':str(base_dir),'base_manifest_sha256':sha(base_dir/'source_manifest.json'),'pseudo_source_sha256':sha(ROOT/'pseudo_regions/source.json'),
            'optimizer':optname,'epochs':epochs,'batch':batchsize,'rank':8,'trainable_parameters':sum(p.numel() for p in params),
            'query_vocabulary':vocab,'extensible':True,'source':'SAM3与GroundingDINO训练分区候选，非人工真值','loss':'匹配候选的软置信度BCE、框L1、IoU；未匹配框不当作确认负例',
            'normalization':'冻结BatchNorm运行统计；训练权重检查点无需重建未保存的BN更新','validation':'组合整体推理另行执行'})
        def save(epoch,offset):
            temp=out/'last.pt.tmp'
            torch.save({'trainable':{k:v.detach().cpu() for k,v in model.state_dict().items() if k in names},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'epoch':epoch,'offset':offset,'steps':steps,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()},temp);os.replace(temp,last)
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        start=time.monotonic();processed=0
        for epoch in range(epoch0,epochs):
            order=np.random.default_rng(SEED+epoch).permutation(len(items)).tolist()
            for offset in range(offset0 if epoch==epoch0 else 0,len(order),batchsize):
                batch=[items[i] for i in order[offset:offset+batchsize]];images=[];targets=[]
                for row,grouped in batch:
                    with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                    flip=bool(random.randrange(2));im=ImageOps.mirror(im) if flip else im
                    images.append(ImageEnhance.Brightness(im).enhance(random.uniform(.85,1.15)))
                    boxes=[];labels=[];scores=[]
                    for query,values in grouped.items():
                        for box,confidence,_ in values:
                            b=xyxy_to_cxcywh(box)
                            if flip:b[0]=1-b[0]
                            boxes.append(b);labels.append(vocab.index(query));scores.append(confidence)
                    targets.append((torch.tensor(boxes,device='cuda',dtype=torch.float32),torch.tensor(labels,device='cuda'),torch.tensor(scores,device='cuda',dtype=torch.float32)))
                if name=='grounding_dino':
                    query=next(iter(batch[0][1]));inputs=processor(images=images,text=query+' .',return_tensors='pt').to('cuda')
                else:inputs=processor(images=images,return_tensors='pt').to('cuda')
                optimizer.zero_grad(set_to_none=True);prediction=model(**inputs);losses=[]
                for j,(boxes,labels,confidence) in enumerate(targets):
                    pred=prediction.pred_boxes[j].float()
                    if name=='grounding_dino':
                        length=int(inputs['attention_mask'][j].sum());classlogits=prediction.logits[j,:,1:max(2,length-2)].float().mean(-1,keepdim=True)
                        labels=torch.zeros_like(labels)
                    else:classlogits=prediction.logits[j].float()
                    with torch.no_grad():
                        cost=2*torch.cdist(pred,boxes,p=1)+2*(1-pair_iou(pred,boxes))-classlogits.sigmoid()[:,labels]
                        a,b=linear_sum_assignment(cost.cpu().numpy())
                    ii=torch.tensor(a,device='cuda');jj=torch.tensor(b,device='cuda');weight=confidence[jj]
                    cl=F.binary_cross_entropy_with_logits(classlogits[ii,labels[jj]],weight,reduction='none')
                    l1=(pred[ii]-boxes[jj]).abs().mean(1);iou=1-pair_iou(pred[ii],boxes[jj]).diagonal()
                    losses.append(((cl+2*l1+2*iou)*weight).sum()/weight.sum().clamp_min(1e-6))
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss):raise RuntimeError('检测损失非有限')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);optimizer.step();steps+=1;processed+=len(batch)
                if steps%50==0:save(epoch,offset+len(batch))
                if steps%10==0:
                    rate=processed/max(time.monotonic()-start,1);d={'status':'training','optimizer':optname,'epoch':epoch+1,'offset':offset+len(batch),'steps':steps,'loss':float(loss.detach()),
                        'items_per_second':rate,'remaining_seconds_current_optimizer':((epochs-epoch-1)*len(items)+len(items)-offset-len(batch))/max(rate,1e-6),'time':time.time()}
                    dump(status,d);print(json.dumps(d,ensure_ascii=False),flush=True)
                del inputs,prediction,loss,losses,cost
            scheduler.step();save(epoch+1,0)
        dump(out/'complete.json',{'status':'training_complete','steps':steps,'checkpoint_sha256':sha(last),'overall_inference':'pending','time':time.time()})
        del model,optimizer,params;scheduler=None;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_optimizers_trained_overall_inference_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:
        name=__import__('sys').argv[-1];dump(ROOT/(name+'_training_status.json'),{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
