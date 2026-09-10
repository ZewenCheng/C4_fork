"""独立进程执行原图与两裁块RT-DETR补证，输出统一原图坐标候选。"""
from pathlib import Path
import argparse,time
from report_contract import read_json,atomic_json,digest_file,digest_json,ContractError
from input_quality import load_views,map_box_to_original


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=read_json(args.request);row=request['row'];base_root=Path(request['asset_root'])
    if digest_file(row['path'])!=row['image_sha256']:raise ContractError('裁块输入变化')
    start=time.time()
    import torch
    from transformers import AutoProcessor,RTDetrV2ForObjectDetection
    from peft import LoraConfig,get_peft_model
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
    if torch.cuda.mem_get_info()[0]<12*1024**3:raise RuntimeError('裁块补证可用显存不足12GiB')
    cfg=read_json(base_root/'checkpoints/rtdetr_adamw/configuration.json');vocab=cfg['query_vocabulary']
    base=base_root/'models/rtdetr_v2_r101vd';processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=RTDetrV2ForObjectDetection.from_pretrained(base,local_files_only=True,num_labels=len(vocab),
        id2label=dict(enumerate(vocab)),label2id={q:i for i,q in enumerate(vocab)},ignore_mismatched_sizes=True)
    embed=model.model.denoising_class_embed;model.get_input_embeddings=lambda:embed
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)]))
    for n,p in model.named_parameters():
        if any(k in n for k in ['class_embed','enc_score_head','denoising_class_embed']):p.requires_grad_(True)
    saved=torch.load(base_root/'checkpoints/rtdetr_adamw/last.pt',map_location='cpu',weights_only=False)['trainable']
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    if set(saved)!=expected:raise ContractError('裁块模型训练参数覆盖不一致')
    loaded=model.load_state_dict(saved,strict=False)
    if loaded.unexpected_keys or expected.intersection(loaded.missing_keys):raise ContractError('裁块检查点未完整加载')
    model.to('cuda').eval();candidates=[];forward=0.;shapes=[]
    for image,view in load_views(row['path'],[[0,0,.7,.7],[.3,.3,1,1]]):
        inputs=processor(images=image,size={'height':640,'width':640},return_tensors='pt').to('cuda')
        torch.cuda.synchronize();tick=time.perf_counter()
        with torch.inference_mode():prediction=model(**inputs)
        torch.cuda.synchronize();forward+=time.perf_counter()-tick;shapes.append(list(inputs['pixel_values'].shape))
        scores,labels=prediction.logits[0].sigmoid().max(-1);values,ids=scores.topk(20)
        for box,label,score in zip(prediction.pred_boxes[0][ids].float().cpu().tolist(),labels[ids].cpu().tolist(),values.cpu().tolist()):
            x,y,w,h=box;b=[max(0,x-w/2),max(0,y-h/2),min(1,x+w/2),min(1,y+h/2)]
            if b[0]>=b[2] or b[1]>=b[3]:continue
            candidates.append({'query':vocab[label],'uncalibrated_score':score,'box_normalized_xyxy':map_box_to_original(b,view),
                'view_id':view['view_id'],'coordinate_space':'exif_oriented_original','measurement_level':'unscaled'})
    # 重叠只标记来源关系，不把候选聚类数称为真实病害实例数。
    def overlap(a,b):
        inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/max(union,1e-12)
    retained=[]
    for c in sorted(candidates,key=lambda x:-x['uncalibrated_score']):
        old=next((o for o in retained if o['query']==c['query'] and overlap(o['box_normalized_xyxy'],c['box_normalized_xyxy'])>.85),None)
        if old is None:retained.append({**c,'supporting_view_ids':[c['view_id']]})
        else:old['supporting_view_ids']=sorted(set(old['supporting_view_ids']+[c['view_id']]))
    atomic_json(args.output,{'status':'ok','result':{'sample_id':row['sample_id'],'outputs':{'adamw_crops':{
        'candidates':retained,'checkpoint_sha256':digest_file(base_root/'checkpoints/rtdetr_adamw/last.pt'),
        'preprocessing_sha256':digest_json({'sizes':shapes,'crops':[[0,0,.7,.7],[.3,.3,1,1]],'merge_iou':.85})}}},
        'input_sha256':row['image_sha256'],'model_calls':3,'runtime_seconds':time.time()-start,'synchronized_forward_seconds':forward,
        'before_merge_candidates':len(candidates),'tensor_shapes':shapes,'new_visual_observation':True,
        'limitations':['固定原图与两块70%局部，不是任意区域查询；合并后仍是同一模型的候选，不是独立投票或病害实例真值']})


if __name__=='__main__':main()
