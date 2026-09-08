"""检测适配完成后恢复全部训练参数，生成完整保留集框候选，限平台留存。"""
import sys,json,time,gc
from pathlib import Path
sys.path.insert(0,'/workspace/work/c4-contract-entry-v3-20260908')
import torch
from PIL import Image,ImageOps
from transformers import AutoProcessor,RTDetrV2ForObjectDetection
from peft import LoraConfig,get_peft_model
from download_assets import ROOT,dump,sha

def main():
    torch.set_num_threads(2);status=ROOT/'rtdetr_inference_status.json'
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='holdout']
    directory=ROOT/'models/rtdetr_v2_r101vd';processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
    for name in ['adamw','sgd_momentum']:
        source=ROOT/'checkpoints'/('rtdetr_'+name);out=ROOT/'gateway_outputs/rtdetr_holdout'/name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        while not (source/'complete.json').exists():
            dump(status,{'status':'waiting_trained_detector','optimizer':name,'time':time.time()});time.sleep(60)
        config=json.loads((source/'configuration.json').read_text());vocab=config['query_vocabulary']
        checkpoint=source/'last.pt';digest=sha(checkpoint)
        assert digest==json.loads((source/'complete.json').read_text())['checkpoint_sha256']
        assert sha(directory/'source_manifest.json')==config['base_manifest_sha256']
        while torch.cuda.mem_get_info()[0]<10*1024**3:
            dump(status,{'status':'waiting_10gib_free_ppu','optimizer':name,'time':time.time()});time.sleep(60)
        torch.cuda.set_per_process_memory_fraction(.09)
        base=RTDetrV2ForObjectDetection.from_pretrained(directory,local_files_only=True,num_labels=len(vocab),id2label=dict(enumerate(vocab)),label2id={q:i for i,q in enumerate(vocab)},ignore_mismatched_sizes=True)
        embedding=base.model.denoising_class_embed;base.get_input_embeddings=lambda:embedding
        targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear)]
        model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets))
        for n,p in model.named_parameters():
            if any(k in n for k in ['class_embed','enc_score_head','denoising_class_embed']):p.requires_grad_(True)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert set(saved['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(saved['trainable'],strict=False);del saved
        model.to('cuda').eval();model.requires_grad_(False);begin=time.monotonic();processed=0
        with torch.inference_mode():
            for i,row in enumerate(rows):
                target=out/(row['sample_id']+'.json')
                if target.exists():
                    assert json.loads(target.read_text())['checkpoint_sha256']==digest
                    continue
                with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                inputs=processor(images=im,return_tensors='pt').to('cuda')
                prediction=model(**inputs);scores,labels=prediction.logits[0].sigmoid().max(-1)
                values,indices=scores.topk(min(20,len(scores)));boxes=prediction.pred_boxes[0][indices]
                xyxy=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1).clamp(0,1)
                candidates=[{'query':vocab[int(labels[j])],'uncalibrated_score':float(v),'box_normalized_xyxy':b.tolist()} for v,j,b in zip(values,indices,xyxy)]
                dump(target,{'sample_id':row['sample_id'],'checkpoint_sha256':digest,'base_manifest_sha256':config['base_manifest_sha256'],'expert':'rtdetr_v2_r101vd','optimizer':name,'original_size_hw':[im.height,im.width],'candidates':candidates,'evidence_type':'ranked_box_candidates','limitations':['仅软伪标签监督，未匹配查询无负例约束，分数可能偏高','每图最多20个候选是输出预算，不是病害数量或阈值验收','框面积不是病害掩码面积，不输出物理尺寸或等级']})
                processed+=1;del inputs,prediction
                if processed%10==0:dump(status,{'status':'extracting_detector_candidates','optimizer':name,'completed':i+1,'total':len(rows),'images_per_second':processed/max(time.monotonic()-begin,1),'time':time.time()})
        dump(out/'complete.json',{'status':'whole_holdout_candidates_complete_gateway_pending','images':len(rows),'checkpoint_sha256':digest,'evaluation':'无独立框真值，不报告mAP','time':time.time()})
        del model,base;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_detector_candidate_sets_complete_gateway_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'rtdetr_inference_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
