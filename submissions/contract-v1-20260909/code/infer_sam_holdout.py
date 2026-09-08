"""训练后SAM完整保留图推理：固定训练词表提示，不读取保留图病害答案。"""
from prepare_regions import ROOT,prompts
import json,time,gc,os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from transformers import Sam3Model,Sam3Processor
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

def main():
    torch.set_num_threads(2)
    status=ROOT/'sam_inference_status.json'
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='holdout']
    vocabulary=ROOT/'data/vocabulary.json'
    queries=sorted({q for label in json.loads(vocabulary.read_text())['raw_classes'] for q in prompts(label)})
    directory=ROOT/'models/sam3'
    processor=Sam3Processor.from_pretrained(directory,local_files_only=True)
    for name in ['adamw','sgd_momentum']:
        source=ROOT/'checkpoints'/('sam3_'+name)
        out=ROOT/'gateway_outputs/sam_holdout'/name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        while not (source/'complete.json').exists():
            dump(status,{'status':'waiting_trained_sam','optimizer':name,'time':time.time()});time.sleep(60)
        config=json.loads((source/'configuration.json').read_text());checkpoint=source/'last.pt';digest=sha(checkpoint)
        assert digest==json.loads((source/'complete.json').read_text())['checkpoint_sha256']
        assert sha(directory/'source_manifest.json')==config['base_manifest_sha256']
        while torch.cuda.mem_get_info()[0]<22*1024**3:
            dump(status,{'status':'waiting_22gib_free_ppu','optimizer':name,'time':time.time()});time.sleep(60)
        torch.cuda.set_per_process_memory_fraction(.22)
        base=Sam3Model.from_pretrained(directory,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=config['lora_target_modules']))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert set(saved['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(saved['trainable'],strict=False);del saved
        model.to('cuda').eval();model.requires_grad_(False)
        source_record={'checkpoint_sha256':digest,'base_manifest_sha256':config['base_manifest_sha256'],'vocabulary_sha256':sha(vocabulary),'queries':queries,'mask_grid_hw':[256,256],'prompt_source':'全局训练词表，不使用保留图答案'}
        if (out/'source.json').exists():assert json.loads((out/'source.json').read_text())==source_record
        dump(out/'source.json',source_record);begin=time.monotonic();processed=0
        with torch.inference_mode():
            for i,row in enumerate(rows):
                target=out/(row['sample_id']+'.json')
                if target.exists():
                    old=json.loads(target.read_text());assert old['checkpoint_sha256']==digest
                    assert sha(out/old['mask_file'])==old['mask_sha256'];continue
                with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                candidates=[];packed=[]
                for query in queries:
                    inputs=processor(images=im,text=query,return_tensors='pt').to('cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(**inputs)
                    scores=prediction.pred_logits[0].float().sigmoid()
                    if prediction.presence_logits is not None:scores=scores*prediction.presence_logits[0].float().sigmoid().squeeze()
                    assert torch.isfinite(scores).all()
                    for index in scores.topk(min(3,len(scores))).indices.tolist():
                        mask=F.interpolate(prediction.pred_masks[0,index][None,None].float(),size=(256,256),mode='bilinear',align_corners=False)[0,0]>0
                        box=prediction.pred_boxes[0,index].float().clamp(0,1)
                        assert torch.isfinite(box).all()
                        candidates.append({'query':query,'uncalibrated_score':float(scores[index]),'box_normalized_xyxy':box.cpu().tolist(),'mask_index':len(packed),'mask_grid_area_pixels':int(mask.sum())})
                        packed.append(np.packbits(mask.cpu().numpy().reshape(-1)))
                    del inputs,prediction
                maskfile=out/(row['sample_id']+'.npz');tmp=maskfile.with_suffix('.tmp')
                with tmp.open('wb') as stream:np.savez_compressed(stream,masks=np.stack(packed))
                os.replace(tmp,maskfile)
                dump(target,{'sample_id':row['sample_id'],'checkpoint_sha256':digest,'original_size_hw':[im.height,im.width],'candidates':candidates,'mask_file':maskfile.name,'mask_sha256':sha(maskfile),'limitations':['软监督候选分数未校准','256网格仅供定位，不是原图像素或物理面积','未匹配查询没有负例监督，候选不能直接判定病害存在']})
                processed+=1
                dump(status,{'status':'extracting_sam_candidates','optimizer':name,'completed':i+1,'total':len(rows),'queries_per_image':len(queries),'images_per_second':processed/max(time.monotonic()-begin,1),'time':time.time()})
        dump(out/'complete.json',{'status':'whole_holdout_sam_candidates_complete_gateway_pending','images':len(rows),'checkpoint_sha256':digest,'time':time.time()})
        del model,base;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_sam_candidate_sets_complete_gateway_pending','time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'sam_inference_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
