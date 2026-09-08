"""训练分区的GroundingDINO/SAM3候选建库；标签指导提示，候选不冒充真值。"""
import os
import sys
import json
import time
from pathlib import Path
ROOT=Path('/workspace/work/c4-contract-entry-v3-20260908')
for key,rel in {'HF_HOME':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'cache','TMPDIR':'tmp','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','ALIPPU_CONFIG_PATH':'ppu','HGRTC_CACHE_PATH':'ppu/hgrtc'}.items():
    p=ROOT/'runtime'/rel;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from transformers import AutoProcessor,GroundingDinoForObjectDetection,Sam3Model,Sam3Processor
from download_assets import dump,sha

def prompts(label):
    pairs=[('裂缝','crack'),('锈','rust'),('渗水','water stain'),('水痕','water stain'),('泛碱','efflorescence'),
      ('剥落','spalled concrete'),('破损','damaged concrete'),('钢筋','exposed reinforcing steel'),('苔藓','moss'),('植被','vegetation'),
      ('粉红','pink stain'),('变色','discoloration'),('污','stain'),('修补','repaired concrete'),('麻面','rough concrete'),
      ('蜂窝','concrete honeycombing'),('接缝','joint'),('伸缩缝','expansion joint'),('划痕','scratch'),('支座','bridge bearing')]
    values=sorted({v for k,v in pairs if k in label})
    return values or ['concrete surface']

def main():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    out=ROOT/'pseudo_regions';out.mkdir(exist_ok=True)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='train']
    gd_dir=ROOT/'models/grounding-dino-base';sam_dir=ROOT/'models/sam3'
    gp=AutoProcessor.from_pretrained(gd_dir,local_files_only=True)
    gd=GroundingDinoForObjectDetection.from_pretrained(gd_dir,local_files_only=True,disable_custom_kernels=True,attn_implementation='eager').to('cuda').eval()
    sp=Sam3Processor.from_pretrained(sam_dir,local_files_only=True)
    sam=Sam3Model.from_pretrained(sam_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    for model in [gd,sam]:model.requires_grad_(False)
    source={'grounding_dino':sha(gd_dir/'source_manifest.json'),'sam3':sha(sam_dir/'source_manifest.json'),
            'data_sha256':sha(ROOT/'data/manifest.json'),'supervision':'训练标签仅作文本提示；无人工框或掩码；这是软候选。','mask_storage':'256x256概率阈值掩码压缩，不作为毫米级测量真值。'}
    dump(out/'source.json',source)
    started=time.monotonic();done=0
    for row in rows:
        target=out/(row['sample_id']+'.json')
        if target.exists():done+=1;continue
        with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
        queries=sorted({p for a in row['annotations'] for p in prompts(str(a['defectType']))})
        gtext=' . '.join(queries)+' .'
        with torch.inference_mode():
            inputs=gp(images=im,text=gtext,return_tensors='pt').to('cuda')
            gout=gd(**inputs)
            gr=gp.post_process_grounded_object_detection(gout,inputs['input_ids'],threshold=.15,text_threshold=.2,target_sizes=[(im.height,im.width)])[0]
            boxes=gr['boxes'].cpu().tolist();scores=gr['scores'].cpu().tolist();labels=gr.get('text_labels',gr.get('labels',[]))
            del inputs,gout
            regions=[];packed=[]
            for query in queries:
                sinputs=sp(images=im,text=query,return_tensors='pt').to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):sout=sam(**sinputs)
                score=sout.pred_logits[0].float().sigmoid()
                if sout.presence_logits is not None:score=score*sout.presence_logits[0].float().sigmoid().squeeze()
                ids=torch.topk(score,min(3,len(score))).indices
                for idx in ids.tolist():
                    confidence=float(score[idx])
                    if confidence<.15:continue
                    mask=F.interpolate(sout.pred_masks[0,idx][None,None].float(),size=(256,256),mode='bilinear',align_corners=False)[0,0]>0
                    packed.append(np.packbits(mask.cpu().numpy().reshape(-1)))
                    regions.append({'query':query,'score':confidence,'box_normalized_xyxy':sout.pred_boxes[0,idx].float().cpu().tolist(),'mask_index':len(packed)-1})
                del sout,sinputs
        maskfile=out/(row['sample_id']+'.npz')
        temp=maskfile.with_name(maskfile.name+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,masks=np.stack(packed) if packed else np.zeros((0,8192),np.uint8))
        os.replace(temp,maskfile)
        dump(target,{'sample_id':row['sample_id'],'original_size_hw':[im.height,im.width],'queries':queries,
            'grounding_dino':{'boxes_xyxy':boxes,'scores':scores,'labels':list(labels)},'sam3':regions,
            'text_visual_candidate_missing':not boxes and not regions,'mask_file':maskfile.name,'teacher_frozen':True})
        done+=1;elapsed=time.monotonic()-started
        state={'status':'generating_training_candidates','completed':done,'total':len(rows),'elapsed_seconds':elapsed,'time':time.time()}
        dump(ROOT/'regions_status.json',state)
        if done%25==0:print(json.dumps(state,ensure_ascii=False),flush=True)
    dump(ROOT/'regions_status.json',{'status':'complete','completed':done,'total':len(rows),'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'regions_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
