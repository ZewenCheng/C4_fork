"""同两张固定无标签输入，双优化器逐候选核验图像/文本编码复用；原始数组只留平台。"""
import time
begin_time=time.time()
import gc,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'benchmark';OUT.mkdir(exist_ok=True)
BASE=Path('/workspace/work/c4-experiments/expert-autotrain-20260906')
for key,name in {'HF_HOME':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'cache','TMPDIR':'tmp','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','ALIPPU_CONFIG_PATH':'ppu','HGRTC_CACHE_PATH':'ppu/hgrtc'}.items():
    p=OUT/'runtime'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from peft import LoraConfig,get_peft_model
from transformers import Sam3Model,Sam3Processor,AutoProcessor,GroundingDinoForObjectDetection
import infer_sam_holdout as sam
import infer_grounding_holdout as grounding
from report_contract import atomic_json,digest_file,digest_json,read_json
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.22)

def record(phase,**more):atomic_json(OUT/'status.json',{'status':phase,'time':time.time(),**more})

def sam_result(query,prediction):
    scores=prediction.pred_logits[0].float().sigmoid()
    if prediction.presence_logits is not None:scores=scores*prediction.presence_logits[0].float().sigmoid().squeeze()
    assert torch.isfinite(scores).all()
    candidates=[];masks=[]
    for index in scores.topk(min(3,len(scores))).indices.tolist():
        mask=F.interpolate(prediction.pred_masks[0,index][None,None].float(),size=(256,256),mode='bilinear',align_corners=False)[0,0]>0
        box=prediction.pred_boxes[0,index].float().clamp(0,1)
        candidates.append({'query':query,'score':float(scores[index]),'box':box.cpu().tolist(),'mask_area':int(mask.sum())})
        masks.append(np.packbits(mask.cpu().numpy().reshape(-1)))
    return candidates,masks

def grounding_result(query,inputs,prediction):
    length=int(inputs['attention_mask'][0].sum())
    scores=prediction.logits[0,:,1:max(2,length-2)].float().mean(-1).sigmoid()
    assert torch.isfinite(scores).all()
    values,indices=scores.topk(min(3,len(scores)));boxes=prediction.pred_boxes[0,indices].float()
    xyxy=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1).clamp(0,1)
    return [{'query':query,'score':float(v),'box':b.cpu().tolist()} for v,b in zip(values,xyxy)],[]

def load(kind,name):
    if kind=='sam':
        directory=BASE/'models/sam3';source=BASE/'checkpoints'/('sam3_'+name)
        config=read_json(source/'configuration.json')
        base=Sam3Model.from_pretrained(directory,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        processor=Sam3Processor.from_pretrained(directory,local_files_only=True)
        lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=config['lora_target_modules'])
        queries=sorted({q for label in read_json(BASE/'data/vocabulary.json')['raw_classes'] for q in sam.prompts(label)})
    else:
        directory=BASE/'models/grounding-dino-base';source=BASE/'checkpoints'/('grounding_dino_'+name)
        config=read_json(source/'configuration.json')
        base=GroundingDinoForObjectDetection.from_pretrained(directory,local_files_only=True,disable_custom_kernels=True,attn_implementation='eager')
        embedding=base.model.text_backbone.embeddings.word_embeddings;base.get_input_embeddings=lambda:embedding
        processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
        lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules='all-linear')
        queries=config['query_vocabulary']
    checkpoint=source/'last.pt'
    assert digest_file(checkpoint)==read_json(source/'complete.json')['checkpoint_sha256']
    assert digest_file(directory/'source_manifest.json')==config['base_manifest_sha256']
    model=get_peft_model(base,lora);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert set(saved['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(saved['trainable'],strict=False);del saved
    model.to('cuda').eval();model.requires_grad_(False)
    return model,processor,queries

def measure(kind,name,model,processor,queries,images,optimized,label):
    record(kind+'_'+name+'_'+label)
    torch.cuda.synchronize();started=time.perf_counter();all_candidates=[];all_masks=[]
    with torch.inference_mode():
        cache=sam.prepare_text_cache(model,processor,queries) if kind=='sam' and optimized else None
        for image in images:
            candidates=[];masks=[]
            if optimized:
                iterator=sam.predict_queries(model,processor,image,queries,cache) if kind=='sam' else grounding.predict_queries(model,processor,image,queries)
                for value in iterator:
                    c,m=sam_result(*value) if kind=='sam' else grounding_result(*value)
                    candidates.extend(c);masks.extend(m)
            else:
                for query in queries:
                    inputs=processor(images=image if kind=='sam' else [image],text=query if kind=='sam' else query+' .',return_tensors='pt').to('cuda')
                    if kind=='sam':
                        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(**inputs)
                        c,m=sam_result(query,prediction)
                    else:c,m=grounding_result(query,inputs,model(**inputs))
                    candidates.extend(c);masks.extend(m)
            all_candidates.append(candidates);all_masks.append(np.stack(masks) if masks else np.zeros((0,),dtype=np.uint8))
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    prefix=kind+'_'+name+'_'+label
    atomic_json(OUT/(prefix+'.cloud_only.json'),all_candidates)
    with (OUT/(prefix+'.cloud_only.npz')).open('wb') as stream:np.savez_compressed(stream,**{str(i):v for i,v in enumerate(all_masks)})
    return {'seconds':elapsed,'candidate_sha256':digest_json(all_candidates),'masks_sha256':[hashlib.sha256(m.tobytes()).hexdigest() for m in all_masks]},all_candidates,all_masks

def main():
    selection=read_json(Path('/workspace/work/c4-parallel-inference-20260909/benchmark/selection.cloud_only.json'))['rows']
    rows=[selection[0],selection[-1]]
    images=[]
    for row in rows:
        assert digest_file(row['path'])==row['image_sha256']
        with Image.open(row['path']) as src:images.append(ImageOps.exif_transpose(src).convert('RGB'))
    config={'输入图数':2,'选择摘要':digest_json(rows),'source_sha256':{n:digest_file(ROOT/n) for n in ['infer_sam_holdout.py','infer_grounding_holdout.py']},'模型权重变化':False,'读取测试标签':False}
    atomic_json(OUT/'configuration.json',config);results=[]
    for kind in ['sam','grounding']:
        for name in ['adamw','sgd_momentum']:
            record(kind+'_'+name+'_loading');model,processor,queries=load(kind,name)
            # 首个原调用预热既有算子；两条测量都包含CPU预处理及逐候选后处理。
            with torch.inference_mode():
                inputs=processor(images=images[0] if kind=='sam' else [images[0]],text=queries[0] if kind=='sam' else queries[0]+' .',return_tensors='pt').to('cuda')
                if kind=='sam':
                    with torch.autocast('cuda',dtype=torch.bfloat16):model(**inputs)
                else:model(**inputs)
            old,old_c,old_m=measure(kind,name,model,processor,queries,images,False,'original')
            new,new_c,new_m=measure(kind,name,model,processor,queries,images,True,'reuse')
            confirm,confirm_c,confirm_m=measure(kind,name,model,processor,queries,images,True,'confirm')
            result={'专家':kind,'优化器':name,'每图查询数':len(queries),'原始':old,'复用':new,'复测':confirm,
                    '候选完全一致':old_c==new_c==confirm_c,'掩码完全一致':all(np.array_equal(a,b) and np.array_equal(a,c) for a,b,c in zip(old_m,new_m,confirm_m)),
                    '复测加速比':old['seconds']/confirm['seconds']}
            results.append(result);atomic_json(OUT/'partial_comparison.json',results)
            assert result['候选完全一致'] and result['掩码完全一致'],'编码复用输出未通过精确一致性核验'
            assert result['复测加速比']>1,'复用未取得实测收益'
            del model,processor,old_c,new_c,confirm_c,old_m,new_m,confirm_m;gc.collect();torch.cuda.empty_cache()
    summary={'状态':'双优化器图像与文本复用精确对照通过','配置':config,'对照':results,'总秒数':time.time()-begin_time,'边界':'同两图、全部原查询、单进程配对测量；另有全量区域推理并行占用设备。原始候选及掩码只留平台。'}
    atomic_json(OUT/'comparison.json',summary);record('complete');print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(OUT/'failure.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc),'time':time.time()});raise
