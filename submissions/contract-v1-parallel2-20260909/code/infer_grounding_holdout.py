"""恢复Grounding适配器，按训练全局词表执行双版本完整保留集候选推理。"""
from prepare_regions import ROOT
import json,time,gc
import torch
from PIL import Image,ImageOps
from transformers import AutoProcessor,GroundingDinoForObjectDetection
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha
from contextlib import contextmanager

@contextmanager
def reuse_image_backbone(backbone,pixel_values,pixel_mask):
    """只在当前图像的独立查询间复用骨干；下游可追加列表，不能污染缓存。"""
    if backbone.training or torch.is_grad_enabled():raise ValueError('骨干复用仅允许冻结推理')
    original=backbone.forward
    features,positions=backbone(pixel_values,pixel_mask)
    def cached(values,mask):
        if values is not pixel_values or mask is not pixel_mask:raise ValueError('骨干缓存不得跨图像或掩码复用')
        return list(features),list(positions)
    backbone.forward=cached
    try:yield
    finally:backbone.forward=original

def predict_queries(model,processor,image,queries):
    """固定原查询字符串和次序，图像预处理与骨干每张只执行一次。"""
    visual=processor.image_processor(images=[image],return_tensors='pt').to('cuda')
    with reuse_image_backbone(model.get_base_model().model.backbone,visual['pixel_values'],visual['pixel_mask']):
        for query in queries:
            text=processor.tokenizer(query+' .',return_tensors='pt').to('cuda')
            inputs={**visual,**text}
            yield query,inputs,model(**inputs)

def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.22)
    status=ROOT/'grounding_inference_status.json'
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='holdout']
    directory=ROOT/'models/grounding-dino-base';processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
    for name in ['adamw','sgd_momentum']:
        source=ROOT/'checkpoints'/('grounding_dino_'+name);out=ROOT/'gateway_outputs/grounding_holdout'/name;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        assert (source/'complete.json').exists()
        config=json.loads((source/'configuration.json').read_text());checkpoint=source/'last.pt';digest=sha(checkpoint)
        assert digest==json.loads((source/'complete.json').read_text())['checkpoint_sha256']
        assert sha(directory/'source_manifest.json')==config['base_manifest_sha256']
        while torch.cuda.mem_get_info()[0]<22*1024**3:
            dump(status,{'status':'waiting_22gib_free_ppu','optimizer':name,'time':time.time()});time.sleep(60)
        base=GroundingDinoForObjectDetection.from_pretrained(directory,local_files_only=True,disable_custom_kernels=True,attn_implementation='eager')
        embedding=base.model.text_backbone.embeddings.word_embeddings;base.get_input_embeddings=lambda:embedding
        model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules='all-linear'))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert set(saved['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(saved['trainable'],strict=False);del saved
        model.to('cuda').eval();model.requires_grad_(False)
        queries=config['query_vocabulary'];begin=time.monotonic();processed=0
        with torch.inference_mode():
            for i,row in enumerate(rows):
                target=out/(row['sample_id']+'.json')
                if target.exists():assert json.loads(target.read_text())['checkpoint_sha256']==digest;continue
                with Image.open(row['path']) as image:im=ImageOps.exif_transpose(image).convert('RGB')
                candidates=[]
                for query,inputs,prediction in predict_queries(model,processor,im,queries):
                    length=int(inputs['attention_mask'][0].sum())
                    scores=prediction.logits[0,:,1:max(2,length-2)].float().mean(-1).sigmoid()
                    assert torch.isfinite(scores).all()
                    values,indices=scores.topk(min(3,len(scores)));boxes=prediction.pred_boxes[0,indices].float()
                    xyxy=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1).clamp(0,1)
                    assert torch.isfinite(xyxy).all()
                    candidates.extend({'query':query,'uncalibrated_score':float(v),'box_normalized_xyxy':b.cpu().tolist()} for v,b in zip(values,xyxy))
                    del inputs,prediction
                dump(target,{'sample_id':row['sample_id'],'checkpoint_sha256':digest,'base_manifest_sha256':config['base_manifest_sha256'],'original_size_hw':[im.height,im.width],'candidates':candidates,'limitations':['提示来自全局训练词表，不读取保留图答案','与训练相同的词元平均分数，未校准','未匹配查询无负例监督；候选不等于病害确认，不输出尺寸或等级']})
                processed+=1
                dump(status,{'status':'extracting_grounding_candidates','optimizer':name,'completed':i+1,'total':len(rows),'queries_per_image':len(queries),'images_per_second':processed/max(time.monotonic()-begin,1),'time':time.time()})
        dump(out/'complete.json',{'status':'whole_holdout_grounding_candidates_complete_gateway_pending','images':len(rows),'checkpoint_sha256':digest,'time':time.time()})
        del model,base;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_grounding_candidate_sets_complete_gateway_pending','time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'grounding_inference_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
