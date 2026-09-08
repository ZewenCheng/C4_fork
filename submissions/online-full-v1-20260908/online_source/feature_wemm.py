"""WeMM两套适配完成后自动提取全量图像嵌入，供融合训练与完整推理。"""
from train_wemm import ROOT
import os,json,time,gc
import numpy as np,torch
from PIL import Image,ImageOps,ImageEnhance
from transformers import AutoModel,AutoProcessor
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

def main():
    torch.set_num_threads(2);status=ROOT/'candidate_online_20260908/wemm_features_status.json';directory=ROOT/'models/WeMM-Embedding-9B'
    rows=json.loads((ROOT/'candidate_online_20260908/input_manifest.json').read_text());processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
    processor.tokenizer.padding_side='right';embedding_id=processor.tokenizer.convert_tokens_to_ids('<embedding>')
    for optimizer in ['adamw','sgd_momentum']:
        source=ROOT/'checkpoints'/('wemm9b_'+optimizer);out=ROOT/'candidate_online_20260908/wemm_features'/optimizer;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        while not (source/'complete.json').exists():
            dump(status,{'status':'waiting_wemm_trained_adapter','optimizer':optimizer,'time':time.time()});time.sleep(60)
        checkpoint=source/'last.pt';digest=sha(checkpoint)
        if digest!=json.loads((source/'complete.json').read_text())['checkpoint_sha256']:raise ValueError('WeMM检查点摘要不符')
        config=json.loads((source/'configuration.json').read_text())
        if config['reviewed_embedding_code_sha256']!=sha(directory/'modeling_wemm_embedding.py'):raise ValueError('WeMM池化源码发生改变')
        while torch.cuda.mem_get_info()[0]<26*1024**3:
            dump(status,{'status':'waiting_26gib_free_ppu','optimizer':optimizer,'time':time.time()});time.sleep(60)
        torch.cuda.set_per_process_memory_fraction(.28)
        base=AutoModel.from_pretrained(directory,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj','o_proj','in_proj_qkv']))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if set(saved['trainable'])!={n for n,p in model.named_parameters() if p.requires_grad}:raise ValueError('WeMM适配参数不完整')
        model.load_state_dict(saved['trainable'],strict=False);del saved
        model.to('cuda').eval();model.requires_grad_(False)
        dump(out/'source.json',{'checkpoint_sha256':digest,'base_manifest_sha256':sha(directory/'source_manifest.json'),'data_manifest_sha256':sha(ROOT/'candidate_online_20260908/input_manifest.json'),'views':['full_max512','brightness_1.05_max512'],'pooling':'官方embedding方法，末位embedding token','supervision_input':False,'holdout_use':'仅图像推理，不参与融合拟合'})
        begin=time.monotonic();processed=0
        with torch.inference_mode():
            for index,row in enumerate(rows):
                target=out/(row['sample_id']+'.npz')
                if target.exists():continue
                with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
                im.thumbnail((512,512),Image.Resampling.LANCZOS);images=[im,ImageEnhance.Brightness(im).enhance(1.05)]
                text=['<|im_start|>user<|vision_start|><|image_pad|><|vision_end|><|im_end|><embedding>']*2
                inputs=processor(images=images,text=text,padding=True,return_tensors='pt').to('cuda');end=inputs['attention_mask'].sum(-1)-1
                if not torch.all(inputs['input_ids'][torch.arange(2,device='cuda'),end]==embedding_id):raise ValueError('WeMM末位token不符')
                with torch.autocast('cuda',dtype=torch.bfloat16):features=model.get_base_model().embedding(**inputs,use_cache=False)
                temp=target.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,features=features.float().cpu().numpy().astype(np.float16))
                os.replace(temp,target);processed+=1
                if processed%10==0:dump(status,{'status':'extracting_wemm_image_features','optimizer':optimizer,'completed':index+1,'total':len(rows),'images_per_second':processed/max(time.monotonic()-begin,1),'time':time.time()})
                del inputs,features
        dump(out/'complete.json',{'status':'all_image_embeddings_complete_fusion_pending','images':len(rows),'checkpoint_sha256':digest,'time':time.time()})
        del model,base;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_wemm_feature_sets_complete_fusion_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'candidate_online_20260908/wemm_features_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
