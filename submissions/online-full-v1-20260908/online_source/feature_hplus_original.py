"""蒸馏完成后自动恢复H+，为完整融合提取训练/保留特征，保留分区不参与拟合。"""
from train_dino import ROOT,Expert,views
import json,time,os,gc
import numpy as np,torch
from transformers import AutoModel
from peft import LoraConfig,get_peft_model
from download_assets import dump,sha

def main():
    torch.set_num_threads(2);status=ROOT/'candidate_online_20260908/hplus_original_status.json'
    rows=json.loads((ROOT/'candidate_online_20260908/input_manifest.json').read_text());vocab=json.loads((ROOT/'data/vocabulary.json').read_text())
    for optimizer in ['adamw','sgd_momentum']:
        source=ROOT/'checkpoints'/('dinov3_hplus_'+optimizer)
        output=ROOT/'candidate_online_20260908/hplus_original_features'/optimizer;output.mkdir(parents=True,exist_ok=True)
        if (output/'complete.json').exists():continue
        while not (source/'complete.json').exists():
            dump(status,{'status':'waiting_distilled_checkpoint','optimizer':optimizer,'time':time.time()});time.sleep(60)
        checkpoint=source/'last.pt';digest=sha(checkpoint)
        if digest!=json.loads((source/'complete.json').read_text())['checkpoint_sha256']:raise ValueError('蒸馏检查点摘要不符')
        while torch.cuda.mem_get_info()[0]<10*1024**3:
            dump(status,{'status':'waiting_10gib_free_ppu','optimizer':optimizer,'time':time.time()});time.sleep(60)
        torch.cuda.set_per_process_memory_fraction(.10)
        base_dir=ROOT/'models/dinov3-vith16plus-pretrain-lvd1689m'
        base=AutoModel.from_pretrained(base_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager')
        base=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj','o_proj']))
        model=Expert(base,len(vocab['raw_classes']))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        if set(saved['trainable'])!=expected:raise ValueError('蒸馏权重不完整或架构不符')
        model.load_state_dict(saved['trainable'],strict=False);model.to('cuda').eval();model.requires_grad_(False)
        mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
        dump(output/'source.json',{'checkpoint_sha256':digest,'base_manifest_sha256':sha(base_dir/'source_manifest.json'),'vocabulary_sha256':sha(ROOT/'data/vocabulary.json'),'views':['full_letterbox','horizontal_flip'],'feature':'CLS与patch均值拼接','holdout_use':'固定权重提取，禁止将保留特征用于融合训练'})
        start=time.monotonic();processed=0
        with torch.inference_mode():
            for index,row in enumerate(rows):
                target=output/(row['sample_id']+'.npz')
                if target.exists():continue
                pixels=views(row)[:2];x=torch.from_numpy(pixels).permute(0,3,1,2).to('cuda',dtype=torch.float32)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,features=model((x/255-mean)/std)
                scores=(torch.logsumexp(logits.float(),0)-np.log(2)).softmax(-1).cpu().numpy()
                temp=target.with_suffix('.npz.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,features=features.float().cpu().numpy().astype(np.float16),scores=scores)
                os.replace(temp,target);processed+=1
                if processed%10==0:dump(status,{'status':'extracting_distilled_features','optimizer':optimizer,'completed':index+1,'total':len(rows),'images_per_second':processed/max(time.monotonic()-start,1),'time':time.time()})
        dump(output/'complete.json',{'status':'online_features_complete','images':len(rows),'checkpoint_sha256':digest,'time':time.time()})
        del model,base,saved;gc.collect();torch.cuda.empty_cache()
    dump(status,{'status':'both_distilled_feature_sets_complete_fusion_pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'candidate_online_20260908/hplus_original_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
