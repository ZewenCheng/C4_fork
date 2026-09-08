"""冻结DINOv3-7B的训练分区多视图特征；断点逐图保存供H+蒸馏。"""
from prepare_regions import ROOT
import json,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from transformers import AutoModel
from download_assets import dump,sha

def main():
    status=ROOT/'teacher_features_status.json'
    while True:
        p=ROOT/'teacher_download_status.json';d=json.loads(p.read_text()) if p.exists() else {}
        if d.get('status')=='complete_bf16_frozen_base':break
        dump(status,{'status':'waiting_teacher_base','time':time.time()});time.sleep(60)
    torch.set_num_threads(2)
    while torch.cuda.mem_get_info()[0]<22*1024**3:
        dump(status,{'status':'waiting_22gib_free_ppu_memory','time':time.time()});time.sleep(60)
    torch.cuda.set_per_process_memory_fraction(.23)
    directory=ROOT/'models/dinov3-vit7b16-bf16'
    model=AutoModel.from_pretrained(directory,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='train']
    out=ROOT/'teacher_features';out.mkdir(exist_ok=True)
    dump(out/'source.json',{'base_source_manifest_sha256':sha(directory/'source_manifest.json'),
         'base_conversion_receipts':[json.loads(p.read_text()) for p in sorted(directory.glob('*.safetensors.receipt.json'))],
         'data_manifest_sha256':sha(ROOT/'data/manifest.json'),'teacher_frozen':True,'views':['full_letterbox','full_horizontal_flip'],
         'size':512,'global_feature':'CLS与patch均值拼接','local_feature':'4x4网格平均patch，水平翻转视图仍保留其自身坐标系'})
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    done=0;processed=0;start=time.monotonic()
    with torch.inference_mode():
        for row in rows:
            target=out/(row['sample_id']+'.npz')
            if target.exists():done+=1;continue
            with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
            im=ImageOps.pad(im,(512,512),color=(124,116,104))
            array=np.stack([np.asarray(im),np.asarray(ImageOps.mirror(im))])
            x=torch.from_numpy(array).permute(0,3,1,2).to('cuda',dtype=torch.float32)
            with torch.autocast('cuda',dtype=torch.bfloat16):h=model(pixel_values=(x/255-mean)/std).last_hidden_state
            patches=h[:,5:]
            if patches.shape[1]!=1024:raise ValueError('7B局部token网格不符，不能错位建库')
            features=torch.cat([h[:,0],patches.mean(1)],dim=-1).float().cpu().numpy().astype(np.float16)
            grid=patches.reshape(2,4,8,4,8,-1).mean((2,4)).reshape(2,16,-1).float().cpu().numpy().astype(np.float16)
            tmp=target.with_name(target.name+'.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,global_features=features,local_features=grid)
            os.replace(tmp,target);done+=1;processed+=1
            if done%10==0:
                rate=processed/max(time.monotonic()-start,1)
                d={'status':'extracting_frozen_teacher','completed':done,'total':len(rows),'images_per_second':rate,'remaining_seconds':(len(rows)-done)/max(rate,1e-6),'time':time.time()}
                dump(status,d);print(json.dumps(d,ensure_ascii=False),flush=True)
            del h,patches,x
    dump(status,{'status':'complete','completed':done,'total':len(rows),'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'teacher_features_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
