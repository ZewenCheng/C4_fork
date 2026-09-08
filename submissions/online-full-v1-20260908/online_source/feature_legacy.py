"""复用原ConvNeXtV2大基底，三视图特征后训练可替换分类头；不改原权重。"""
import os,json,time,random,gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel,AutoImageProcessor
from download_assets import ROOT,dump,sha

class LegacyHead(nn.Module):
    def __init__(self,width,classes):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,512),nn.GELU(),nn.Dropout(.2),nn.Linear(512,classes))
    def forward(self,x):return self.net(x)

def views(row):
    with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
    w,h=im.size;s=min(w,h);left=(w-s)//2;top=(h-s)//2
    return [im,ImageOps.mirror(im),im.crop((left,top,left+s,top+s))]

def main():
    torch.set_num_threads(2)
    status=ROOT/'candidate_online_20260908/legacy_features_status.json';rows=json.loads((ROOT/'candidate_online_20260908/input_manifest.json').read_text())
    vocab=json.loads((ROOT/'data/vocabulary.json').read_text());classes=len(vocab['raw_classes'])
    folder=ROOT/'candidate_online_20260908/legacy_convnext_features';folder.mkdir(exist_ok=True)
    missing=[r for r in rows if not (folder/(r['sample_id']+'.npz')).exists()]
    base=Path('/workspace/work/road-infrastructure-finals-cloud/models/transformers/convnextv2_large')
    if missing:
        while torch.cuda.mem_get_info()[0]<9*1024**3:
            dump(status,{'status':'waiting_9gib_free_ppu','time':time.time()});time.sleep(60)
        torch.cuda.set_per_process_memory_fraction(.09)
        processor=AutoImageProcessor.from_pretrained(base,local_files_only=True)
        model=AutoModel.from_pretrained(base,local_files_only=True).to('cuda').eval();model.requires_grad_(False)
        dump(folder/'source.json',{'base':str(base),'base_sha256':sha(base/'model.safetensors'),'manifest_sha256':sha(ROOT/'candidate_online_20260908/input_manifest.json'),
             'frozen_base':True,'views':['full','horizontal_flip','center_square'],'holdout_use':'只提取固定特征，不参与训练与预处理拟合'})
        start=time.monotonic();done=len(rows)-len(missing)
        with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
            for offset in range(0,len(missing),4):
                batch=missing[offset:offset+4];images=[im for group in pool.map(views,batch) for im in group]
                inputs=processor(images=images,return_tensors='pt').to('cuda');output=model(**inputs)
                features=output.pooler_output.float().cpu().numpy().reshape(len(batch),3,-1)
                for row,f in zip(batch,features):
                    target=folder/(row['sample_id']+'.npz');tmp=target.with_suffix('.npz.tmp')
                    with tmp.open('wb') as stream:np.savez_compressed(stream,features=f.astype(np.float16))
                    os.replace(tmp,target)
                done+=len(batch)
                dump(status,{'status':'extracting_legacy_features','completed':done,'total':len(rows),'images_per_second':(offset+len(batch))/max(time.monotonic()-start,1),'time':time.time()})
                del inputs,output,features
        del model,processor;gc.collect();torch.cuda.empty_cache()
    dump(folder/'complete.json',{'status':'online_features_complete','images':len(rows),'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'candidate_online_20260908/legacy_features_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
