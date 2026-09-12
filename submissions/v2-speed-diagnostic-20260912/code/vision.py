"""线上旧v2轨道六分支及隔离teacher三视图，全部重新前向。"""
from common import *
import sys,time
import numpy as np
LEGACY=Path('/workspace/work/road-infrastructure-finals-cloud')
BASE=LEGACY/'models/transformers/convnextv2_large'
def features(x):
 x=np.asarray(x,dtype=np.float64);x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
 x=x.mean(axis=1);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def prediction(model,x,labels):
 z=np.clip(x@model['coef'].T+model['intercept'],-700,700)
 scores=1/(1+np.exp(-z));scores=np.where(model['constant']>=0,model['constant'],scores)
 order=np.argsort(-scores,axis=1,kind='stable')[:,:3]
 return [[{'label':labels[j],'uncalibrated_score':float(scores[i,j])} for j in row] for i,row in enumerate(order)]
def main():
 rows=read(ROOT/'input_manifest.json');mode=sys.argv[1];started=time.time()
 if mode=='rail':
  from v2_rail_partner import V2RailPartner
  labels=read(ASSETS/'v2_labels.json');adapter=V2RailPartner(LEGACY,ROOT/'v2_features',{s:s for s in labels},timeout=7200)
  rails=[dict(r,questionCategory='轨道') for r in rows if r['category']=='轨道']
  responses=adapter.inspect_many(rails) if rails else {}
  dump(ROOT/'v2_candidates.json',{r['sample_id']:responses[r['sample_id']]['result']['outputs']['frozen_v2']['candidates'][:3] for r in rails})
  dump(ROOT/'rail_complete.json',{'完成':len(rails),'秒':time.time()-started,'版本':adapter.version,'新前向':not any(v['cache_reused'] for v in responses.values())})
 else:
  import torch
  from transformers import AutoModel,AutoImageProcessor
  from PIL import Image,ImageOps
  torch.set_num_threads(4);model=AutoModel.from_pretrained(BASE,local_files_only=True).to('cuda').eval();model.requires_grad_(False)
  proc=AutoImageProcessor.from_pretrained(BASE,local_files_only=True)
  with np.load(ASSETS/'teacher.npz',allow_pickle=False) as z:head={k:z[k] for k in ['coef','intercept','constant']}
  labels=read(ASSETS/'teacher_labels.json');allfeatures=[];preds={};seconds=0
  def views(path):
   with Image.open(path) as im:im=ImageOps.exif_transpose(im).convert('RGB')
   w,h=im.size;s=min(w,h);return [im,ImageOps.mirror(im),im.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))]
  with torch.inference_mode():
   for start in range(0,len(rows),4):
    batch=rows[start:start+4];images=[im for r in batch for im in views(r['path'])]
    encoded=proc(images=images,return_tensors='pt').to('cuda');torch.cuda.synchronize();tick=time.time()
    result=model(**encoded).pooler_output;torch.cuda.synchronize();seconds+=time.time()-tick
    x=result.float().cpu().numpy().reshape(len(batch),3,-1).astype(np.float16).astype(np.float32)
    allfeatures.extend(x);fx=features(x)
    preds.update({r['sample_id']:prediction(head,fx[[i]],labels)[0] for i,r in enumerate(batch)})
    for im in images:im.close()
    dump(ROOT/'progress.json',{'阶段':'隔离teacher新前向','完成':start+len(batch),'总数':len(rows)})
  np.savez_compressed(ROOT/'teacher_features.npz',sample_ids=np.asarray([r['sample_id'] for r in rows]),x=np.asarray(allfeatures,dtype=np.float16))
  dump(ROOT/'teacher_candidates.json',preds);dump(ROOT/'teacher_complete.json',{'完成':len(rows),'秒':time.time()-started,'模型前向秒':seconds,'新前向':True})
if __name__=='__main__':main()
