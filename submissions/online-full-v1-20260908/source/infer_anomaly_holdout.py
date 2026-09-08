"""冻结教师与已建coreset对完整保留图推理，提供粗网格距离证据。"""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,'/workspace/work/c4-experiments/expert-autotrain-20260906')
from download_assets import ROOT,dump,sha
import numpy as np,torch
from PIL import Image,ImageOps
from transformers import AutoModel

def main():
    torch.set_num_threads(2);status=ROOT/'anomaly_inference_status.json'
    while torch.cuda.mem_get_info()[0]<20*1024**3:
        dump(status,{'status':'waiting_20gib_free_ppu','time':time.time()});time.sleep(60)
    torch.cuda.set_per_process_memory_fraction(.20)
    folder=ROOT/'checkpoints/anomaly_banks';config=json.loads((folder/'configuration.json').read_text());banks={}
    for name,item in config['banks'].items():
        if item['status']!='built':continue
        file=folder/(name+'.npz')
        if sha(file)!=item['sha256']:raise ValueError('异常参考库摘要不符')
        with np.load(file,allow_pickle=False) as d:features=torch.from_numpy(d['features'].astype(np.float32))
        banks[name]=torch.nn.functional.normalize(features,dim=-1).to('cuda')
    base=ROOT/'models/dinov3-vit7b16-bf16'
    model=AutoModel.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval();model.requires_grad_(False)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='holdout']
    out=ROOT/'gateway_outputs/anomaly_holdout';out.mkdir(parents=True,exist_ok=True)
    mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1);std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)
    begin=time.monotonic();processed=0
    with torch.inference_mode():
        for index,row in enumerate(rows):
            target=out/(row['sample_id']+'.json')
            if target.exists():continue
            with Image.open(row['path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
            original_size=list(im.size);im=ImageOps.pad(im,(512,512),color=(124,116,104))
            x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).unsqueeze(0).to('cuda',dtype=torch.float32)
            with torch.autocast('cuda',dtype=torch.bfloat16):h=model(pixel_values=(x/255-mean)/std).last_hidden_state
            patches=h[:,5:]
            if patches.shape[1]!=1024:raise ValueError('冻结教师patch网格不符')
            grid=patches.reshape(1,4,8,4,8,-1).mean((2,4)).reshape(16,-1).float()
            # 与训练库存储float16再归一化的精度链一致。
            grid=torch.nn.functional.normalize(grid.half().float(),dim=-1);scores={}
            for name,reference in banks.items():
                distance=(2-2*(grid@reference.T).clamp(-1,1)).clamp_min(0).sqrt().min(1).values
                scores[name]={'grid_distances':distance.reshape(4,4).cpu().tolist(),'max_distance':float(distance.max()),'mean_distance':float(distance.mean()),'bank_sha256':config['banks'][name]['sha256']}
            dump(target,{'protocol_version':1,'tool':'inspect_anomaly','status':'ok','result':{'sample_id':row['sample_id'],'expert':'dinov3_7b_coreset','base_source_sha256':sha(base/'source_manifest.json'),'evidence_type':'uncalibrated_local_feature_distance','grid_frame':'512x512_letterbox_4x4','original_size':original_size,'scores':scores,'rating':None,'limitations':['网格含可能的填充区域，不是像素病害分割','弱正常库未经确认正常，距离不等于病害概率','不据新颖度强行判为病害或完好']}})
            processed+=1;dump(status,{'status':'whole_holdout_anomaly_inference','completed':index+1,'total':len(rows),'images_per_second':processed/max(time.monotonic()-begin,1),'time':time.time()})
    dump(status,{'status':'whole_holdout_distance_evidence_complete_qwen_fusion_pending','images':len(rows),'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'anomaly_inference_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
