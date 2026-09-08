from storage_usage import workspace_used
"""7B冻结教师按分片下载并保存BF16执行基底，降低100GB内峰值存储。"""
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import requests
import torch
from safetensors.torch import load_file,save_file
from download_assets import ROOT,dump,sha,download

REPO='facebook/dinov3-vit7b16-pretrain-lvd1689m'
def main():
    raw=ROOT/'models'/REPO.split('/')[-1]
    dest=ROOT/'models/dinov3-vit7b16-bf16';dest.mkdir(parents=True,exist_ok=True)
    manifest=dest/'source_manifest.json'
    if manifest.exists():records=json.loads(manifest.read_text())['files']
    else:
        r=requests.get(f'https://modelscope.cn/api/v1/models/{REPO}/repo/files',params={'Revision':'master','Recursive':'true'},timeout=(15,30));r.raise_for_status()
        records=[x for x in r.json()['Data']['Files'] if x['Type']=='blob' and x['Path']!='.gitattributes']
        dump(manifest,{'repo':REPO,'files':records,'conversion':'公开原始FP32分片SHA核验后转BF16；教师实际按BF16冻结运行，无训练更新'})
    converted=[]
    for rec in records:
        target=dest/rec['Path'];receipt=target.with_name(target.name+'.receipt.json')
        if receipt.exists():converted.append(json.loads(receipt.read_text()));continue
        if rec['Path'].endswith('.safetensors'):
            # 预留仍在下载的WeMM全量、三个初始模型及训练检查点，不能按当前part低估。
            used=workspace_used()
            wdir=ROOT/'models/WeMM-Embedding-9B'
            wactual=sum(p.stat().st_size for p in wdir.rglob('*') if p.is_file()) if wdir.exists() else 0
            reserve=max(0,18_840_000_000-wactual)
            if used+reserve+rec['Size']*1.5+2_000_000_000>100_000_000_000:
                dump(ROOT/'teacher_download_status.json',{'status':'waiting_storage','time':time.time(),'required_next_shard_bytes':int(rec['Size']*1.5)})
                raise RuntimeError('需进一步去重或调整存储后续传教师，未越过平台容量')
        download(REPO,rec)
        source=raw/rec['Path']
        if rec['Path'].endswith('.safetensors'):
            tensors=load_file(source,device='cpu')
            bf16={k:v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in tensors.items()}
            temp=target.with_name(target.name+'.tmp');save_file(bf16,temp,metadata={'format':'pt','source_sha256':rec['Sha256']})
            os.replace(temp,target);del tensors,bf16;gc.collect()
            item={'file':rec['Path'],'source_sha256':rec['Sha256'],'bf16_sha256':sha(target),'bytes':target.stat().st_size}
            dump(receipt,item)
            resolved=source.resolve()
            if not resolved.is_relative_to(raw.resolve()):raise ValueError('临时原始分片越界')
            source.unlink()
        else:
            target.write_bytes(source.read_bytes())
            item={'file':rec['Path'],'sha256':sha(target),'bytes':target.stat().st_size};dump(receipt,item)
        converted.append(item)
        dump(ROOT/'teacher_download_status.json',{'status':'preparing_frozen_teacher','files_complete':len(converted),'files_total':len(records),'time':time.time()})
    index=dest/'model.safetensors.index.json'
    d=json.loads(index.read_text());d['metadata']['total_size']=sum(x['bytes'] for x in converted if x['file'].endswith('.safetensors'));dump(index,d)
    dump(ROOT/'teacher_download_status.json',{'status':'complete_bf16_frozen_base','converted':converted,'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'teacher_error.json',{'error_type':type(e).__name__,'time':time.time()});raise
