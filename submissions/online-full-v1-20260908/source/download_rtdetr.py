"""准备官方RT-DETRv2-X对应的R101VD基底，核对原始权重摘要。"""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
import hashlib,json,time
import requests
from huggingface_hub import snapshot_download
from download_assets import ROOT,dump,sha
from storage_usage import workspace_used
REPO='PekingU/rtdetr_v2_r101vd'
EXPECTED='fa2d79aa50064eb2d41abdc1acaa5ca20fc9d95347851c93ab6aad30a8974ee0'
def main():
    r=requests.get(f'https://hf-mirror.com/api/models/{REPO}',params={'blobs':'true'},timeout=(15,30));r.raise_for_status();d=r.json()
    if d['id']!=REPO or d.get('gated') is not False:raise ValueError('并非指定公开模型')
    records=[x for x in d['siblings'] if x['rfilename']!='.gitattributes']
    weight=next(x for x in records if x['rfilename']=='model.safetensors')
    if weight['lfs']['sha256']!=EXPECTED:raise ValueError('权重与原始发布摘要不符')
    if workspace_used()+sum(x['size'] for x in records)+1_000_000_000>100_000_000_000:raise RuntimeError('当前空间不足')
    out=ROOT/'models/rtdetr_v2_r101vd'
    snapshot_download(repo_id=REPO,revision=d['sha'],endpoint='https://hf-mirror.com',token=False,local_dir=out,allow_patterns=[x['rfilename'] for x in records],max_workers=3)
    for x in records:
        p=out/x['rfilename']
        if p.stat().st_size!=x['size']:raise ValueError('发布大小不符')
        if x.get('lfs'):
            if sha(p)!=x['lfs']['sha256']:raise ValueError('权重摘要不符')
        else:
            h=hashlib.sha1(f"blob {x['size']}\0".encode());h.update(p.read_bytes())
            if h.hexdigest()!=x['blobId']:raise ValueError('配置摘要不符')
    dump(out/'source_manifest.json',d);dump(ROOT/'rtdetr_download_status.json',{'status':'complete','revision':d['sha'],'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'rtdetr_download_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
