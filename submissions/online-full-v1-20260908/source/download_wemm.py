from storage_usage import workspace_used
"""下载公开无门禁WeMM固定版本；镜像字节核对腾讯发布的原始SHA256。"""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import hashlib
import json
import subprocess
import time
from huggingface_hub import snapshot_download
from download_assets import ROOT,dump,sha
EXPECTED='b6d5dff9e632973991f1d0cbfcfd26c42ffd66fbb8ebd6f852aece08e9794fa4'
def main():
    metadata=json.loads((ROOT/'wemm_public_metadata.json').read_text())
    if metadata['id']!='tencent/WeMM-Embedding-9B' or metadata['gated'] is not False:raise ValueError('来源不是指定公开模型')
    records=[r for r in metadata['siblings'] if r['rfilename']!='.gitattributes']
    model_rec=next(r for r in records if r['rfilename']=='model.safetensors')
    if model_rec['lfs']['sha256']!=EXPECTED:raise ValueError('与腾讯原始发布摘要不符')
    used=workspace_used()
    total=sum(r['size'] for r in records)
    out=ROOT/'models/WeMM-Embedding-9B'
    existing=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    if used+max(0,total-existing)+3_000_000_000>100_000_000_000:raise RuntimeError('当前100GB内空间不足，待已授权去重完成后恢复')
    dump(ROOT/'wemm_download_status.json',{'status':'downloading','revision':metadata['sha'],'bytes':total,'time':time.time()})
    snapshot_download(repo_id=metadata['id'],revision=metadata['sha'],endpoint='https://hf-mirror.com',token=False,local_dir=out,
                      allow_patterns=[r['rfilename'] for r in records],max_workers=4)
    for rec in records:
        p=out/rec['rfilename']
        if p.stat().st_size!=rec['size']:raise ValueError('文件长度不符')
        if rec.get('lfs'):
            if sha(p)!=rec['lfs']['sha256']:raise ValueError('LFS摘要不符')
        else:
            h=hashlib.sha1(f"blob {rec['size']}\0".encode());h.update(p.read_bytes())
            if h.hexdigest()!=rec['blobId']:raise ValueError('源码blob摘要不符')
    dump(out/'source_manifest.json',metadata)
    dump(ROOT/'wemm_download_status.json',{'status':'complete','revision':metadata['sha'],'verified_bytes':total,'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'wemm_download_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
