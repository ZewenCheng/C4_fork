"""公开模型下载：固定文件修订、SHA256核验、续传，保留许可证；不执行下载源码。"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import requests

ROOT=Path('/workspace/work/c4-experiments/expert-autotrain-20260906')
REPOS=['facebook/dinov3-vith16plus-pretrain-lvd1689m','facebook/sam3','IDEA-Research/grounding-dino-base']
def dump(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def download(repo, rec):
    name=rec['Path']; relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('发布路径越界')
    dest=ROOT/'models'/repo.split('/')[-1]/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        if dest.stat().st_size==rec['Size'] and sha(dest)==rec['Sha256']:return
        raise ValueError('已有文件摘要不匹配')
    partial=dest.with_name(dest.name+'.part')
    url=f"https://modelscope.cn/models/{repo}/resolve/{rec['Revision']}/{name}"
    for attempt in range(2):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>rec['Size']:raise ValueError('续传文件大小异常')
            if offset<rec['Size']:
                with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(15,60)) as r:
                    r.raise_for_status()
                    if offset and r.status_code==206:
                        if not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('续传范围错误')
                        mode='ab'
                    elif r.status_code==200:mode='wb';offset=0
                    else:raise ValueError('下载状态异常')
                    last=time.monotonic()
                    with partial.open(mode) as f:
                        for b in r.iter_content(4*1024*1024):
                            f.write(b);offset+=len(b)
                            if offset>rec['Size']:raise ValueError('下载超出发布大小')
                            if time.monotonic()-last>30:
                                print(json.dumps({'下载':repo,'文件':name,'完成字节':offset,'总字节':rec['Size']},ensure_ascii=False),flush=True);last=time.monotonic()
            if partial.stat().st_size!=rec['Size'] or sha(partial)!=rec['Sha256']:raise ValueError('下载摘要不匹配')
            os.replace(partial,dest)
            print(json.dumps({'已核验':repo+'/'+name,'字节':rec['Size']},ensure_ascii=False),flush=True)
            return
        except requests.RequestException as e:
            print(json.dumps({'失败类型':type(e).__name__,'模型':repo,'尝试':attempt+1},ensure_ascii=False),flush=True)
            if attempt:raise RuntimeError('下载连接失败，保留续传') from None
def main():
    jobs=[]
    for repo in REPOS:
        folder=ROOT/'models'/repo.split('/')[-1];folder.mkdir(parents=True,exist_ok=True)
        manifest=folder/'source_manifest.json'
        if manifest.exists():records=json.loads(manifest.read_text())['files']
        else:
            r=requests.get(f'https://modelscope.cn/api/v1/models/{repo}/repo/files',params={'Revision':'master','Recursive':'true'},timeout=(15,30));r.raise_for_status()
            records=[x for x in r.json()['Data']['Files'] if x['Type']=='blob' and not x['Path'].endswith(('.bin','.pt')) and x['Path']!='.gitattributes']
            dump(manifest,{'repo':repo,'source':'ModelScope公开发布库','files':records})
        jobs.extend((repo,r) for r in records)
    if sum(r['Size'] for _,r in jobs)>8_000_000_000:raise ValueError('本下载批次超出8GB分配')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda x:download(*x),jobs))
    dump(ROOT/'download_status.json',{'status':'complete','repos':REPOS,'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'download_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()})
        raise
