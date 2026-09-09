"""固定新架构权重，独立线上输入与输出，先并行执行区域专家。"""
import json,time,sys,subprocess,fcntl,hashlib
from pathlib import Path
root=Path('/workspace/work/c4-contract-entry-v3-20260908')
out=root/'candidate_online_20260908';out.mkdir(exist_ok=True)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
with (out/'launch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    inp=Path('/dataset/决赛数据集1/赛题4/线上测试集')
    images=sorted(p for p in inp.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
    assert images,'线上图片未发现'
    rows=[{'sample_id':hashlib.sha256(str(p.relative_to(inp)).encode()).hexdigest()[:24],'path':str(p),'split':'holdout','input_role':'online_inference_only','image_sha256':digest(p)} for p in images]
    assert len({r['sample_id'] for r in rows})==len(rows)
    manifest=out/'input_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==rows
    else:manifest.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    stages=[('rtdetr','infer_rtdetr_holdout.py','rtdetr_holdout'),('sam','infer_sam_holdout.py','sam_holdout'),('grounding','infer_grounding_holdout.py','grounding_holdout')]
    processes=[]
    for name,original,cache in stages:
        source=(root/original).read_text()
        source=source.replace("ROOT/'data/manifest.json'","ROOT/'candidate_online_20260908/input_manifest.json'")
        source=source.replace("ROOT/'gateway_outputs/"+cache+"'","ROOT/'candidate_online_20260908/"+cache+"'")
        for state in ['rtdetr_inference_status.json','sam_inference_status.json','grounding_inference_status.json']:
            source=source.replace("ROOT/'"+state+"'","ROOT/'candidate_online_20260908/"+state+"'")
        target=out/('infer_'+name+'.py');compile(source,str(target),'exec')
        if target.exists():assert target.read_text()==source
        else:target.write_text(source)
        pidfile=out/(name+'_pid.json')
        if pidfile.exists():
            old=json.loads(pidfile.read_text());cmd=Path('/proc')/str(old['pid'])/'cmdline'
            if cmd.exists() and str(target).encode() in cmd.read_bytes():processes.append(old);continue
        if all((out/cache/o/'complete.json').exists() for o in ['adamw','sgd_momentum']):continue
        with (out/(name+'.log')).open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u',str(target)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env=dict(__import__('os').environ,PYTHONPATH=str(root)))
        rec={'stage':name,'pid':child.pid,'script_sha256':digest(target),'time':time.time()};pidfile.write_text(json.dumps(rec));processes.append(rec)
    result={'status':'online_region_experts_started','images':len(rows),'input_manifest_sha256':digest(manifest),'processes':processes,'remaining':['H+/WeMM/原视觉特征与融合','线上输入网关适配与Qwen报告','完整结果计时与GitHub版本备份'],'time':time.time()}
    (out/'pipeline_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))
