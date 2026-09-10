"""线上新架构统一运行入口：区域→特征→全部专家组装→冻结Qwen→结果。"""
import os,sys,json,time,subprocess,fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from download_assets import ROOT,dump,sha
ONLINE=ROOT/'candidate_online_20260908'

def stage_script(name,original,replacements,transform=None):
    text=(ROOT/original).read_text()
    for old,new in replacements:
        assert old in text,(original,old);text=text.replace(old,new)
    if transform:text=transform(text)
    target=ONLINE/(name+'.py');compile(text,str(target),'exec')
    if target.exists():assert target.read_text()==text,'运行源码已变化，应另建版本'
    else:target.write_text(text)
    return target

def record_phase(name,started,complete=True):
    path=ONLINE/'phase_timings.json'
    values=json.loads(path.read_text()) if path.exists() else []
    ended=time.time()
    values.append({'phase':name,'started_at':started,'ended_at':ended,'seconds':ended-started,'complete':complete})
    dump(path,values)

def run_group(scripts,phase=None):
    children=[]
    pool=None
    started=time.time();complete=False
    try:
        for specification in scripts:
            args=list(specification) if isinstance(specification,(list,tuple)) else [specification]
            script=Path(args[0])
            with (ONLINE/(script.stem+'.log')).open('ab') as log:
                child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=dict(os.environ,PYTHONPATH=str(ROOT)))
            children.append((script,child))
        if children:
            pool=ThreadPoolExecutor(max_workers=len(children))
            waiting={pool.submit(child.wait):(script,child) for script,child in children}
            for future in as_completed(waiting):
                script,child=waiting[future];code=future.result()
                if code:raise RuntimeError(script.name+'退出码'+str(code))
                children.remove((script,child))
        complete=True
    finally:
        # 只清理本次调用实际创建的仍在运行的子进程，防止失败后后台继续写同一缓存。
        for _,child in children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        if pool is not None:pool.shutdown(wait=True)
        if phase is not None:record_phase(phase,started,complete)

def main():
    if (ONLINE/'result.json').exists() and not (ONLINE/'contract_reports/configuration.json').exists():
        raise ValueError('当前目录已有旧版结果，请通过独立新入口运行协议版本')
    with (ONLINE/'full_pipeline.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('完整流水线已运行');return
        dump(ONLINE/'full_pipeline_pid.json',{'pid':os.getpid(),'time':time.time()})
        base=[("ROOT/'data/manifest.json'","ROOT/'candidate_online_20260908/input_manifest.json'")]
        def remove_holdout_score(text):
            a=text.index('        correct=0;total=0');b=text.index('        del model,base,saved',a)
            return text[:a]+"        dump(output/'complete.json',{'status':'online_features_complete','images':len(rows),'checkpoint_sha256':digest,'time':time.time()})\n"+text[b:]
        hplus=stage_script('feature_hplus','extract_distilled_hplus.py',base+[("ROOT/'hplus_distilled_features'","ROOT/'candidate_online_20260908/hplus_distilled_features'"),("ROOT/'hplus_features_status.json'","ROOT/'candidate_online_20260908/hplus_features_status.json'")],remove_holdout_score)
        original=stage_script('feature_hplus_original','extract_distilled_hplus.py',base+[("'dinov3_hplus_distilled_'","'dinov3_hplus_'"),("ROOT/'hplus_distilled_features'","ROOT/'candidate_online_20260908/hplus_original_features'"),("ROOT/'hplus_features_status.json'","ROOT/'candidate_online_20260908/hplus_original_status.json'")],remove_holdout_score)
        wemm=stage_script('feature_wemm','extract_wemm_features.py',base+[("ROOT/'wemm_features'","ROOT/'candidate_online_20260908/wemm_features'"),("ROOT/'wemm_features_status.json'","ROOT/'candidate_online_20260908/wemm_features_status.json'")])
        def feature_only(text):
            a=text.index('    x=torch.from_numpy(np.stack(');b=text.index("if __name__=='__main__':",a)
            return text[:a]+"    dump(folder/'complete.json',{'status':'online_features_complete','images':len(rows),'time':time.time()})\n\n"+text[b:]
        legacy=stage_script('feature_legacy','train_legacy_head.py',base+[("ROOT/'legacy_convnext_features'","ROOT/'candidate_online_20260908/legacy_convnext_features'"),("ROOT/'legacy_head_status.json'","ROOT/'candidate_online_20260908/legacy_features_status.json'")],feature_only)
        anomaly=stage_script('feature_anomaly','infer_anomaly_holdout.py',base+[("ROOT/'gateway_outputs/anomaly_holdout'","ROOT/'candidate_online_20260908/anomaly_holdout'"),("ROOT/'anomaly_inference_status.json'","ROOT/'candidate_online_20260908/anomaly_inference_status.json'")])
        regions=[ONLINE/folder/opt/'complete.json' for folder in ['rtdetr_holdout','sam_holdout','grounding_holdout'] for opt in ['adamw','sgd_momentum']]
        region_started=time.time()
        while not all(p.exists() for p in regions):
            dump(ONLINE/'full_pipeline_status.json',{'status':'waiting_online_region_experts','completed_receipts':sum(p.exists() for p in regions),'total_receipts':6,'time':time.time()})
            for name in ['rtdetr','sam','grounding']:
                pf=ONLINE/(name+'_pid.json');pid=json.loads(pf.read_text())['pid'];proc=Path('/proc')/str(pid)/'stat'
                complete=all((ONLINE/(name+'_holdout')/o/'complete.json').exists() for o in ['adamw','sgd_momentum'])
                if not complete and (not proc.exists() or proc.read_text().split(') ')[1].startswith('Z')):raise RuntimeError(name+'区域进程停止且缺完成凭证')
            time.sleep(30)
        record_phase('区域专家等待',region_started)
        dump(ONLINE/'full_pipeline_status.json',{'status':'extracting_online_features_group1','time':time.time()})
        run_group([hplus,wemm,legacy],'特征并行第一组')
        dump(ONLINE/'full_pipeline_status.json',{'status':'extracting_online_features_group2','time':time.time()})
        run_group([original,anomaly],'特征并行第二组')
        dump(ONLINE/'full_pipeline_status.json',{'status':'assembling_all_experts','time':time.time()})
        run_group([ROOT/'build_online_packets.py'],'全部专家装配')
        # 报告与条件评级由同一合同执行器处理，禁止生成后再覆盖模型的某几个字段。
        dump(ONLINE/'full_pipeline_status.json',{'status':'generating_contract_reports_and_ratings','time':time.time()})
        run_group([[ROOT/'run_contract_reports.py','--manifest',ONLINE/'input_manifest.json',
                    '--packets',ONLINE/'packets','--output',ONLINE/'contract_reports','--result',ONLINE/'result.json',
                    '--batch-size',os.environ.get('C4_REPORT_BATCH_SIZE','4'),'--cpu-workers',os.environ.get('C4_CPU_WORKERS','4'),
                    '--attention-backend',os.environ.get('C4_ATTENTION_BACKEND','sdpa')]],'批量报告与评级')
        report_state=json.loads((ONLINE/'contract_reports/status.json').read_text())
        assert report_state['status']=='complete'
        rows=json.loads((ONLINE/'input_manifest.json').read_text());receipts=[]
        for row in rows:
            p=ONLINE/'contract_reports/records'/(row['sample_id']+'.json')
            receipts.append({'sample_id':row['sample_id'],'sha256':sha(p)})
        target=ONLINE/'result.json'
        assert report_state['images']==len(rows) and sha(target)==report_state['result_sha256']
        entry_state=ROOT/'entry_execution.json'
        start=json.loads(entry_state.read_text())['first_started_at'] if entry_state.exists() else json.loads((ONLINE/'pipeline_status.json').read_text())['time']
        dump(ONLINE/'timing.json',{'elapsed_seconds_since_region_launch':time.time()-start,'scope':'含区域、特征、报告及病害评级；恢复时含中断等待，不冒充冷启动基准'})
        dump(ONLINE/'full_pipeline_status.json',{'status':'all_online_results_complete_packaging_pending','protocol':report_state['protocol'],'images':len(rows),'result_sha256':sha(target),'receipts':receipts,'remaining':['正式格式与设计书打包','GitHub版本备份核验'],'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ONLINE/'full_pipeline_status.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'time':time.time()});raise
