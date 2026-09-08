"""线上新架构统一运行入口：区域→特征→全部专家组装→冻结Qwen→结果。"""
import os,sys,json,time,subprocess,fcntl
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

def run_group(scripts):
    children=[]
    for script in scripts:
        with (ONLINE/(script.stem+'.log')).open('ab') as log:
            child=subprocess.Popen([sys.executable,'-u',str(script)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=dict(os.environ,PYTHONPATH=str(ROOT)))
        children.append((script,child))
    while children:
        for script,child in children[:]:
            code=child.poll()
            if code is not None:
                if code:raise RuntimeError(script.name+'退出码'+str(code))
                children.remove((script,child))
        if children:time.sleep(15)

def main():
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
        while not all(p.exists() for p in regions):
            dump(ONLINE/'full_pipeline_status.json',{'status':'waiting_online_region_experts','completed_receipts':sum(p.exists() for p in regions),'total_receipts':6,'time':time.time()})
            for name in ['rtdetr','sam','grounding']:
                pf=ONLINE/(name+'_pid.json');pid=json.loads(pf.read_text())['pid'];proc=Path('/proc')/str(pid)/'stat'
                complete=all((ONLINE/(name+'_holdout')/o/'complete.json').exists() for o in ['adamw','sgd_momentum'])
                if not complete and (not proc.exists() or proc.read_text().split(') ')[1].startswith('Z')):raise RuntimeError(name+'区域进程停止且缺完成凭证')
            time.sleep(30)
        dump(ONLINE/'full_pipeline_status.json',{'status':'extracting_online_features_group1','time':time.time()})
        run_group([hplus,wemm,legacy])
        dump(ONLINE/'full_pipeline_status.json',{'status':'extracting_online_features_group2','time':time.time()})
        run_group([original,anomaly])
        dump(ONLINE/'full_pipeline_status.json',{'status':'assembling_all_experts','time':time.time()})
        run_group([ROOT/'build_online_packets.py'])
        def online_report(text):
            a=text.index('def prepare(row):');b=text.index('\ndef main():',a)
            text=text[:a]+'''def prepare(row):
    packet=json.loads((ROOT/'candidate_online_20260908/packets'/(row['sample_id']+'.json')).read_text())
    assert packet['sample_id']==row['sample_id']
    return row,packet['evidence'],packet['events']
'''+text[b:]
            anchor="                valid=isinstance(normalized,dict)"
            insert="""                if normalized is not None:
                    packet=json.loads((ROOT/'candidate_online_20260908/packets'/(row['sample_id']+'.json')).read_text())
                    normalized['defectType']=packet['predicted_type']
                    normalized['questionCategory']=packet['category']
                    normalized['filename']=Path(row['path']).name
                    normalized['ratingScale(1-5)']=''
                    normalized['defectDescription']='分类专家预测：'+packet['predicted_type']+'。'+str(normalized.get('defectDescription',''))
"""
            assert anchor in text;text=text.replace(anchor,insert+anchor)
            return text
        report=stage_script('report_online','run_qwen_parallel.py',base+[("ROOT/'gateway_outputs/qwen_expert_reports_v4_parallel'","ROOT/'candidate_online_20260908/reports'"),("ROOT/'qwen_report_status.json'","ROOT/'candidate_online_20260908/qwen_report_status.json'"),("'v4_parallel'","'online_full_v1'")],online_report)
        dump(ONLINE/'full_pipeline_status.json',{'status':'generating_online_qwen_reports','time':time.time()});run_group([report])
        rows=json.loads((ONLINE/'input_manifest.json').read_text());results=[];receipts=[]
        for row in rows:
            p=ONLINE/'reports'/(row['sample_id']+'.json');d=json.loads(p.read_text());assert d['structural_validation']['seven_fields_valid']
            results.append(d['final']);receipts.append({'sample_id':row['sample_id'],'sha256':sha(p)})
        target=ONLINE/'result.json'
        if target.exists():assert json.loads(target.read_text())==results
        else:dump(target,results)
        start=json.loads((ONLINE/'pipeline_status.json').read_text())['time']
        dump(ONLINE/'timing.json',{'elapsed_seconds_since_region_launch':time.time()-start,'scope':'含模型加载、区域/特征/组装/Qwen；不含先前训练；正式计时格式打包时核对'})
        dump(ONLINE/'full_pipeline_status.json',{'status':'all_online_results_complete_packaging_pending','images':len(results),'result_sha256':sha(target),'receipts':receipts,'remaining':['正式格式与设计书打包','GitHub版本备份核验'],'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ONLINE/'full_pipeline_status.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'time':time.time()});raise
