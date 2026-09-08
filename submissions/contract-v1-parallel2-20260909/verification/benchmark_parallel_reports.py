"""同一固定8条病害的串行/批量对照；原始响应留官方work，仅返回聚合指标。"""
import time
begin_time=time.time()
import gc,hashlib,json,os,re,subprocess,threading
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'benchmark'
OUT.mkdir(parents=True,exist_ok=True)
for key,name in {'HF_HOME':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'cache','TMPDIR':'tmp','TRITON_CACHE_DIR':'triton',
                 'CUDA_CACHE_PATH':'cuda','ALIPPU_CONFIG_PATH':'ppu','HGRTC_CACHE_PATH':'ppu/hgrtc'}.items():
    p=OUT/'runtime'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import torch
from report_contract import atomic_json,assemble_final,digest_file,digest_json,evidence_catalog,grade_messages,locked_diagnosis,parse_grade,read_json,run_lock
from run_contract_reports import FrozenGrader

SOURCE=Path('/workspace/work/c4-contract-entry-v3-20260908/candidate_online_20260908')
phase='prepare';samples=[];stop=threading.Event()

def monitor():
    before_cpu=sum(os.times()[:2]);before=time.perf_counter()
    while not stop.wait(10):
        now=time.perf_counter();cpu=sum(os.times()[:2]);value={'phase':phase,'time':time.time(),'process_cpu_percent':100*(cpu-before_cpu)/(now-before)}
        before_cpu,before=cpu,now
        try:
            text=subprocess.run(['ppu-smi'],capture_output=True,text=True,timeout=12).stdout
            match=re.search(r'(\d+)MiB /\s*(\d+)MiB\s*\|\s*(\d+)%',text)
            if match:value.update(ppu_used_mib=int(match[1]),ppu_total_mib=int(match[2]),ppu_utilization_percent=int(match[3]))
        except (OSError,subprocess.TimeoutExpired):value['ppu_sample_unavailable']=True
        samples.append(value)

def main():
    global phase
    with run_lock(OUT/'run.lock'):
        if (OUT/'comparison.json').exists():raise ValueError('对照已完成，禁止重复覆盖')
        rows=read_json(SOURCE/'input_manifest.json');cases=[]
        for row in rows:
            packet=read_json(SOURCE/'packets'/(row['sample_id']+'.json'))
            diagnosis=locked_diagnosis(row,packet)
            if not diagnosis['is_intact']:
                catalog=evidence_catalog(packet);cases.append((row,diagnosis,catalog,grade_messages(diagnosis,catalog)))
        cases.sort(key=lambda c:c[0]['sample_id'])
        selected=[cases[round(i*(len(cases)-1)/7)] for i in range(8)]
        assert len({c[0]['sample_id'] for c in selected})==8
        requests=[(c[3],1) for c in selected]
        atomic_json(OUT/'selection.cloud_only.json',{'rows':[c[0] for c in selected],'source':str(SOURCE)})
        config={'source_input_sha256':digest_file(SOURCE/'input_manifest.json'),'selection_sha256':digest_json([c[0] for c in selected]),
                'samples':8,'candidates':[1,2,4,8],'precision':'bfloat16','attention':'sdpa','max_new_tokens':512,
                'runner_sha256':digest_file(ROOT/'run_contract_reports.py'),'parallel_sha256':digest_file(ROOT/'parallel_reports.py')}
        atomic_json(OUT/'configuration.json',config)
        resource_thread=threading.Thread(target=monitor,daemon=True);resource_thread.start()
        grader=FrozenGrader();records=[];raw_outputs={};serial=None
        try:
            phase='model_loading';atomic_json(OUT/'status.json',{'status':phase,'time':time.time()});grader.load()
            def measure(size,label):
                nonlocal serial
                global phase
                phase=label
                atomic_json(OUT/'status.json',{'status':phase,'batch_size':size,'time':time.time(),'completed_measurements':records})
                start_index=len(grader.batch_metrics);torch.cuda.synchronize();started=time.perf_counter();texts=[]
                for i in range(0,len(requests),size):texts.extend(grader.generate_batch(requests[i:i+size]))
                torch.cuda.synchronize();seconds=time.perf_counter()-started
                finals=[];errors=[]
                for (_,diagnosis,catalog,_),text in zip(selected,texts):
                    try:finals.append(assemble_final(diagnosis,parse_grade(text,diagnosis,catalog),catalog))
                    except ValueError as exc:errors.append(type(exc).__name__);finals.append(None)
                if serial is None:serial=finals
                metrics=grader.batch_metrics[start_index:]
                record={'batch_size':size,'seconds':seconds,'images_per_second':len(requests)/seconds,
                        'protocol_failures':len(errors),'final_differences_vs_serial':sum(a!=b for a,b in zip(finals,serial)),
                        'peak_allocated_bytes':max((m.get('peak_allocated_bytes',0) for m in metrics),default=0),
                        'oom_splits':sum(m['status']=='oom_split' for m in metrics),'actual_batches':[m['size'] for m in metrics if m['status']=='complete'],
                        'phase':label}
                records.append(record);raw_outputs[label]={'texts':texts,'finals':finals,'metrics':metrics}
                atomic_json(OUT/'responses.cloud_only.json',raw_outputs)
                atomic_json(OUT/'status.json',{'status':'measurement_complete','time':time.time(),'completed_measurements':records})
                return record
            for size in [1,2,4,8]:
                phase='warmup_batch_'+str(size);grader.batch_limit=8
                atomic_json(OUT/'status.json',{'status':phase,'time':time.time(),'completed_measurements':records})
                grader.generate_batch(requests[:size])
                if grader.batch_limit<size:
                    records.append({'batch_size':size,'skipped':'预热已超出显存预算','fallback_limit':grader.batch_limit})
                    continue
                measure(size,'measure_batch_'+str(size))
            candidates=[r for r in records if 'seconds' in r and not r['protocol_failures'] and not r['final_differences_vs_serial'] and not r['oom_splits']]
            if not candidates:raise RuntimeError('没有通过同版本结果对照的批量配置')
            chosen=min(candidates,key=lambda r:r['seconds']);grader.batch_limit=chosen['batch_size']
            confirmation=measure(chosen['batch_size'],'confirm_selected_batch')
            if confirmation['protocol_failures'] or confirmation['final_differences_vs_serial']:raise RuntimeError('选定配置再次核验不一致')
            baseline=next(r for r in records if r.get('phase')=='measure_batch_1')
            summary={'状态':'同输入串行批量对照通过','配置':config,'模型加载秒':grader.model_load_seconds,
                     '选定批量':chosen['batch_size'],'对照':records,'确认批次相对串行加速比':baseline['seconds']/confirmation['seconds'],
                     '总墙钟秒':time.time()-begin_time,'精度边界':'只比较本次同模型输出，不衡量准确率；未读取测试标签'}
            atomic_json(OUT/'comparison.json',summary)
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        finally:
            stop.set();resource_thread.join(timeout=20)
            atomic_json(OUT/'resource_samples.json',samples)
            grader.router=None;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(OUT/'failure.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc),'time':time.time()})
        raise
