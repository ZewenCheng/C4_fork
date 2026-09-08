"""用原固定8条病害串行输出核验平台现有去补齐SDPA批量吞吐；逐图数据不导出。"""
import time
begin_time=time.time()
import gc,json,os,re,subprocess,threading
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'benchmark';OUT.mkdir(exist_ok=True)
BASELINE=Path('/workspace/work/c4-parallel-inference-20260909/benchmark')
SOURCE=Path('/workspace/work/c4-contract-entry-v3-20260908/candidate_online_20260908')
for key,name in {'HF_HOME':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'cache','TMPDIR':'tmp','TRITON_CACHE_DIR':'triton',
                 'CUDA_CACHE_PATH':'cuda','ALIPPU_CONFIG_PATH':'ppu','HGRTC_CACHE_PATH':'ppu/hgrtc'}.items():
    p=OUT/'runtime'/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import torch
from report_contract import atomic_json,assemble_final,digest_file,digest_json,evidence_catalog,grade_messages,locked_diagnosis,parse_grade,read_json,run_lock
from run_contract_reports import FrozenGrader

phase='prepare';frames=[];stop=threading.Event()
def monitor():
    cpu_before=sum(os.times()[:2]);before=time.perf_counter()
    while not stop.wait(10):
        cpu=sum(os.times()[:2]);now=time.perf_counter();frame={'phase':phase,'time':time.time(),'process_cpu_percent':100*(cpu-cpu_before)/(now-before)}
        cpu_before,before=cpu,now
        try:
            output=subprocess.run(['ppu-smi'],capture_output=True,text=True,timeout=12).stdout
            match=re.search(r'(\d+)MiB /\s*(\d+)MiB\s*\|\s*(\d+)%',output)
            if match:frame.update(ppu_used_mib=int(match[1]),ppu_utilization_percent=int(match[3]))
        except (OSError,subprocess.TimeoutExpired):frame['ppu_sample_unavailable']=True
        frames.append(frame)

def main():
    global phase
    with run_lock(OUT/'run.lock'):
        assert not (OUT/'comparison.json').exists(),'本版对照已完成，不覆盖'
        rows=read_json(BASELINE/'selection.cloud_only.json')['rows'];old=read_json(BASELINE/'responses.cloud_only.json')['measure_batch_1']
        baseline=next(v for v in read_json(BASELINE/'status.json')['completed_measurements'] if v.get('phase')=='measure_batch_1')
        baseline={**baseline,'attention_backend':'sdpa'}
        assert len(rows)==len(old['finals'])==8 and baseline['protocol_failures']==0
        cases=[]
        for row in rows:
            packet=read_json(SOURCE/'packets'/(row['sample_id']+'.json'));locked=locked_diagnosis(row,packet);catalog=evidence_catalog(packet)
            assert not locked['is_intact'];cases.append((locked,catalog,grade_messages(locked,catalog)))
        requests=[(c[2],1) for c in cases]
        selection=digest_json(rows);assert selection==read_json(BASELINE/'configuration.json')['selection_sha256']
        config={'source_input_sha256':digest_file(SOURCE/'input_manifest.json'),'selection_sha256':selection,'samples':8,
                'candidates':[3],'previous_comparison_sha256':digest_file(Path('/workspace/work/c4-parallel-unpad-20260909/benchmark/comparison.json')),'precision':'bfloat16','attention':'c4_unpadded_sdpa','max_new_tokens':512,
                'unpad_attention_sha256':digest_file(ROOT/'unpad_attention.py'),'runner_sha256':digest_file(ROOT/'run_contract_reports.py'),'parallel_sha256':digest_file(ROOT/'parallel_reports.py'),
                'baseline_configuration_sha256':digest_file(BASELINE/'configuration.json'),'baseline_responses_sha256':digest_file(BASELINE/'responses.cloud_only.json')}
        atomic_json(OUT/'configuration.json',config)
        worker=threading.Thread(target=monitor,daemon=True);worker.start();grader=FrozenGrader('c4_unpadded_sdpa',oom_fallback=False)
        previous=read_json(Path('/workspace/work/c4-parallel-unpad-20260909/benchmark/comparison.json'))
        records=[v for v in previous['对照'] if v.get('phase')!='confirm_selected_batch'];raw_outputs={}
        def status(**extra):atomic_json(OUT/'status.json',{'status':phase,'time':time.time(),'completed_measurements':records,**extra})
        def measure(size,label):
            global phase
            phase=label;status(batch_size=size,completed_requests=0)
            start_metrics=len(grader.batch_metrics);torch.cuda.synchronize();started=time.perf_counter();texts=[]
            for i in range(0,8,size):
                texts.extend(grader.generate_batch(requests[i:i+size]));status(batch_size=size,completed_requests=len(texts),elapsed_seconds=time.perf_counter()-started)
            torch.cuda.synchronize();seconds=time.perf_counter()-started;finals=[];failures=0
            for (locked,catalog,_),text in zip(cases,texts):
                try:finals.append(assemble_final(locked,parse_grade(text,locked,catalog),catalog))
                except ValueError:finals.append(None);failures+=1
            metrics=grader.batch_metrics[start_metrics:]
            record={'batch_size':size,'attention_backend':'c4_unpadded_sdpa','seconds':seconds,'images_per_second':8/seconds,
                    'protocol_failures':failures,'final_differences_vs_serial':sum(a!=b for a,b in zip(finals,old['finals'])),
                    'peak_allocated_bytes':max(m.get('peak_allocated_bytes',0) for m in metrics),'oom_splits':0,
                    'actual_batches':[m['size'] for m in metrics],'phase':label}
            records.append(record);raw_outputs[label]={'texts':texts,'finals':finals,'metrics':metrics}
            atomic_json(OUT/'responses.cloud_only.json',raw_outputs);status(batch_size=size,completed_requests=8)
            return record
        try:
            phase='model_loading';status();grader.load()
            for size in [3]:
                phase='warmup_unpad_batch_'+str(size);status(batch_size=size)
                try:
                    grader.generate_batch(requests[:size])
                    if 8%size:grader.generate_batch(requests[-(8%size):])
                    measure(size,'measure_unpad_batch_'+str(size))
                except RuntimeError as exc:
                    if not isinstance(exc,torch.OutOfMemoryError) and 'out of memory' not in str(exc).lower():raise
                    records.append({'batch_size':size,'attention_backend':'c4_unpadded_sdpa','skipped':'超出本轮显存预算，不作为可用批量','phase':phase,'error':str(exc),'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
                gc.collect();torch.cuda.empty_cache()
            valid=[v for v in records if v.get('attention_backend')=='c4_unpadded_sdpa' and 'seconds' in v and not v['protocol_failures'] and not v['final_differences_vs_serial']]
            if not valid:raise RuntimeError('现有去补齐SDPA无通过原串行结果对照的配置')
            chosen=min(valid,key=lambda v:v['seconds'])
            if chosen['batch_size']!=3:grader.generate_batch(requests[:chosen['batch_size']])
            confirmation=measure(chosen['batch_size'],'confirm_selected_batch')
            if confirmation['protocol_failures'] or confirmation['final_differences_vs_serial']:raise RuntimeError('选定配置复测不一致')
            summary={'状态':'同输入串行批量对照通过','配置':config,'模型加载秒':grader.model_load_seconds,'选定批量':chosen['batch_size'],
                     '选定注意力后端':'c4_unpadded_sdpa','对照':records,'确认批次相对串行加速比':baseline['seconds']/confirmation['seconds'],
                     '总墙钟秒':time.time()-begin_time,'精度边界':'保持模型、提示词及bfloat16，仅比较同输入七字段；未读取测试标签，不衡量准确率',
                     '原SDPA批量负面证据':str(BASELINE.parent/'interruption_receipt.json')}
            atomic_json(OUT/'comparison.json',summary);phase='complete';status()
            print(json.dumps(summary,ensure_ascii=False),flush=True)
        finally:
            stop.set();worker.join(timeout=20);atomic_json(OUT/'resource_samples.json',frames)
            grader.router=None;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(OUT/'failure.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc),'time':time.time()})
        raise
