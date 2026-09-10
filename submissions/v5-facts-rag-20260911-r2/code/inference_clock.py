"""仅记录显式模型调用区间；传输、预处理和输出写入由调用方置于边界外。"""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,math,os,threading,time,sys

VERSION='model-inference-intervals-v1'

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def interval_union(intervals):
    values=[]
    for item in intervals:
        a,b=item['start'],item['end']
        if type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a) or not math.isfinite(b) or b<a:
            raise ValueError('模型计时间隔非法')
        values.append((a,b))
    merged=[]
    for a,b in sorted(values):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],b))
        else:merged.append((a,b))
    return sum(b-a for a,b in merged)

class InferenceClock:
    def __init__(self,context,*,journal=None,clock=time.perf_counter):
        self.context=context;self.context_sha256=digest(context);self.clock=clock
        self.origin=clock();self.intervals=[];self.lock=threading.RLock()
        self.journal=Path(journal) if journal is not None else None
        if self.journal and self.journal.exists():raise FileExistsError('计时日志已存在；新调用须单独计时，不覆盖历史')

    @contextmanager
    def model(self,name,*,batch_size,model_sha256,synchronize=None):
        if not name or type(batch_size) is not int or batch_size<1 or not isinstance(model_sha256,str) or len(model_sha256)!=64:
            raise ValueError('模型计时身份缺失')
        if synchronize is not None:
            try:synchronize()
            except BaseException:
                point=self.clock()-self.origin
                with self.lock:
                    self.intervals.append({'name':name,'start':point,'end':point,'batch_size':batch_size,
                        'model_sha256':model_sha256,'status':'failed','timing_valid':False,'timing_method':'pre_sync_failed'})
                    self.save()
                raise
        start=self.clock()-self.origin;status='failed'
        try:
            yield
            status='complete'
        finally:
            sync_error=None
            if synchronize is not None:
                try:synchronize()
                except BaseException as exc:sync_error=exc;status='failed'
            end=self.clock()-self.origin
            with self.lock:
                self.intervals.append({'name':name,'start':start,'end':end,'batch_size':batch_size,
                                       'model_sha256':model_sha256,'status':status,'timing_valid':sync_error is None,
                                       'timing_method':'synchronized_host' if synchronize else 'host_synchronous'})
                self.save()
            if sync_error is not None:raise sync_error

    def snapshot(self):
        with self.lock:
            return {'schema_version':VERSION,'timing_scope':'model_inference_only','context':self.context,
                    'context_sha256':self.context_sha256,'intervals':list(self.intervals),
                    'infer_time':interval_union(self.intervals),'summed_call_seconds':sum(x['end']-x['start'] for x in self.intervals),
                    'failed_calls':sum(x['status']!='complete' for x in self.intervals),
                    'excludes':['model_loading','preprocessing','data_transfer','formatting','json_write'],
                    'boundary_requirement':'传入model上下文内仅运行模型；调用方须另行验证边界，无旧预测缓存冒充本次调用。'}

    def save(self):
        if self.journal:
            self.journal.parent.mkdir(parents=True,exist_ok=True)
            tmp=self.journal.with_suffix(self.journal.suffix+'.pending')
            tmp.write_text(json.dumps(self.snapshot(),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            os.replace(tmp,self.journal)

    def require_complete(self):
        result=self.snapshot()
        if not result['intervals'] or result['failed_calls']:raise ValueError('模型推理计时为空或存在失败调用')
        return result
