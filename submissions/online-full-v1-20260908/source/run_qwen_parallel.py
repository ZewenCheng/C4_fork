"""共享冻结权重批量生成，四路CPU准备；保留各版本结果与逐图证据。"""
import json,time,threading,gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from run_qwen_expert_holdout import ROOT,ReportGateway,FIELDS
from qwen_tool_router import QwenToolRouter
from grounding_gateway import GROUNDING_TOOLS
from download_assets import dump,sha

local=threading.local()
init_lock=threading.Lock()
REQUIRED=['inspect_semantics','inspect_regions','inspect_masks','inspect_grounding']

def prepare(row):
    if not hasattr(local,'gateway'):
        with init_lock:local.gateway=ReportGateway()
    gateway=local.gateway;gateway.active_sample=row['sample_id']
    evidence={};events=[]
    for name in REQUIRED:
        start=time.monotonic()
        result=gateway.dispatch(name,{'sample_id':row['sample_id']})
        events.append({'name':name,'arguments':{'sample_id':row['sample_id']},'status':result['status'],'seconds':time.monotonic()-start,'invoked_by':'mandatory_gateway'})
        if result['status']!='ok':raise RuntimeError('必需专家未就绪：'+name)
        evidence[name]=result
    # 每个线程独立网关；证据对象由本次请求持有，释放网关历史引用。
    for key,value in vars(gateway).items():
        if key.endswith('_evidence') and isinstance(value,dict):value.clear()
    return row,evidence,events

def main():
    out=ROOT/'gateway_outputs/qwen_expert_reports_v4_parallel';out.mkdir(exist_ok=True,parents=True)
    status=ROOT/'qwen_report_status.json'
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='holdout']
    # 先处理旧批次未完成项，再完整生成同版本结果；不混淆旧版和新版指标。
    rows.sort(key=lambda r:(ROOT/'gateway_outputs/qwen_expert_reports_v3'/(r['sample_id']+'.json')).exists())
    done=sum((out/(r['sample_id']+'.json')).exists() for r in rows)
    pending=[r for r in rows if not (out/(r['sample_id']+'.json')).exists()]
    dump(out/'configuration.json',{'runner_sha256':sha(Path(__file__)),'manifest_sha256':sha(ROOT/'data/manifest.json'),'model':'/model/Qwen3.6-27B','cpu_workers':4,'batch_initial':2,'fields':FIELDS,'protocol':'必需专家由网关调用，冻结Qwen批量编排；独立版本，不作正式提交'})
    router=QwenToolRouter(None,GROUNDING_TOOLS)
    dump(status,{'status':'loading_frozen_qwen_parallel','time':time.time()});router.load_frozen()
    tokenizer=router.tokenizer;tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    batch_size=2;processed=0;begin=time.monotonic();batch_history=[]
    def generate(prompts):
        encoded=None;tokens=None
        try:
            encoded=tokenizer(prompts,return_tensors='pt',padding=True).to('cuda')
            if encoded.input_ids.shape[1]>24000:raise ValueError('专家上下文超过24000词元')
            with torch.inference_mode():
                tokens=router.model.generate(**encoded,max_new_tokens=2048,do_sample=False,temperature=None,top_p=None,top_k=None,pad_token_id=tokenizer.pad_token_id)
            return tokenizer.batch_decode(tokens[:,encoded.input_ids.shape[1]:],skip_special_tokens=True)
        finally:
            del encoded,tokens
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={i:pool.submit(prepare,r) for i,r in enumerate(pending[:8])}
        position=0
        while position<len(pending):
            size=min(batch_size,len(pending)-position)
            packets=[futures.pop(i).result() for i in range(position,position+size)]
            for i in range(position+8,min(position+8+size,len(pending))):futures[i]=pool.submit(prepare,pending[i])
            prompts=[]
            for row,evidence,events in packets:
                messages=[{'role':'system','content':'根据已调用专家的证据编写巡检JSON。严格使用这些键：'+json.dumps(FIELDS,ensure_ascii=False)+'。所有值必须为字符串。filename保持输入值。不得根据文件名猜测桥名、位置和等级；无依据填空字符串。候选不是确认病害，描述专家分歧与证据不足。无标定不得给出物理尺寸；不得将候选分数转换为等级。只输出JSON。'}, {'role':'user','content':json.dumps({'sample_id':row['sample_id'],'filename':Path(row['path']).name,'expert_evidence':evidence},ensure_ascii=False)}]
                prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
            started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            try:texts=generate(prompts)
            except torch.OutOfMemoryError:
                gc.collect();torch.cuda.empty_cache();batch_size=1
                texts=[generate([p])[0] for p in prompts]
            duration=time.monotonic()-started
            for (row,evidence,events),text in zip(packets,texts):
                try:final=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                except json.JSONDecodeError:final=None
                normalized=dict(final) if isinstance(final,dict) else None
                if normalized is not None and 'ratingScale' in normalized and 'ratingScale(1-5)' not in normalized:normalized['ratingScale(1-5)']=normalized.pop('ratingScale')
                valid=isinstance(normalized,dict) and set(normalized)==set(FIELDS) and all(isinstance(v,str) for v in normalized.values()) and normalized['filename']==Path(row['path']).name and normalized['ratingScale(1-5)'] in ['','1','2','3','4','5']
                dump(out/(row['sample_id']+'.json'),{'status':'draft_requires_effectiveness_review' if valid else 'invalid_final_structure','sample_id':row['sample_id'],'final':normalized,'raw_final':final,'raw_text':text,'events':events,'structural_validation':{'seven_fields_valid':valid,'missing_visual_tools':[]},'batch_size':size,'batch_seconds':duration})
            processed+=size;position+=size
            batch_history.append({'size':size,'seconds':duration,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            dump(status,{'status':'generating_whole_holdout_reports','version':'v4_parallel','completed':done+processed,'total':len(rows),'images_per_second':processed/(time.monotonic()-begin),'cpu_workers':4,'batch_size':batch_size,'recent_batches':batch_history[-10:],'time':time.time()})
    dump(status,{'status':'whole_holdout_drafts_complete_effectiveness_review_pending','version':'v4_parallel','images':len(rows),'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'qwen_report_status.json',{'status':'failed','version':'v4_parallel','error_type':type(e).__name__,'time':time.time()});raise
