"""聚合作业监控；每十次按照剩余阶段进度重新计算建议间隔。"""
import json
import time
from pathlib import Path
from download_assets import ROOT,dump
statefile=ROOT/'monitor_state.json'
state=json.loads(statefile.read_text()) if statefile.exists() else {'count':0,'interval_seconds':900,'history':[]}
statuses={}
for name in ['download_status','wemm_download_status','teacher_download_status','dino_status','tabular_status','regions_status','sam_training_status','wemm_training_status','teacher_features_status','rtdetr_download_status','grounding_dino_training_status','rtdetr_training_status','anomaly_banks_status','dino_distillation_status','standards_status','ordinal_status','legacy_head_status','normative_area_status','standard_index_status','tabpfn_status','tabpfn_adapter_status','anomaly_inference_status','tabpfn_evaluation_status','visual_gateway_status','completed_backup_status']:
    p=ROOT/(name+'.json')
    if p.exists():
        statuses[name]=json.loads(p.read_text())
        if 'converted' in statuses[name]:statuses[name]['converted_files']=len(statuses[name].pop('converted'))
for name in ['hplus_features_status','combined_gateway_status','wemm_features_status','fusion_status','rtdetr_inference_status','sam_inference_status','grounding_inference_status','qwen_report_status']:
    p=ROOT/(name+'.json')
    if p.exists():statuses[name]=json.loads(p.read_text())
count=state['count']+1
sample={'time':time.time(),'statuses':statuses}
state['history']=(state['history']+[sample])[-10:];state['count']=count
remaining=[x.get('remaining_seconds_current_optimizer') for x in statuses.values() if x.get('remaining_seconds_current_optimizer') is not None and x.get('status') not in ('failed','complete','completed')]
for value in statuses.values():
    rate=value.get('images_per_second',0)
    if (value.get('status','').startswith('extracting_') or value.get('status')=='generating_whole_holdout_reports') and rate>0 and value.get('total',0)>value.get('completed',0):
        remaining.append((value['total']-value['completed'])/rate)
nearest=min(remaining) if remaining else None
previous_interval=state.get('interval_seconds')
state['interval_seconds']=3600 if nearest is None or nearest>7200 else 1800 if nearest>3600 else 900
state['interval_changed']=previous_interval!=state['interval_seconds']
if count%10==0 or state['interval_changed']:
    state['recalculation']={'at_count':count,'reason':'最近运行阶段剩余>2小时每60分钟，1至2小时每30分钟，1小时内每15分钟；每次检查阈值，每10次全面重算；阶段ETA不是全架构ETA','time':time.time()}
dump(statefile,state)
print(json.dumps({'监控次数':count,'建议间隔秒':state['interval_seconds'],'本次重算':count%10==0,'状态':statuses},ensure_ascii=False))
