"""完整352图字段映射整改：显式输出分类预测，保留原稿与证据来源。"""
import json,time,collections
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from run_qwen_parallel import prepare,ROOT,FIELDS
from download_assets import dump,sha

def repair(row):
    _,evidence,events=prepare(row)
    source=ROOT/'gateway_outputs/qwen_expert_reports_v4_parallel'/(row['sample_id']+'.json')
    original=json.loads(source.read_text());final=dict(original['final'])
    outputs=evidence['inspect_semantics']['result']['outputs']
    ranked={}
    for optimizer,value in outputs.items():
        candidates=value['candidates']
        if isinstance(candidates,dict):candidates=candidates['shown_candidates']
        ranked[optimizer]=sorted(candidates,key=lambda x:x['uncalibrated_score'],reverse=True)
    # 固定沿用已训练AdamW融合分类器首选，不按保留集标签逐图挑选。
    prediction=ranked['adamw'][0];label=prediction['label']
    assert isinstance(label,str) and label.strip()
    alternatives={k:v[0]['label'] for k,v in ranked.items()}
    final['defectType']=label
    explanation='分类专家预测病害类型为“'+label+'”（模型预测，非现场确证）。'
    if len(set(alternatives.values()))>1:
        explanation+='优化器版本存在分歧：'+json.dumps(alternatives,ensure_ascii=False)+'。'
    final['defectDescription']=explanation+final.get('defectDescription','')
    # 原描述可能表达不确定性，明确区分预测类型与确证，不伪造尺度和等级。
    validation=set(final)==set(FIELDS) and all(isinstance(v,str) for v in final.values()) and final['filename']==Path(row['path']).name and final['ratingScale(1-5)'] in ['','1','2','3','4','5']
    assert validation
    record={'sample_id':row['sample_id'],'status':'prediction_mapping_repaired_effectiveness_pending','final':final,'source_report_sha256':sha(source),'events':events,'field_provenance':{'defectType':{'tool':'inspect_semantics','optimizer':'adamw','checkpoint_sha256':outputs['adamw']['checkpoint_sha256'],'selection':'top1_prediction','candidate':prediction,'other_predictions':alternatives}},'remaining_gaps':['规范评级未完整接入','桥名缺少可用身份输入','位置缺少构件与病害区域关联'],'structural_validation':{'seven_fields_valid':validation,'missing_visual_tools':[]}}
    return record

def main():
    out=ROOT/'gateway_outputs/qwen_expert_reports_v5_mapping';out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='holdout']
    dump(out/'configuration.json',{'runner_sha256':sha(Path(__file__)),'input_version':'v4_parallel','method':'固定AdamW融合首选预测映射，不使用保留图标签；保留原报告','images':len(rows)})
    counts=collections.Counter();start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(repair,rows)):
            dump(out/(result['sample_id']+'.json'),result)
            counts['七字段合格']+=result['structural_validation']['seven_fields_valid']
            counts['类型非空']+=bool(result['final']['defectType'])
            dump(ROOT/'report_mapping_status.json',{'status':'repairing_full_holdout','completed':i+1,'total':len(rows),'counts':dict(counts),'seconds':time.monotonic()-start,'time':time.time()})
    dump(ROOT/'report_mapping_status.json',{'status':'full_holdout_prediction_mapping_complete_other_gaps_pending','images':len(rows),'counts':dict(counts),'seconds':time.monotonic()-start,'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(ROOT/'report_mapping_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
