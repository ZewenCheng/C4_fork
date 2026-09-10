"""独立读回330图候选与模型区间，不下载逐图正文；无真值不声称准确率。"""
from pathlib import Path
import json,hashlib,sys,collections,math
root=Path('/workspace/work/c4-v5-release-20260911-v1');out=root/'inference'
sys.path.insert(0,str(root/'code'))
from report_contract import locked_diagnosis,validate_final
from fact_protocol import validate_facts
from grounded_description import validate_description
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
if not (out/'complete.json').is_file():
    print(json.dumps({'状态':'尚未完成','进度':read(out/'progress.json') if (out/'progress.json').exists() else None,'失败':read(out/'failure.json') if (out/'failure.json').exists() else None},ensure_ascii=False));raise SystemExit(2)
manifest=out/'input_manifest.json'
rows=read(manifest);finals=read(out/'result.json');config=read(out/'configuration.json');receipt=read(out/'complete.json');clock=read(out/'model_intervals.json')
assets=read(out/'assets.json')
assert clock['context']==config and clock['context_sha256']==digest(config)
grade_root=Path('/workspace/work/c4-final-fact-grade-20260910-v1')
assert sha(grade_root/'manifest.json')==config['grade_manifest_sha256']==assets['grade_manifest_sha256']
assert sha(grade_root/'grade_model.joblib')==assets['grade_model_sha256']
cv_root=Path('/workspace/work/c4-cv-multiview-final-20260910')
assert sha(cv_root/'multiview_model.npz')==assets['vision']['head_sha256']
assert read(cv_root/'manifest.json')['base_sha256']==assets['vision']['base_sha256']
models={'convnext_three_views':assets['vision']['base_sha256'],'cv_multiview_head':assets['vision']['head_sha256'],
        'fact_grade_head':assets['grade_model_sha256'],'rag_encoder':next(v for k,v in assets['rag']['knowledge_identity']['encoder_files'].items() if k.endswith('.onnx'))}
assert len(rows)==len(finals)==330 and len(list((out/'records').glob('*.json')))==330
assert sha(manifest)==config['manifest_sha256'] and config['release_authorized'] is True and config['quality_gain_verified'] is False
from v5_entry import verify_code
assert verify_code(root/'code')==config['source_manifest_sha256']
for k,name in [('summary_sha256','summary.json'),('result_sha256','result.json'),('timing_sha256','infer_time.json')]:assert receipt[k]==sha(out/name)
disease=0;source_cases=collections.Counter();pages=collections.Counter();memo=0;rag_calls=0
for row,final in zip(rows,finals):
    record=read(out/'records'/(row['sample_id']+'.json'))
    assert record['sample_id']==row['sample_id'] and record['input_sha256']==sha(Path(row['path']))==row['image_sha256']
    assert record['version']==config['version'] and record['final']==final
    assert record['cv']['checkpoint_sha256']==models['cv_multiview_head']
    diag=locked_diagnosis(row,{'sample_id':row['sample_id'],'category':final['questionCategory'],'predicted_type':record['cv']['candidates'][0]['label'],'prediction_source':'inspect_cv_multiview/multiview/top1'})
    validate_final(final,diag);validate_facts(record['facts']);validate_description(record['description'],record['facts'],record['facts']['sources'])
    assert final['defectDescription']==record['description']['text']
    if not diag['is_intact']:
        disease+=1;assert final['ratingScale(1-5)']==record['grade']['predicted_grade']
        assert record['grade']['model_version']==assets['grade_model_sha256']
    knowledge=record['knowledge'];assert knowledge['rating_decision'] is None and not knowledge['numeric_rules']
    memo+=int(knowledge['memo_hit'])
    for packet in knowledge['source_packets']:
        assert packet['input_tokens']+packet['reserved_output_tokens']<=8192
        assert packet['rating_decision'] is None and not packet['numeric_rules']
        sources=set(e['source_id'] for e in packet['evidence']);assert len(sources)<=1
        for source in sources:source_cases[source]+=1
        for e in packet['evidence']:pages[e['source_id']]+=1
        if not knowledge['memo_hit']:rag_calls+=len(packet['queries'])
events=[]
for x in clock['intervals']:
    assert x['status']=='complete' and x.get('timing_valid',True) and x['end']>=x['start']
    assert x['name'] in models and x['model_sha256']==models[x['name']]
    events.extend([(x['start'],1),(x['end'],-1)])
active=0;last=0;seconds=0
for point,delta in sorted(events):
    if active:seconds+=point-last
    active+=delta;last=point
assert math.isclose(seconds,read(out/'infer_time.json')['infer_time'],abs_tol=1e-9)
for name,expected in [('convnext_three_views',330),('cv_multiview_head',330),('fact_grade_head',disease),('rag_encoder',rag_calls)]:
    assert sum(x['batch_size'] for x in clock['intervals'] if x['name']==name)==expected,(name,expected)
summary=read(out/'summary.json')
assert math.isclose(seconds,summary['纯模型秒'],abs_tol=1e-9) and summary['660秒速度门']==(seconds<=660)
snapshot={'complete_sha256':sha(out/'complete.json'),'assets_sha256':sha(out/'assets.json'),'configuration_sha256':sha(out/'configuration.json'),
          'clock_sha256':sha(out/'model_intervals.json'),'record_sha256':{p.name:sha(p) for p in (out/'records').glob('*.json')}}
(root/'verified_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
evidence={'状态':'独立330图七字段、来源、RAG预算和模型计时验收通过','候选汇总':summary,'RAG命中病例数':dict(source_cases),'RAG证据页引用次数':dict(pages),'同次运行知识复用病例':memo,'应有RAG前向':rag_calls,'完整凭证SHA256':sha(out/'complete.json'),'独立资产与记录快照SHA256':sha(root/'verified_snapshot.json'),'外层状态':read(root/'launcher_status.json')}
(root/'independent_verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2));print(json.dumps(evidence,ensure_ascii=False))

# 输入重新扫描后与旧固定清单按路径和图像摘要核对，旧预测只用于结束后的差异统计。
old_root=Path('/workspace/work/c4-cv-partner-release-20260910-v2/inference/baseline/candidate_online_20260908')
old_rows=read(old_root/'input_manifest.json')
assert {(r['path'],r['image_sha256']) for r in rows}=={(r['path'],r['image_sha256']) for r in old_rows}
old_by_path={r['path']:r for r in old_rows}
changes=collections.Counter()
for row,final in zip(rows,finals):
    old=read(old_root/'contract_reports/records'/(old_by_path[row['path']]['sample_id']+'.json'))['final']
    for key in final:changes[key]+=int(final[key]!=old[key])
assert seconds<=660
assert read(root/'output/result.json')==finals
assert sha(root/'output/infer_time.json')==sha(out/'infer_time.json')
evidence['旧伙伴版字段变化']=dict(changes)
evidence['输入集合与旧330图一致']=True
assert evidence['外层状态']['退出码']==0
(root/'independent_verification.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
print(json.dumps({'V5独立验收':'通过','字段变化':dict(changes)},ensure_ascii=False))
