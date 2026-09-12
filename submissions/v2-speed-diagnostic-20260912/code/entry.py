"""v2提速版独立330图完整入口；时间以Python入口time.time差值计秒。"""
import time
BEGIN=time.time()
import os,sys,json,hashlib,argparse,subprocess,uuid,collections
from pathlib import Path
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def verify_source(code):
 manifest=read(code/'source_manifest.json')
 assert {p.name:sha(p) for p in code.iterdir() if p.is_file() and p.name!='source_manifest.json'}==manifest
 return sha(code/'source_manifest.json')
def scan(root):
 from metadata_semantics import image_gps,nearest_bridge_name
 rows=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file() or p.suffix.lower() not in ['.jpg','.jpeg','.png','.bmp','.webp']:continue
  assert p.resolve().is_relative_to(root)
  rel=p.relative_to(root);cats=set(rel.parts)&{'桥梁','轨道'};assert len(cats)==1
  category=next(iter(cats));h=sha(p);sid=hashlib.sha256((rel.as_posix()+'\0'+h).encode()).hexdigest()
  bridge=nearest_bridge_name(image_gps(p),p.name)[0] if category=='桥梁' else ''
  rows.append({'sample_id':sid,'path':str(p),'image_sha256':h,'category':category,'metadata':{'questionCategory':category,'bridgeName':bridge,'filename':p.name}})
 assert rows and len({r['sample_id'] for r in rows})==len(rows)
 return rows
def main():
 a=argparse.ArgumentParser();a.add_argument('--input',type=Path,default=Path('/dataset/决赛数据集1/赛题4/线上测试集'));a.add_argument('--work',type=Path);a.add_argument('--output',type=Path);a.add_argument('--resume',action='store_true');a.add_argument('--check',action='store_true');args=a.parse_args()
 code=Path(__file__).resolve().parent;assets=Path('/workspace/work/c4-v2-speed-release-20260912-r1/assets');source=verify_source(code)
 for item in read(assets/'asset_manifest.json')['files']:
  p=Path(item['path']);assert p.exists() and sha(p)==item['sha256'],'资产缺失或摘要变化'
 rows=scan(args.input.resolve());assert len(rows)==330,'指定完整330图输入不符'
 assert not {r['image_sha256'] for r in rows}&{r['id'] for r in read(assets/'train_corpus.json')},'线上输入与训练语料内容重叠'
 work=(args.work or Path('/workspace/work')/('c4-v2-speed-run-'+uuid.uuid4().hex)).resolve()
 assert work.is_relative_to('/workspace/work') and work!=Path('/workspace/work')
 out=(args.output or work/'output').resolve();assert out.is_relative_to('/workspace')
 if args.check:print(json.dumps({'状态':'入口资产与330图预检通过','源码摘要':source},ensure_ascii=False));return
 if work.exists() and not args.resume:raise FileExistsError('工作目录已存在，请显式恢复或换新目录')
 if args.resume:
  assert read(work/'input_manifest.json')==rows
  assert read(work/'configuration.json')['source_manifest_sha256']==source
 else:
  work.mkdir();dump(work/'input_manifest.json',rows)
  dump(work/'configuration.json',{'version':'v2-speed-diagnostic-20260912','source_manifest_sha256':source,'sources':read(code/'source_manifest.json'),'asset_manifest_sha256':sha(assets/'asset_manifest.json'),'inputs':sha(work/'input_manifest.json'),'v6_adapter':sha(Path('/workspace/work/c4-v6-qwen-posttrain-20260911-v1/epoch-2/adapter_model.safetensors')),'adapter_disabled':True,'selected_dev_rating':'162/195','no_development_prediction_dependency':True,'timing_scope':'Python入口至完整推理结果写入，time.time差值秒；恢复调用单独留账'})
 os.environ['C4_RUN_ROOT']=str(work)
 if (work/'complete.json').exists():
  for n in ['result.json','infer_time.json']:assert sha(work/n)==read(work/'complete.json')['files'][n]
  print(json.dumps({'状态':'完成恢复只读核验通过','新模型调用':0},ensure_ascii=False));return
 for mode,receipt in [('rail','rail_complete.json'),('teacher','teacher_complete.json')]:
  if not (work/receipt).exists():
   dump(work/'progress.json',{'阶段':mode+'新前向'})
   with (work/(mode+'.log')).open('ab') as log:subprocess.run([sys.executable,'-B',str(code/'vision.py'),mode],stdout=log,stderr=log,check=True)
 rail=read(work/'v2_candidates.json');teacher=read(work/'teacher_candidates.json')
 for r in rows:r.update(v2_cv=rail.get(r['sample_id'],[]),teacher=teacher[r['sample_id']])
 dump(work/'inputs.json',rows)
 from runner import Runner
 run=Runner();run.run()
 from common import FIELDS
 results=[read(work/'results/v2'/(r['sample_id']+'.json')) for r in rows];finals=[r['final'] for r in results]
 assert all(list(f)==FIELDS and all(isinstance(v,str) for v in f.values()) and f['defectType'] and f['defectDescription'] and f['ratingScale(1-5)'] in ['','1','2','3','4','5'] and (f['defectType']!='完好' or f['ratingScale(1-5)']=='') for f in finals)
 dump(work/'result.json',finals)
 attempts=read(work/'attempt_times.json') if (work/'attempt_times.json').exists() else []
 attempts.append({'秒':time.time()-BEGIN,'完成':True});dump(work/'attempt_times.json',attempts)
 infer=sum(x['秒'] for x in attempts);dump(work/'infer_time.json',{'infer_time':infer})
 assert verify_source(code)==source
 calls=[read(p) for p in (work/'calls').glob('*.json')]
 summary={'样本数':len(rows),'类别':dict(collections.Counter(r['category'] for r in rows)),'协议问题':dict(collections.Counter(k for r in results for k in r['issues'])),'类型分布':dict(collections.Counter(f['defectType'] for f in finals)),'评级分布':dict(collections.Counter(f['ratingScale(1-5)'] for f in finals)),'infer_time秒':infer,'新LLM调用':len(calls),'生成调用秒':sum(c['seconds'] for c in calls),'输出词元':sum(c['output_tokens'] for c in calls),'类型继承开发结果':False,'所有结果留平台':True}
 dump(work/'summary.json',summary)
 out.mkdir(parents=True,exist_ok=True)
 for n in ['result.json','infer_time.json']:
  if (out/n).exists():raise FileExistsError('结果目标已存在，禁止覆盖')
  (out/n).write_bytes((work/n).read_bytes())
 dump(work/'complete.json',{'files':{n:sha(work/n) for n in ['result.json','infer_time.json','summary.json','inputs.json','configuration.json']},'source_manifest_sha256':source,'状态':'330图完整新推理完成，待独立验收'})
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if os.environ.get('C4_RUN_ROOT'):
   work=Path(os.environ['C4_RUN_ROOT']);p=work/'attempt_times.json';old=read(p) if p.exists() else [];old.append({'秒':time.time()-BEGIN,'完成':False,'异常':type(e).__name__});dump(p,old)
  raise
