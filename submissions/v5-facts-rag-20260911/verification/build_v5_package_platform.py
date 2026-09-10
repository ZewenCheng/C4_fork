"""仅在v5独立验收通过后冻结提交包及聚合清单，不切换官方作品。"""
from pathlib import Path
import hashlib,json,shutil,tarfile,sys,os,subprocess,platform,importlib.metadata
VERSION='v5-facts-rag-20260911'
ROOT=Path('/workspace/work/c4-v5-release-20260911-v1')
FIXED=Path('/workspace/work/c4-submission-versions')/VERSION
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def inventory(root):
    result=[]
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
        if p.is_file():result.append({'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    return result
payload=json.load(sys.stdin)
assert payload['version']==VERSION
assert read(ROOT/'verification_status.json')['退出码']==0
evidence=read(ROOT/'independent_verification.json')
summary=evidence['候选汇总']
assert summary['样本数']==330 and summary['660秒速度门'] and evidence['输入集合与旧330图一致']
run=ROOT/'inference'
assert sha(run/'complete.json')==evidence['完整凭证SHA256']
sys.path.insert(0,str(ROOT/'code'))
from v5_entry import verify_code,BASE_ROOT,HEAD_ROOT,GRADE_ROOT
assert verify_code(ROOT/'code')==read(run/'configuration.json')['source_manifest_sha256']
design=Path(payload['design_pdf'])
assert design.resolve().is_relative_to(ROOT) and sha(design)==payload['design_sha256']
assert not FIXED.exists()
FIXED.mkdir()
submit=FIXED/'submit'
shutil.copytree(ROOT/'code',submit/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
os.chmod(submit/'code/run',0o755)
(submit/'design').mkdir();shutil.copy2(design,submit/'design/智能体设计方案_v5.pdf')
(submit/'result').mkdir()
for name in ['result.json','infer_time.json']:shutil.copy2(run/name,submit/'result'/name)
assert verify_code(submit/'code')==verify_code(ROOT/'code')
# 运行原始记录留官方冻结目录，逐图结果不得导出公开仓库。
shutil.copytree(run,FIXED/'run_snapshot')
assert inventory(FIXED/'run_snapshot')==inventory(run)
for name in ['independent_verification.json','verified_snapshot.json','launcher_status.json','verification_status.json']:
    shutil.copy2(ROOT/name,FIXED/name)
rag=Path('/workspace/work/c4-facts-rag-20260910-v1/rag')
assets=read(run/'assets.json')
required=[BASE_ROOT/'model.safetensors',BASE_ROOT/'config.json',BASE_ROOT/'preprocessor_config.json',
          HEAD_ROOT/'multiview_model.npz',HEAD_ROOT/'manifest.json',GRADE_ROOT/'grade_model.joblib',GRADE_ROOT/'manifest.json',
          rag/'knowledge_retriever.py',rag/'rag_local.py',rag/'source_receipt.json',rag/'bridge_knowledge_v1/manifest.json']
for folder in [rag/'bridge_knowledge_v1',rag/'.models']:
    required.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for name in assets['rag']['tokenizer_files_sha256']:required.append(Path('/model/Qwen3.6-27B')/name)
asset_files=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(required))]
assert next(x['sha256'] for x in asset_files if x['path']==str(BASE_ROOT/'model.safetensors'))==assets['vision']['base_sha256']
asset_manifest={'version':VERSION,'files':asset_files,'dependencies_root':'/workspace/work/c4-facts-rag-20260910-v1/deps',
                'environment':{'python':platform.python_version(),'platform':platform.platform()},
                'runtime_requirement':'平台PPU SDK、PyTorch及RAG隔离依赖；使用code/run进入既有SDK环境。',
                'restore_policy':'按精确路径和SHA校验既有外置资产；缺失时停止，不联网自动换模型。原始模型与索引保留官方工作区。'}
save(FIXED/'asset_manifest.json',asset_manifest)
old=inventory(Path('/workspace/result'))
old_fixed=read(Path('/workspace/work/c4-submission-versions/cv-multiview-partner-20260910/package_manifest.json'))
assert old==sorted([{k:r[k] for k in ['path','sha256','bytes']} for r in old_fixed['files']],key=lambda r:r['path'])
save(FIXED/'old_official_inventory.json',{'files':old})
files=inventory(submit)
archive=FIXED/('c4-'+VERSION+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for row in files:tar.add(submit/row['path'],arcname=row['path'],recursive=False)
manifest={'version':VERSION,'github_tag':'c4-'+VERSION,'files':files,'archive':str(archive),
          'archive_sha256':sha(archive),'run_snapshot':str(FIXED/'run_snapshot'),
          'run_receipt_sha256':sha(run/'complete.json'),'source_manifest_sha256':verify_code(submit/'code'),
          'asset_manifest_sha256':sha(FIXED/'asset_manifest.json'),'inference_seconds':summary['纯模型秒'],
          'end_to_end_seconds':summary['端到端秒'],'timing_mode':'model_inference_only',
          'timing_transformation':'本次同步模型计算区间并集，单位秒；加载、预处理、传输、事实编码、检索整理、写入不计入，端到端另列。',
          'score':None,'quality_gain_verified':False,'selection_basis':'用户明确指定轻量事实链加RAG，沿用伙伴版分类，封装为v5。'}
save(FIXED/'package_manifest.json',manifest)
# 压缩包逐字节读回，不能用创建成功代替验包。
with tarfile.open(archive) as tar:
    members=tar.getmembers();assert len(members)==len(files)
    actual={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members if m.isfile()}
    assert actual=={r['path']:r['sha256'] for r in files}
    assert next(m for m in members if m.name=='code/run').mode & 0o111
assert {p.name for p in submit.iterdir()}=={'code','design','result'}
assert {p.name for p in (submit/'result').iterdir()}=={'result.json','infer_time.json'}
assert len(list((submit/'design').iterdir()))==1
validation={'状态':'v5冻结包压缩成员与逐文件摘要验收通过','文件数':len(files),'清单SHA256':sha(FIXED/'package_manifest.json'),
            '压缩包SHA256':sha(archive),'原始运行快照一致':True,'source_verified':True}
save(FIXED/'package_validation.json',validation)
export={'package_manifest':manifest,'asset_manifest':asset_manifest,'package_validation':validation,
        'verification_summary':evidence,'fixed_root':str(FIXED)}
save(ROOT/'public_export.json',export)
print(json.dumps(export,ensure_ascii=False))
