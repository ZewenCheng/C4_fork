"""由已验收v5结果构建入口兼容修订包，分别绑定运行源码和封装源码。"""
from pathlib import Path
import hashlib,json,shutil,tarfile,sys,os
VERSION='v5-facts-rag-20260911-r2'
old=Path('/workspace/work/c4-submission-versions/v5-facts-rag-20260911')
root=Path('/workspace/work/c4-v5-entryfix-20260911-r2')
fixed=old.parent/VERSION
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def inventory(root):
    rows=[]
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
        if p.is_file():rows.append({'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    return rows
payload=json.load(sys.stdin);assert payload['version']==VERSION
fix=read(root/'entry_fix_validation.json');assert fix['退出码']==0 and fix['Python推理源码全部不变']
assert inventory(Path('/workspace/result'))==read(old/'old_official_inventory.json')['files']
for name,digest in fix['python_files'].items():assert sha(root/'code'/name)==sha(old/'submit/code'/name)==digest
design=Path(payload['design_pdf']);assert design.resolve().is_relative_to(root) and sha(design)==payload['design_sha256']
fixed.mkdir(exist_ok=False);submit=fixed/'submit'
shutil.copytree(root/'code',submit/'code');os.chmod(submit/'code/run',0o755)
(submit/'design').mkdir();shutil.copy2(design,submit/'design/智能体设计方案_v5.pdf')
shutil.copytree(old/'submit/result',submit/'result')
shutil.copytree(old/'run_snapshot',fixed/'run_snapshot')
assert inventory(fixed/'run_snapshot')==inventory(old/'run_snapshot')
for name in ['asset_manifest.json','old_official_inventory.json','independent_verification.json','verified_snapshot.json']:
    shutil.copy2(old/name,fixed/name)
shutil.copy2(root/'entry_fix_validation.json',fixed/'entry_fix_validation.json')
files=inventory(submit);archive=fixed/('c4-'+VERSION+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for row in files:tar.add(submit/row['path'],arcname=row['path'],recursive=False)
with tarfile.open(archive) as tar:
    assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}=={r['path']:r['sha256'] for r in files}
manifest=read(old/'package_manifest.json')
manifest.update(version=VERSION,github_tag='c4-'+VERSION,files=files,archive=str(archive),archive_sha256=sha(archive),run_snapshot=str(fixed/'run_snapshot'),
    source_manifest_sha256=sha(submit/'code/source_manifest.json'),run_source_manifest_sha256=fix['原始运行源码清单SHA256'],
    entry_fix_validation_sha256=sha(fixed/'entry_fix_validation.json'),parent_version=read(old/'package_manifest.json')['version'],
    post_run_change='仅run包装器和对应源码清单改变；SDK初始化移到nounset启用前。Python推理字节、模型、结果、计时未改，完整330图来自本日首轮新运行，未伪称修复后重新推理。')
save(fixed/'package_manifest.json',manifest)
validation={'状态':'v5 r2入口与逐成员摘要验收通过','文件数':len(files),'清单SHA256':sha(fixed/'package_manifest.json'),'压缩包SHA256':sha(archive),
            '原始运行快照一致':True,'source_verified':True,'入口修复验证SHA256':sha(fixed/'entry_fix_validation.json')}
save(fixed/'package_validation.json',validation)
export={'package_manifest':manifest,'asset_manifest':read(fixed/'asset_manifest.json'),'package_validation':validation,
        'verification_summary':read(fixed/'independent_verification.json'),'entry_fix_validation':fix,'fixed_root':str(fixed)}
save(root/'public_export.json',export)
print(json.dumps({'状态':'v5 r2冻结包完成','文件数':len(files),'清单SHA256':validation['清单SHA256'],'压缩包SHA256':sha(archive)},ensure_ascii=False))
