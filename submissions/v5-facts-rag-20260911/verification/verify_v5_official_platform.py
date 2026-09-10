"""独立读回v5官方目录、旧版本保留、归档和本轮计时绑定。"""
from pathlib import Path
import hashlib,json,tarfile
VERSION='v5-facts-rag-20260911'
fixed=Path('/workspace/work/c4-submission-versions')/VERSION
official=Path('/workspace/result')
backup=fixed.parent/('before-'+VERSION)
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inventory(root):
    result=[]
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
        if p.is_file() and p.name!='preservation_manifest.json':result.append({'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    return result
manifest=read(fixed/'package_manifest.json');published=read(fixed/'publication_receipt.json')
expected=sorted(manifest['files'],key=lambda x:x['path'])
assert inventory(official)==inventory(fixed/'submit')==expected
old=read(fixed/'old_official_inventory.json')['files']
assert inventory(backup)==old
assert read(backup/'preservation_manifest.json')['files']==old
assert published['GitHub']['remote_verified'] and published['GitHub']['tag']=='c4-'+VERSION
assert published['package_manifest_sha256']==sha(fixed/'package_manifest.json')
assert published['entry_check_exit_code']==0 and published['比赛提交或评测'] is False
assert {p.name for p in official.iterdir()}=={'code','design','result'}
assert {p.name for p in (official/'result').iterdir()}=={'result.json','infer_time.json'}
assert len(list((official/'design').glob('*.pdf')))==1
assert (official/'code/run').stat().st_mode & 0o111
results=read(official/'result/result.json');assert len(results)==330
assert all(set(r)=={'questionCategory','bridgeName','defectLocation','filename','defectType','defectDescription','ratingScale(1-5)'} and all(isinstance(v,str) for v in r.values()) for r in results)
timing=read(official/'result/infer_time.json')
assert timing=={'infer_time':manifest['inference_seconds']} and 0<timing['infer_time']<=660
assert sha(official/'result/result.json')==sha(fixed/'run_snapshot/result.json')
assert sha(official/'result/infer_time.json')==sha(fixed/'run_snapshot/infer_time.json')
archive=Path(manifest['archive']);assert sha(archive)==manifest['archive_sha256']
with tarfile.open(archive) as tar:
    assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers() if m.isfile()}=={r['path']:r['sha256'] for r in expected}
report={'状态':'v5官方作品与固定压缩包、原始运行及旧版保留独立验收通过','版本':VERSION,'新文件数':len(expected),'旧文件数':len(old),
        '样本数':330,'infer_time':timing['infer_time'],'端到端秒':manifest['end_to_end_seconds'],
        '结果SHA256':sha(official/'result/result.json'),'计时SHA256':sha(official/'result/infer_time.json'),
        '压缩包':str(archive),'压缩包SHA256':sha(archive),'清单SHA256':sha(fixed/'package_manifest.json'),
        'GitHub':published['GitHub'],'比赛上传或评测':False}
with (fixed/'official_independent_verification.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False))
