"""独立只读检查提交压缩包、固定目录、双JSON和代码清单，只保存聚合核验。"""
import collections
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile
import sys

ROOT=Path('/workspace/work/c4-submission-versions/contract-v2-metadata2-20260909')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(value,why):
    if not value:raise ValueError(why)

def main():
    manifest=json.loads((ROOT/'package_manifest.json').read_text())
    require(sha(ROOT/'package_manifest.json')==json.load(sys.stdin)['manifest_sha256'],'固定清单变化')
    archive=Path(manifest['archive']);require(sha(archive)==manifest['archive_sha256'],'压缩包摘要不符')
    expected={item['path']:item for item in manifest['files']}
    payload={}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();require(len({m.name for m in members})==len(members),'压缩包成员重名')
        require({PurePosixPath(m.name).parts[0] for m in members}=={'code','design','result'},'压缩包顶层不符')
        for member in members:
            rel=PurePosixPath(member.name)
            require(not rel.is_absolute() and '..' not in rel.parts and not member.issym() and not member.islnk(),'压缩包路径或链接不符')
            if member.isdir():continue
            require(member.isfile() and member.name in expected,'压缩包含未登记文件')
            item=expected[member.name];raw=tar.extractfile(member).read();digest=hashlib.sha256(raw).hexdigest()
            require(digest==item['sha256']==sha(ROOT/'submit'/member.name),'压缩内容与固定目录不一致')
            require(member.mode==int(item['mode'],8) and len(raw)==item['bytes'],'压缩包权限或大小不符')
            payload[member.name]=raw
    require(set(payload)==set(expected) and len(payload)==32,'提交文件集合不符')
    require({p.name for p in (ROOT/'submit').iterdir()}=={'code','design','result'},'固定目录顶层不符')
    require(len([name for name in payload if name.startswith('design/')])==1,'方案设计书数量不符')
    require({name for name in payload if name.startswith('result/')}=={'result/result.json','result/infer_time.json'},'结果目录不是双JSON')
    require(payload['design/智能体设计方案_已核验.pdf'].startswith(b'%PDF-'),'方案设计书不是PDF')
    require((ROOT/'submit/code/run').stat().st_mode & 0o111,'run缺少执行权限')
    for name,raw in payload.items():
        if name.startswith('code/'):
            filename=PurePosixPath(name).name
            require(filename in {'run','MANIFEST.sha256'} or name.endswith(('.py','.json')),'代码目录含额外资产')
            if name.endswith('.py'):compile(raw,name,'exec')
    code_manifest={line.split('  ',1)[1]:line.split('  ',1)[0] for line in payload['code/MANIFEST.sha256'].decode().splitlines()}
    require(set(code_manifest)=={name[5:] for name in payload if name.startswith('code/') and name!='code/MANIFEST.sha256'},'代码清单成员集合不符')
    for name,digest in code_manifest.items():require(hashlib.sha256(payload['code/'+name]).hexdigest()==digest,'代码清单与压缩内容不符')
    results=json.loads(payload['result/result.json']);timing=json.loads(payload['result/infer_time.json'])
    fields={'questionCategory','bridgeName','defectLocation','filename','defectType','defectDescription','ratingScale(1-5)'}
    require(len(results)==330 and len({(r['questionCategory'],r['filename']) for r in results})==330,'330图结果身份重复或缺失')
    require(all(set(row)==fields and all(isinstance(v,str) for v in row.values()) for row in results),'七字段结构不符')
    require(type(timing)==dict and set(timing)=={'infer_time'} and type(timing['infer_time']) is float and timing['infer_time']==manifest['infer_time'],'提交计时合同不符')
    require(all(r['bridgeName'] not in {'桥梁','轨道'} and r['defectLocation'] != '全桥' and not r['defectLocation'].startswith('图像') for r in results),'实体及结构位置语义不符')
    require(hashlib.sha256(payload['result/result.json']).hexdigest()=='ecc86813d9d5c9eeccb1a7942a98e07b717f7e77d9ac5102acd5009e4d7bc1d0','结果并非已核验修复版本')
    require(manifest['entry_validation']['full_cold_rerun'] is False and manifest['timing']['new_full_inference_time_seconds'] is None,'计时性质不符')
    empty={field:sum(row[field]=='' for row in results) for field in sorted(fields)}
    grades=dict(collections.Counter(row['ratingScale(1-5)'] for row in results))
    require(empty==manifest['field_summary'] and grades==manifest['grade_summary'],'聚合结果与已核验版本不一致')
    value={'状态':'提交压缩包与固定目录独立核验通过','版本':manifest['version'],'总文件数':len(payload),'代码文件数':29,'设计书文件数':1,
           '结果文件数':2,'图片数':330,'七字段空值数':empty,'评级分布':grades,'infer_time':timing,'计时范围':manifest['timing_transformation'],
           '压缩包':str(archive),'压缩包SHA':sha(archive),'压缩包字节':archive.stat().st_size,
           '结果SHA':hashlib.sha256(payload['result/result.json']).hexdigest(),'计时SHA':hashlib.sha256(payload['result/infer_time.json']).hexdigest(),
           '设计书SHA':hashlib.sha256(payload['design/智能体设计方案_已核验.pdf']).hexdigest(),'清单SHA':sha(ROOT/'package_manifest.json'),
           'run可执行':True,'代码无模型及缓存':True,'入口检查':manifest['entry_validation'],'正式评测':False}
    target=ROOT/'package_validation.json'
    if target.exists():require(json.loads(target.read_text())==value,'已有不同独立核验记录')
    else:target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'状态':value['状态'],'文件数':32,'图片数':330,'验证记录SHA':sha(target),'压缩包SHA':sha(archive),'推理计时':timing},ensure_ascii=False))

if __name__=='__main__':main()
