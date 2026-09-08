"""官方训练集建立可扩展词表与按图像摘要隔离的清单；逐样本资料仅留平台。"""
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from download_assets import ROOT, dump, sha

def main():
    out=ROOT/'data';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        print('已有固定训练清单，复用',flush=True);return
    train=Path('/dataset/决赛数据集1/赛题4/训练集').resolve()
    paths=defaultdict(list)
    for p in train.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:paths[p.name].append(p)
    records=[]
    for source in ['桥梁数据.json','轨道数据.json']:
        rows=json.loads((train/source).read_text(encoding='utf-8-sig'))
        for r in rows:
            choices=paths.get(r['filename'],[])
            bridge=re.sub(r'[（(]?(?:左右幅|左幅|右幅)[）)]?','',r['bridgeName']).strip()
            if len(choices)>1:
                choices=[p for p in choices if p.parent.name==bridge or (source=='轨道数据.json' and '轨道' in p.parts)]
            if len(choices)!=1:raise ValueError('训练标注无法唯一定位图片；须按记录修正映射')
            p=choices[0].resolve()
            if not p.is_relative_to(train):raise ValueError('训练输入越界')
            records.append({'path':str(p),'source':source,'annotation':r,'group':p.parent.name})
    with ThreadPoolExecutor(max_workers=12) as pool:
        hashes=list(pool.map(lambda r:sha(Path(r['path'])),records))
    buckets=defaultdict(list)
    for r,h in zip(records,hashes):buckets[h].append(r)
    # 同一图像有多条监督时保留其集合，不擅自选择一条或当作独立样本。
    raw_classes=sorted({str(r['annotation']['defectType']).strip() for r in records})
    atom_map={s:[v.strip() for v in re.split(r'[+、]',s) if v.strip()] for s in raw_classes}
    atoms=sorted({a for v in atom_map.values() for a in v})
    bridge_groups=sorted({r['group'] for r in records if r['source']=='桥梁数据.json'},key=lambda x:hashlib.sha256(('20260906:'+x).encode()).hexdigest())
    hold_groups=set(bridge_groups[:max(1,round(len(bridge_groups)*.2))])
    result=[]
    for h,bucket in sorted(buckets.items()):
        hold=any(r['group'] in hold_groups for r in bucket)
        if all(r['source']=='轨道数据.json' for r in bucket):hold=int(h[:8],16)%5==0
        result.append({'sample_id':h,'path':bucket[0]['path'],'annotations':[r['annotation'] for r in bucket],
                       'groups':sorted({r['group'] for r in bucket}),
                       'label_ids':sorted({raw_classes.index(str(r['annotation']['defectType']).strip()) for r in bucket}),
                       'split':'holdout' if hold else 'train'})
    vocab={'version':1,'extensible':True,'raw_classes':raw_classes,'atoms':atoms,'raw_to_atoms':atom_map,
           'semantics':'原始标注字符串分类与组合映射；未提及病害不等同确认阴性，类别数不锁死'}
    dump(out/'vocabulary.json',vocab)
    dump(out/'manifest.json',result)
    audit={'原标注数':len(records),'唯一图像':len(result),'原始标签数':len(raw_classes),'拆分词项数':len(atoms),
           '分区':dict(Counter(r['split'] for r in result)), '桥梁保留组数':len(hold_groups),
           '输入':str(train),'标注摘要':{n:sha(train/n) for n in ['桥梁数据.json','轨道数据.json']},
           '限制':'轨道按图像SHA分组；未声称跨图近重复已完全隔离。验证记录不用于教师或训练。'}
    dump(out/'audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
