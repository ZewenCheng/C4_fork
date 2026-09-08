"""逐文件校验旧打包副本与当前原权重相等，仅将重复副本替换为可恢复引用。"""
from pathlib import Path
import os
import time
from download_assets import ROOT,dump,sha
SOURCE=Path('/workspace/work/road-infrastructure-finals-cloud').resolve()
BACKUP=(SOURCE/'.runtime/result-backups/result-before-code-only-contract/publish-before-correction/code/models').resolve()
CANONICAL=(SOURCE/'models').resolve()
if not BACKUP.is_relative_to(SOURCE) or not CANONICAL.is_relative_to(SOURCE):raise ValueError('路径越界')
records=[];freed=0
for p in sorted(BACKUP.rglob('*')):
    if p.is_symlink() or not p.is_file():continue
    rel=p.relative_to(BACKUP);target=(CANONICAL/rel).resolve()
    if not target.is_relative_to(CANONICAL) or not target.is_file() or p.stat().st_size!=target.stat().st_size:continue
    a=sha(p);b=sha(target)
    if a!=b:continue
    size=p.stat().st_size
    records.append({'backup':str(p),'canonical':str(target),'sha256':a,'bytes':size})
    # 先持久记录恢复位置再移除字节副本；软链接保持旧目录的读取与恢复路径。
    dump(ROOT/'dedup_receipt.json',{'status':'running','records':records,'freed_bytes':freed,'time':time.time()})
    tmp=p.with_name(p.name+'.dedup-link')
    if tmp.exists() or tmp.is_symlink():raise ValueError('存在未处理临时引用')
    tmp.symlink_to(target)
    os.replace(tmp,p)
    freed+=size
    print('已核验去重字节',freed,flush=True)
dump(ROOT/'dedup_receipt.json',{'status':'complete','records':records,'freed_bytes':freed,'time':time.time(),
    '恢复要求':'规范原权重必须保留且不原地修改；备份链接失效时按此清单SHA恢复。原打包归档和独有代码未删除。'})
