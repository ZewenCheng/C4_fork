"""完整核对旧传输tar与已解包副本，仅删除冗余传输包；不将其数据用于训练。"""
from pathlib import Path,PurePosixPath
import hashlib,json,os,subprocess,sys,tarfile,time
from download_assets import ROOT,dump,sha
archive=Path('/workspace/data/c4-preliminary-test-20260905.tar').resolve()
allowed=Path('/workspace/data/初赛测试集').resolve()
if archive!=Path('/workspace/data/c4-preliminary-test-20260905.tar') or not allowed.is_relative_to(Path('/workspace/data')):raise ValueError('清理范围不匹配')
if not archive.is_file():raise FileNotFoundError('传输包不存在，未重复清理')
records=[]
with tarfile.open(archive,'r|') as tf:
    for member in tf:
        rel=PurePosixPath(member.name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('归档路径越界')
        target=(Path('/workspace/data')/member.name).resolve()
        if not target.is_relative_to(allowed):raise ValueError('归档成员不属于指定解包目录')
        if member.isdir():continue
        if not member.isfile() or not target.is_file() or target.is_symlink() or target.stat().st_size!=member.size:raise ValueError('归档与已有解包副本不匹配')
        h=hashlib.sha256()
        with tf.extractfile(member) as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=sha(target):raise ValueError('解包副本摘要不匹配，未删除归档')
        records.append({'relative_path':member.name,'sha256':h.hexdigest(),'bytes':member.size})
        if len(records)%200==0:print('已核对传输成员数',len(records),flush=True)
receipt={'status':'verified_before_delete','archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
         'canonical_root':str(allowed),'files':records,'purpose':'仅核验清理重复传输包，不训练或查看图片内容','time':time.time()}
dump(ROOT/'transport_cleanup.cloud_only.json',receipt)
archive.unlink()
receipt['status']='duplicate_transport_deleted';dump(ROOT/'transport_cleanup.cloud_only.json',receipt)
print('已释放重复传输字节',receipt['archive_bytes'],flush=True)
pidfile=ROOT/'download_teacher.pid.json'
old=json.loads(pidfile.read_text())['pid'] if pidfile.exists() else -1
cmd=Path(f'/proc/{old}/cmdline')
if not cmd.exists() or b'download_teacher.py' not in cmd.read_bytes():
    with (ROOT/'download_teacher.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'download_teacher.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
    dump(pidfile,{'pid':p.pid});print('教师下载已恢复',p.pid,flush=True)
