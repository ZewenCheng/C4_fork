"""流式核验历史归档与已有展开快照；只读，不删除或提取任何成员。"""
import json,tarfile,time,hashlib,os
from pathlib import Path,PurePosixPath
from download_assets import ROOT,dump

def digest(stream):
    h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    os.nice(10)
    folder=Path('/workspace/work/road-infrastructure-finals-cloud/.runtime/result-backups/result-before-code-only-contract/publish-before-correction')
    archive=folder/'cslq-lqwz-fullweight-v2-20260904115547.tar.gz'
    initial=archive.stat();records=[];bad=0;matched=0;total=0
    status=ROOT/'historical_archive_audit_status.json'
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            rel=PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('归档路径越界')
            target=folder.joinpath(*rel.parts)
            record={'member':member.name,'bytes':member.size,'mode':member.mode,'type':member.type.decode('ascii','replace')}
            if member.isdir():record['equivalent']=target.is_dir()
            elif member.isfile():
                with tf.extractfile(member) as stream:record['archive_sha256']=digest(stream)
                resolved=target.resolve()
                if not resolved.is_relative_to(Path('/workspace/work').resolve()):raise ValueError('展开快照引用超出官方工作区')
                if target.is_file() and target.stat().st_size==member.size:
                    before=target.stat()
                    with target.open('rb') as stream:record['existing_sha256']=digest(stream)
                    after=target.stat()
                    record['equivalent']=record['archive_sha256']==record['existing_sha256'] and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                else:record['equivalent']=False
                total+=member.size
            elif member.issym():record['equivalent']=target.is_symlink() and os.readlink(target)==member.linkname
            else:record['equivalent']=False
            matched+=int(record['equivalent']);bad+=int(not record['equivalent']);records.append(record)
            if len(records)%20==0:dump(status,{'status':'auditing_read_only','members':len(records),'matched':matched,'unmatched':bad,'uncompressed_bytes_read':total,'time':time.time()})
    final=archive.stat()
    if (initial.st_size,initial.st_mtime_ns)!=(final.st_size,final.st_mtime_ns):raise ValueError('归档审计期间变化')
    receipt=ROOT/'backups/historical_archive_equivalence.json'
    dump(receipt,{'archive':str(archive),'archive_bytes':initial.st_size,'archive_mtime_ns':initial.st_mtime_ns,'members':records,'all_contents_equivalent':bad==0,'deleted':False,'time':time.time()})
    dump(status,{'status':'read_only_audit_complete','members':len(records),'matched':matched,'unmatched':bad,'archive_bytes':initial.st_size,'receipt':str(receipt),'deleted':False,'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'historical_archive_audit_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
