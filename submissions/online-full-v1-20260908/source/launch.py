"""启动本作业自己的脱离SSH进程；不停止或覆盖其他任务。"""
import json
import os
from pathlib import Path
import subprocess
import sys
from download_assets import ROOT,dump

name=sys.argv[1]
if name not in {'prepare_data','train_dino','train_tabular','reclaim_duplicates'}:raise ValueError('未审阅任务名')
pidfile=ROOT/(name+'.pid.json')
if pidfile.exists():
    old=json.loads(pidfile.read_text())
    cmd=Path(f"/proc/{old['pid']}/cmdline")
    if cmd.exists() and (name+'.py').encode() in cmd.read_bytes():
        print('任务已运行',old['pid']);sys.exit(0)
with (ROOT/(name+'.log')).open('ab',buffering=0) as log:
    proc=subprocess.Popen([sys.executable,'-u',str(ROOT/(name+'.py')),*sys.argv[2:]],cwd=ROOT,
                          stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
dump(pidfile,{'pid':proc.pid,'name':name})
print(json.dumps({'已启动进程':name,'pid':proc.pid},ensure_ascii=False))
