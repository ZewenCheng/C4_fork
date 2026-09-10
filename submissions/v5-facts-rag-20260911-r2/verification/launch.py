"""v5唯一启动与超时保护，平台日志保留中文。"""
from pathlib import Path
import subprocess,json,sys,os,time
root=Path('/workspace/work/c4-v5-release-20260911-v1')
claim=root/'launch_claim.json'
with claim.open('x') as f:json.dump({'pid':os.getpid(),'started':time.time()},f)
env=dict(os.environ,PYTHONPATH=str(root/'code')+':'+str(root)+':/workspace/work/c4-facts-rag-20260910-v1/deps',PYTHONDONTWRITEBYTECODE='1')
status={'退出码':1,'状态':'未完成'}
try:
    with (root/'合同测试.log').open('x') as log:
        subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(root/'tests'),'-v'],env=env,stdout=log,stderr=log,check=True,timeout=300)
    with (root/'入口预检.log').open('x') as log:
        subprocess.run(['bash',str(root/'code/run'),'--check'],env=env,stdout=log,stderr=log,check=True,timeout=600)
    with (root/'完整推理.log').open('x') as log:
        subprocess.run(['bash',str(root/'code/run'),'--work',str(root/'inference'),'--output',str(root/'output')],env=env,stdout=log,stderr=log,check=True,timeout=3600)
    status={'退出码':0,'状态':'完成'}
except BaseException as e:
    status={'退出码':getattr(e,'returncode',1),'状态':'失败','类型':type(e).__name__}
finally:
    (root/'launcher_status.json').write_text(json.dumps(status,ensure_ascii=False))
if status['退出码']==0:
    with (root/'独立验收.log').open('x') as log:
        p=subprocess.run([sys.executable,'-B',str(root/'verify_v5.py')],env=env,stdout=log,stderr=log,timeout=600)
    (root/'verification_status.json').write_text(json.dumps({'退出码':p.returncode,'状态':'通过' if p.returncode==0 else '失败'},ensure_ascii=False))
