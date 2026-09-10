"""保留首版失败证据，仅修SDK包装器，验证干净环境入口与推理Python逐字节不变。"""
from pathlib import Path
import json,hashlib,shutil,subprocess,os,time
old=Path('/workspace/work/c4-submission-versions/v5-facts-rag-20260911')
root=Path('/workspace/work/c4-v5-entryfix-20260911-r2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def inventory(root):return [{'path':p.relative_to(root).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file()]
assert inventory(Path('/workspace/result'))==read(old/'old_official_inventory.json')['files'],'首版回滚不完整'
failure={'状态':'首版正式入口自检失败，旧伙伴目录完整回滚','原因':'SDK envsetup.sh在nounset环境读取未定义LD_LIBRARY_PATH','模型运行是否失败':False,'正式采用':False,'旧作品逐文件核验':True}
with (old/'publication_failure.json').open('x') as f:json.dump(failure,f,ensure_ascii=False,indent=2)
root.mkdir(exist_ok=False)
code=root/'code';shutil.copytree(old/'submit/code',code)
run=code/'run';raw=run.read_bytes()
assert b'set -euo pipefail\nsource /usr/local/PPU_SDK/envsetup.sh >/dev/null\n' in raw
run.write_bytes(raw.replace(b'set -euo pipefail\nsource /usr/local/PPU_SDK/envsetup.sh >/dev/null\n',b'set -eo pipefail\nsource /usr/local/PPU_SDK/envsetup.sh >/dev/null\nset -u\n',1))
os.chmod(run,0o755)
manifest={p.name:sha(p) for p in sorted(code.iterdir()) if p.is_file() and p.name!='source_manifest.json'}
(code/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
py_old={p.name:sha(p) for p in (old/'submit/code').glob('*.py')}
py_new={p.name:sha(p) for p in code.glob('*.py')}
assert py_old==py_new
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('LD_LIBRARY_PATH',None)
started=time.time()
check=subprocess.run([str(run),'--check'],capture_output=True,text=True,env=env,timeout=600)
(root/'干净环境入口检查.log').write_text(check.stdout+'\n'+check.stderr)
receipt={'状态':'干净SDK环境入口通过' if check.returncode==0 else '入口失败','退出码':check.returncode,'验证秒':time.time()-started,
         '预先设置LD_LIBRARY_PATH':False,'Python推理源码全部不变':True,'python_files':py_new,
         '原run_sha256':sha(old/'submit/code/run'),'修复run_sha256':sha(run),'修复源码清单SHA256':sha(code/'source_manifest.json'),
         '原始运行源码清单SHA256':read(old/'package_manifest.json')['source_manifest_sha256'],
         '修复范围':'SDK脚本在nounset启用前初始化，之后恢复严格变量检查；模型、Python推理、结果和计时保持。'}
(root/'entry_fix_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False));raise SystemExit(check.returncode)
