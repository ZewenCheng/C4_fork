"""三版本诊断链固定配置与原子证据。"""
import json,hashlib,os
from pathlib import Path
ROOT=Path(os.environ['C4_RUN_ROOT']); ASSETS=Path('/workspace/work/c4-v2-speed-release-20260912-r1/assets')
OLD=Path('/workspace/work/c4-v6-qwen-posttrain-20260911-v1')
RAG=ASSETS/'rag'
MODEL=Path('/model/Qwen3.6-27B')
VERSIONS=['v2']
FIELDS=['questionCategory','bridgeName','defectLocation','filename','defectType','defectDescription','ratingScale(1-5)']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
def parse(t):
 try:return json.loads(t.strip().removeprefix('```json').removesuffix('```').strip())
 except ValueError:return None
def assemble(raw,metadata,healthy=True):
 raw=raw if isinstance(raw,dict) else {};out={k:'' for k in FIELDS}
 for k in ['questionCategory','bridgeName','filename']:out[k]=metadata.get(k,'')
 for k in ['defectType','defectDescription','ratingScale(1-5)']:out[k]=raw.get(k,'') if isinstance(raw.get(k,''),str) else ''
 if healthy and out['defectType']=='完好':out['ratingScale(1-5)']=''
 return out
