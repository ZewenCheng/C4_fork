"""本轮研究候选真实RAG旁路；不产生规范等级，不读取历史预测缓存。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

RAG_ROOT = Path('/workspace/work/c4-facts-rag-20260910-v1/rag')
DEPS_ROOT = RAG_ROOT.parent / 'deps'
TOKENIZER_ROOT = Path('/model/Qwen3.6-27B')
REQUIRED_SYS_PATH = [str(DEPS_ROOT), str(RAG_ROOT)]
MANIFEST_SHA256 = '5a4676ee7e1b6585c49b94bb7b26a5126368ced2730d1556af1d76aea0fcae96'
WRAPPER_SHA256 = '71d061977a663cbd3f10c8e1d506bc2d99aed16cdd58ff5f6cc82a977c830019'
BACKEND_SHA256 = '2e5212d0cdfb6971513ecb068442dcd02200f313812a616fcd81b5e20c4eb37e'
SOURCES = [('456d6a5c35d50152', 'H21'), ('3bdc4ec41025a978', 'CJJ233')]
TOKENIZER_SHA256 = {
 'merges.txt':'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d',
 'tokenizer.json':'5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
 'tokenizer_config.json':'5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b',
 'vocab.json':'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003'}


def digest(value):
 return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def known(facts,name):
 value=facts.get(name,{})
 return value.get('value') if isinstance(value,dict) and value.get('state') in {'observed','estimated'} else None


class ResearchRagAdapter:
 def __init__(self,retriever,*,assets):
  self.retriever=retriever
  self.assets=copy.deepcopy(assets)
  self.fingerprint=digest(self.assets)
  self._memo={}

 def lookup(self,fact_packet,cv_prediction,is_intact):
  if type(is_intact) is not bool:raise ValueError('is_intact必须是显式布尔值')
  facts=fact_packet.get('facts')
  if not isinstance(facts,dict):raise ValueError('fact_packet缺少facts映射')
  self.retriever.check_assets()
  domain=known(facts,'domain')
  binding=digest({'fact_packet':fact_packet,'cv_prediction':cv_prediction,'is_intact':is_intact,'assets':self.fingerprint})
  result={'schema':'research-rag-adapter-v1','research_only':True,'input_sha256':binding,
          'asset_fingerprint':self.fingerprint,'rating_decision':None,'numeric_rules':[],
          'source_packets':[],'memo_hit':False,'budget_scope':'每来源独立请求8192 token，预留1024；两包不得直接拼为单个请求'}
  if is_intact or domain not in {'桥梁','公路桥梁','城市桥梁','bridge'}:
   result.update(status='skipped',skip_reason='intact' if is_intact else 'non_bridge_or_unknown_domain')
   result['receipt_sha256']=digest(result);return result
  if not isinstance(cv_prediction,dict) or not isinstance(cv_prediction.get('candidates'),list):
   raise ValueError('cv_prediction必须提供candidates列表')
  predictions=cv_prediction['candidates'][:2]
  if any(not isinstance(v,dict) or not isinstance(v.get('label'),str) or not v['label'].strip() for v in predictions):
   raise ValueError('CV候选必须提供label')
  candidates=[]
  for item in [v['label'] for v in predictions]:
   if isinstance(item,str) and item.strip() and item not in candidates:candidates.append(item)
  if not candidates:
   result.update(status='skipped',skip_reason='missing_defect_candidate')
   result['receipt_sha256']=digest(result);return result
  base={'domain':domain,'component_candidate':known(facts,'component'),'defect_candidates':candidates[:2],
        'target_level':'局部指标','facts':copy.deepcopy(facts)}
  key=digest({'request':base,'assets':self.fingerprint})
  if key in self._memo:
   # 使用现有恢复校验重新核对来源和请求，绝不执行历史预测读取。
   for request,packet in self._memo[key]:
    verified=self.retriever.retrieve(request,max_input_tokens=8192,reserved_output_tokens=1024,limit=1,resume=packet)
    result['source_packets'].append(copy.deepcopy(verified))
   result['memo_hit']=True
  else:
   entries=[]
   for source_id,standard in SOURCES:
    request=dict(base,source_id=source_id,standard=standard)
    packet=self.retriever.retrieve(request,max_input_tokens=8192,reserved_output_tokens=1024,limit=1)
    if packet['rating_decision'] is not None or packet['numeric_rules'] or packet['input_tokens']+1024>8192:
     raise ValueError('RAG返回越权等级或超预算')
    if any(e['source_id']!=source_id for e in packet['evidence']):raise ValueError('RAG串源')
    entries.append((copy.deepcopy(request),copy.deepcopy(packet)))
    result['source_packets'].append(copy.deepcopy(packet))
   self._memo[key]=entries
  result['status']='retrieved_requires_applicability_check'
  result['receipt_sha256']=digest(result)
  return result


def load_research_rag(clock):
 """需在第三方库导入前将 REQUIRED_SYS_PATH 加入 sys.path；模型只计 ONNX run。"""
 for path in reversed(REQUIRED_SYS_PATH):
  if path not in sys.path:sys.path.insert(0,path)
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 # 固定平台已验收源码与实际tokenizer，禁止自动联网下载。
 if sha(RAG_ROOT/'knowledge_retriever.py')!=WRAPPER_SHA256:raise ValueError('已验收RAG封装源码变化')
 if sha(RAG_ROOT/'rag_local.py')!=BACKEND_SHA256:raise ValueError('已验收RAG后端源码变化')
 if {name:sha(TOKENIZER_ROOT/name) for name in TOKENIZER_SHA256}!=TOKENIZER_SHA256:raise ValueError('实际Qwen tokenizer摘要变化')
 import importlib.util
 spec=importlib.util.spec_from_file_location('_research_verified_knowledge',RAG_ROOT/'knowledge_retriever.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(str(TOKENIZER_ROOT),local_files_only=True,trust_remote_code=False)
 manifest=json.loads((RAG_ROOT/'bridge_knowledge_v1/manifest.json').read_text())
 encoder_sha=next(v for k,v in manifest['model_files'].items() if k.endswith('.onnx'))
 retriever=module.KnowledgeRetriever(base_path=RAG_ROOT,index_path=RAG_ROOT/'bridge_knowledge_v1',model_path=RAG_ROOT/'.models',
  manifest_sha256=MANIFEST_SHA256,tokenizer=tokenizer,tokenizer_fingerprint=digest(TOKENIZER_SHA256),
  source_receipt_path=RAG_ROOT/'source_receipt.json',source_receipt_sha256=sha(RAG_ROOT/'source_receipt.json'),
  model_forward_context=lambda:clock.model('rag_encoder',batch_size=1,model_sha256=encoder_sha))
 assets={'adapter_sha256':sha(__file__),'wrapper_sha256':WRAPPER_SHA256,'backend_sha256':BACKEND_SHA256,
         'knowledge_identity':retriever.identity,'tokenizer_files_sha256':TOKENIZER_SHA256,
         'source_receipt_sha256':sha(RAG_ROOT/'source_receipt.json'),'required_sys_path':REQUIRED_SYS_PATH}
 return ResearchRagAdapter(retriever,assets=assets)
