"""固定类型，紧凑描述与评级；缓存去重，单模型保留供后续审查。"""
import contextlib,time,collections,json
from common import *
from type_prompts import classify,SYSTEM as TYPE_SYSTEM
from speed_prompts import SYSTEM,shorten,describe,rate
from retrieval import Corpus

def valid(obj,stage,mapping,tokens,limit):
 if not isinstance(obj,dict) or tokens>=limit:return False
 refs=obj.get('references')
 if not isinstance(refs,list) or any(not isinstance(x,str) or x not in mapping for x in refs):return False
 if stage=='description':return isinstance(obj.get('defectDescription'),str) and bool(obj['defectDescription'].strip())
 return obj.get('ratingScale(1-5)') in ['','1','2','3','4','5'] and isinstance(obj.get('basis'),str)

class Runner:
 def __init__(self):
  import torch
  from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
  from peft import PeftModel
  torch.set_num_threads(4);self.torch=torch;self.cfg=read(ROOT/'configuration.json');self.corpus=Corpus()
  tick=time.time();self.proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
  base=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
  self.model=PeftModel.from_pretrained(base,OLD/'epoch-2',is_trainable=False).eval();self.model.requires_grad_(False)
  dump(ROOT/'model_loaded.json',{'加载秒':time.time()-tick,'模型':str(MODEL),'适配器摘要':self.cfg['v6_adapter'],'用途':'本轮线上三阶段推理；保持原Peft包装并禁用适配器'})
  self.fresh=[];self.prepared={};self.image_cache={}
 def generate(self,row,version,stage,prompt,limit,image):
  from PIL import Image,ImageOps
  torch=self.torch
  identity={'image':row['sample_id'] if image else None,'model':self.cfg['v6_adapter'] if version=='v6' else 'base','prompt':prompt,'system':TYPE_SYSTEM if stage=='type' else SYSTEM,'limit':limit,'source':self.cfg['sources']}
  key=digest(identity);path=ROOT/'calls'/(key+'.json')
  if path.exists():return read(path),False
  content=[{'type':'text','text':prompt}];ims=None;tick=time.time()
  if image:
   if row['sample_id'] not in self.image_cache:
    with Image.open(row['path']) as im:self.image_cache={row['sample_id']:ImageOps.exif_transpose(im).convert('RGB')}
   ims=[self.image_cache[row['sample_id']]];content.insert(0,{'type':'image'})
  text=self.proc.apply_chat_template([{'role':'system','content':TYPE_SYSTEM if stage=='type' else SYSTEM},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  opts={'images_kwargs':{'size':{'shortest_edge':4096,'longest_edge':262144}}} if image else {}
  enc=self.proc(text=[text],images=ims,return_tensors='pt',**opts);cpu_seconds=time.time()-tick
  assert enc.input_ids.shape[1]<=10000
  tick=time.time();enc=enc.to('cuda');transfer_seconds=time.time()-tick
  tick=time.time()
  with (contextlib.nullcontext() if version=='v6' else self.model.disable_adapter()),torch.inference_mode():
   y=self.model.generate(**enc,max_new_tokens=limit,do_sample=False,use_cache=True,temperature=None,top_p=None,top_k=None)
  torch.cuda.synchronize();out=self.proc.tokenizer.decode(y[0,enc.input_ids.shape[1]:],skip_special_tokens=True);obj=parse(out);seconds=time.time()-tick
  value={'call_id':key,'input':identity,'text':out,'parsed':obj,'output_tokens':int(y.shape[1]-enc.input_ids.shape[1]),'seconds':seconds,'cpu_prepare_seconds':cpu_seconds,'input_transfer_seconds':transfer_seconds,'stage':stage}
  dump(path,value);self.fresh.append(key);return value,True
 def checked(self,row,version,stage,prompt,mapping,image):
  calls=[];fresh=[]
  for limit in [192,768]:
   call,new=self.generate(row,version,stage,prompt,limit,image);calls.append(call['call_id'])
   if new:fresh.append(call['call_id'])
   if valid(call['parsed'],stage,mapping,call['output_tokens'],limit):return call,calls,fresh,True
  return call,calls,fresh,False
 def one(self,row,version):
  p=ROOT/'results'/version/(row['sample_id']+'.json')
  if p.exists():return read(p)
  tick=time.time();first=row['v2_cv'][0]['native_label'] if row['v2_cv'] else row['teacher'][0]['label']
  type_context=self.corpus.get(row,first);tp=classify(row,'v2',type_context)
  v1,new=self.generate(row,'v2','type',tp,96,True);ta=[v1['call_id']];tf=[v1['call_id']] if new else []
  if not isinstance(v1['parsed'],dict) or v1['output_tokens']>=96:
   v1,new=self.generate(row,'v2','type',tp,768,True);ta.append(v1['call_id'])
   if new:tf.append(v1['call_id'])
  typ=(v1['parsed'] or {}).get('defectType','') if isinstance(v1['parsed'],dict) else ''
  typ=typ.strip() if isinstance(typ,str) else ''
  if not typ:raise ValueError('类型无有效字符串，保留调用停止')
  context=self.corpus.get(row,typ);short,mapping=shorten(context)
  description,da,df,dok=self.checked(row,version,'description',describe(row,typ,short),mapping,True)
  obj=description['parsed'] if isinstance(description['parsed'],dict) else {};desc=obj.get('defectDescription','');desc=desc if isinstance(desc,str) else ''
  rt=time.time();rating_context=self.corpus.get(row,typ,desc);retrieval_seconds=time.time()-rt
  short_rating,rating_mapping=shorten(rating_context)
  if typ=='完好':
   rating={'parsed':{'ratingScale(1-5)':'','basis':'预测完好，确定性空','references':[]},'call_id':None};ra=[];rf=[];rok=True
  else:rating,ra,rf,rok=self.checked(row,version,'rating',rate(row,typ,desc,short_rating),rating_mapping,False)
  robj=rating['parsed'] if isinstance(rating['parsed'],dict) else {};grade=robj.get('ratingScale(1-5)','');grade=grade if grade in ['','1','2','3','4','5'] else ''
  def provenance(o,m):
   refs=o.get('references',[]) if isinstance(o,dict) else []
   return {'提供的来源映射':m,'模型声明使用短编号':refs,'可映射的声明来源':[m[x] for x in refs if isinstance(x,str) and x in m] if isinstance(refs,list) else [],'语义适用性':'未由编号有效性证明'}
  value={'final':assemble({'defectType':typ,'defectDescription':desc,'ratingScale(1-5)':grade},row['metadata']),'steps':{'type':v1['call_id'],'description':description['call_id'],'rating':rating['call_id']},'generation_attempts':{'type':ta,'description':da,'rating':ra},'new_calls':tf+df+rf,'contexts':{'type':type_context,'description':context,'rating':rating_context},'provenance':{'description':provenance(obj,mapping),'rating':provenance(robj,rating_mapping)},'description_audit':obj,'rating_basis':robj,'issues':([] if dok else ['描述协议失败'])+([] if rok else ['评级协议失败']),'source_configuration_sha256':sha(ROOT/'configuration.json'),'seconds':time.time()-tick,'rating_retrieval_seconds':retrieval_seconds}
  dump(p,value);return value
 def run(self):
  rows=read(ROOT/'inputs.json');started=time.time()
  for i,row in enumerate(rows):
   self.one(row,'v2');dump(ROOT/'progress.json',{'阶段':'线上三阶段新推理','完成':i+1,'总数':len(rows),'秒':time.time()-started})
  dump(ROOT/'generation_complete.json',{'完成':len(rows),'生成墙钟秒':time.time()-started,'新调用数':len(self.fresh),'类型新推理':True})
