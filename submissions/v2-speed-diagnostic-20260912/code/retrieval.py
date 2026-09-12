"""训练JSON与规范OCR的可追溯检索，不使用开发标签。"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import *
class Corpus:
 def __init__(self):
  self.rows=read(ASSETS/'train_corpus.json');self.chunks=read(RAG/'chunks.json')
  self.vector=TfidfVectorizer(analyzer='char',ngram_range=(1,3),max_features=50000,sublinear_tf=True)
  self.x=self.vector.fit_transform([r['defectType']+' '+r['defectDescription'] for r in self.rows])
  self.rv=TfidfVectorizer(analyzer='char',ngram_range=(1,3),max_features=60000,sublinear_tf=True)
  self.rx=self.rv.fit_transform([c['text'] for c in self.chunks]);self.by={r['id']:i for i,r in enumerate(self.rows)}
  with np.load(ASSETS/'train_features.npz',allow_pickle=False) as z:
   a=z['x'].astype(np.float32);a=a.mean(1) if a.ndim==3 else a;self.features={s:v/max(float(np.linalg.norm(v)),1e-9) for s,v in zip(z['sample_ids'].tolist(),a)}
  with np.load(ROOT/'teacher_features.npz',allow_pickle=False) as z:
   a=z['x'].astype(np.float32);a=a.mean(1) if a.ndim==3 else a
   self.features.update({s:v/max(float(np.linalg.norm(v)),1e-9) for s,v in zip(z['sample_ids'].tolist(),a)})
 def get(self,row,typ,description=''):
  query=typ+' '+description;score=np.asarray((self.x@self.vector.transform([query]).T).toarray()).ravel()
  scores=[]
  for i,r in enumerate(self.rows):
   if r['category']!=row['category'] or (description and r['ratingScale(1-5)'] is None):continue
   if r['id']==row['sample_id']:raise ValueError('验证图进入训练语料')
   visual=float(self.features[row['sample_id']]@self.features[r['id']]);value=float(score[i])+float(r['defectType']==typ)+(.2*visual if not description else 0)
   scores.append((value,r['id'],i))
  selected=sorted(scores,key=lambda t:(-t[0],t[1]))[:2]
  examples=[{'id':'T:'+self.rows[i]['id'],'类型':self.rows[i]['defectType'],'描述':self.rows[i]['defectDescription'],'评级':self.rows[i]['ratingScale(1-5)'],'用途':'其他训练图标注，仅术语和程度映射参考，不代表本图'} for _,_,i in selected]
  evidence=[]
  if row['category']=='桥梁':
   q=np.asarray((self.rx@self.rv.transform([query+' 病害 评定 标度']).T).toarray()).ravel()
   for sid in sorted({c['source_id'] for c in self.chunks}):
    ix=max((i for i,c in enumerate(self.chunks) if c['source_id']==sid),key=lambda i:float(q[i]));c=self.chunks[ix]
    evidence.append({'id':'R:'+c['id'],'来源':c['filename'],'页码':c['pdf_page'],'角色':c['document_role'],'内容':c['text'],'阈值已核定':c['thresholds_verified'],'注意':'OCR候选；先核对构件及指标适用性，不能把局部构件标度冒充整桥评分'})
  return {'examples':examples,'rag':evidence,'rag_scope':'桥梁两份既有文档；轨道无适用规范，禁止套用桥梁标准','retrieval':'固定字符TF-IDF，描述示例加入冻结CV视觉相似度；不作官方语义评分'}
