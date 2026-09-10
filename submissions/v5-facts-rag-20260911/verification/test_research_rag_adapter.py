"""研究RAG有界旁路风险测试：仅合成检索器，不运行模型。"""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_rag_adapter import ResearchRagAdapter,digest

class Fake:
 def __init__(self):self.calls=[];self.changed=False
 def check_assets(self):
  if self.changed:raise ValueError('来源摘要变化')
 def retrieve(self,request,*,max_input_tokens,reserved_output_tokens,limit,resume=None):
  self.check_assets()
  assert (max_input_tokens,reserved_output_tokens,limit)==(8192,1024,1)
  if resume is not None:
   assert resume['fingerprint']==digest(request)
   return resume
  self.calls.append(copy.deepcopy(request))
  return {'fingerprint':digest(request),'rating_decision':None,'numeric_rules':[],
          'input_tokens':10,'evidence':[{'source_id':request['source_id']} ]}

class AdapterTests(unittest.TestCase):
 def setUp(self):
  self.fake=Fake();self.adapter=ResearchRagAdapter(self.fake,assets={'test':'fixed'})
  self.packet={'facts':{'domain':{'state':'estimated','value':'桥梁'},'component':{'state':'unknown','value':None}}}
  self.prediction={'checkpoint_sha256':'fixed','candidates':[{'label':'钢筋锈蚀'},{'label':'表面污渍'},{'label':'不得进入'}]}
 def test_two_sources_top_two_and_run_only_memo(self):
  first=self.adapter.lookup(self.packet,self.prediction,False)
  self.assertEqual(len(self.fake.calls),2)
  self.assertEqual(self.fake.calls[0]['defect_candidates'],['钢筋锈蚀','表面污渍'])
  self.assertNotEqual(self.fake.calls[0]['source_id'],self.fake.calls[1]['source_id'])
  again=self.adapter.lookup(self.packet,self.prediction,False)
  self.assertTrue(again['memo_hit']);self.assertEqual(len(self.fake.calls),2)
  self.assertIsNone(first['rating_decision'])
  self.fake.changed=True
  with self.assertRaisesRegex(ValueError,'来源摘要变化'):self.adapter.lookup(self.packet,self.prediction,False)
 def test_intact_and_rail_skip(self):
  self.assertEqual(self.adapter.lookup(self.packet,self.prediction,True)['skip_reason'],'intact')
  self.packet['facts']['domain']['value']='轨道'
  self.assertEqual(self.adapter.lookup(self.packet,self.prediction,False)['status'],'skipped')
  self.assertEqual(self.fake.calls,[])
 def test_changed_request_not_reused(self):
  self.adapter.lookup(self.packet,self.prediction,False)
  self.packet['facts']['component']={'state':'estimated','value':'桥墩'}
  result=self.adapter.lookup(self.packet,self.prediction,False)
  self.assertFalse(result['memo_hit']);self.assertEqual(len(self.fake.calls),4)

if __name__=='__main__':unittest.main()
