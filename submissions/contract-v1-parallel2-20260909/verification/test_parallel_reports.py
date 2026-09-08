"""批量身份、失败隔离、中断预算及显存拆批的必要行为回归。"""
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from test_report_contract import ProtocolTests
from run_contract_reports import FrozenGrader, run_reports
from report_contract import read_json


class ParallelProtocolTests(ProtocolTests):
    def run_samples(self, samples, generator, retry=False):
        class Batch:
            def generate_batch(self, requests):
                return [generator(*request) for request in requests]
        return run_reports([row for row, _ in samples], {row['sample_id']: p for row, p in samples},
                           self.folder / 'output', self.folder / 'result.json', self.producer, Batch(), retry,
                           batch_size=2, cpu_workers=2)

    def test_batches_preserve_input_order_and_identity(self):
        samples=[self.sample('b'),self.sample('a'),self.sample('z','完好')]
        calls=[]
        class Batch:
            def generate_batch(inner, requests):
                calls.append(len(requests))
                responses=[]
                for messages,attempt in requests:
                    filename=json.loads(messages[1]['content'])['locked_diagnosis']['filename']
                    responses.append({**self.response(),'rating':'1' if filename=='a.png' else '4'})
                return responses
        result=run_reports([r for r,_ in samples],{r['sample_id']:p for r,p in samples},self.folder/'output',
                           self.folder/'result.json',self.producer,Batch(),batch_size=2,cpu_workers=2)
        values=read_json(self.folder/'result.json')
        self.assertEqual(calls,[2])
        self.assertEqual([v['filename'] for v in values],['b.png','a.png','z.png'])
        self.assertEqual([v['ratingScale(1-5)'] for v in values],['4','1',''])
        self.assertEqual(result['failed'],0)

    def test_batch_size_mismatch_cannot_publish(self):
        samples=[self.sample('a'),self.sample('b')]
        class Broken:
            def generate_batch(self, requests):return []
        result=run_reports([r for r,_ in samples],{r['sample_id']:p for r,p in samples},self.folder/'output',
                           self.folder/'result.json',self.producer,Broken(),batch_size=2,cpu_workers=2)
        self.assertEqual(result['failed'],2)
        self.assertFalse((self.folder/'result.json').exists())
        self.assertEqual([read_json(self.folder/f'output/records/{sid}.json')['attempts'] for sid in ['a','b']],[2,2])

    def test_interrupted_batch_marks_every_member_before_call(self):
        samples=[self.sample('a'),self.sample('b')]
        class Interrupted:
            def generate_batch(self,requests):raise KeyboardInterrupt()
        args=([r for r,_ in samples],{r['sample_id']:p for r,p in samples},self.folder/'output',self.folder/'result.json',self.producer)
        with self.assertRaises(KeyboardInterrupt):run_reports(*args,Interrupted(),batch_size=2,cpu_workers=2)
        self.assertEqual([read_json(self.folder/f'output/records/{sid}.json')['attempts'] for sid in ['a','b']],[1,1])
        attempts=[]
        class Recovered:
            def generate_batch(inner, requests):
                attempts.extend(a for _,a in requests)
                return [self.response() for _ in requests]
        result=run_reports(*args,Recovered(),batch_size=2,cpu_workers=2)
        self.assertEqual(attempts,[2,2]);self.assertEqual(result['status'],'complete')


class MemoryFallbackTests(unittest.TestCase):
    def test_oom_split_keeps_batch_order_and_caches_smaller_limit(self):
        class OOM(RuntimeError):pass
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None,synchronize=lambda:None,empty_cache=lambda:None,
                             max_memory_allocated=lambda:10,max_memory_reserved=lambda:20)
        torch=SimpleNamespace(cuda=cuda,OutOfMemoryError=OOM)
        grader=FrozenGrader();grader.router=SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=lambda m,**k:m[0]['content']))
        seen=[]
        def generate(prompts,attempt):
            seen.append(list(prompts))
            if len(prompts)>2:raise OOM('out of memory')
            return prompts,20,len(prompts)
        requests=[([{'content':str(i)}],1) for i in range(4)]
        with patch.dict(sys.modules,{'torch':torch}),patch.object(grader,'load'),patch.object(grader,'_generate',side_effect=generate):
            self.assertEqual(grader.generate_batch(requests),['0','1','2','3'])
            self.assertEqual(grader.generate_batch(requests),['0','1','2','3'])
        self.assertEqual(grader.batch_limit,2)
        self.assertEqual([len(x) for x in seen],[4,2,2,2,2])


del ProtocolTests  # 基类在自身测试模块执行，避免discover重复收集。

if __name__=='__main__':unittest.main()
