"""提交包装新增行为：来源变化拒绝、旧结果保护、失败时禁止发布。"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import submission_entry as entry


class SubmissionEntryTests(unittest.TestCase):
    def test_resume_rejects_changed_execution_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();code=root/'code';code.mkdir();inp=root/'dataset';inp.mkdir()
            names=['start_online_candidate.py','online_full_pipeline.py','report_contract.py','run_contract_reports.py']
            for name in [*names,'submission_entry.py']:(code/name).write_text('pass\n',encoding='utf-8')
            entry.write(code/'source_manifest.json',{'files':{name:entry.digest(code/name) for name in names}})
            entry.write(code/'model_manifest.json',{'files':[],'qwen_metadata':{'metadata_sha256':{}}})
            work=root/'work/run';execution={'report_batch_size':4,'cpu_workers':4}
            with patch.object(entry,'WORK_BOUNDARY',root/'work'),patch.object(entry,'DATA_BOUNDARY',inp),patch.object(entry,'ASSETS',[]):
                entry.prepare(code,work,inp,execution=execution)
                entry.prepare(code,work,inp,resume=True,check_only=True,execution=execution)
                with self.assertRaisesRegex(ValueError,'禁止混用'):
                    entry.prepare(code,work,inp,resume=True,execution={'report_batch_size':8,'cpu_workers':4})

    def test_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            name='report_contract.py'
            (root/name).write_text('pass\n',encoding='utf-8')
            entry.write(root/'source_manifest.json',{'files':{name:entry.digest(root/name)}})
            (root/name).write_text('raise RuntimeError()\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'摘要'):
                entry.sources_for_work(root,root/'work',root/'input')

    def fixture(self,root,status='complete'):
        work=root/'work/run';online=work/'candidate_online_20260908';online.mkdir(parents=True)
        entry.write(online/'result.json',[{'synthetic':'fixture'}])
        entry.write(online/'contract_reports/status.json',{'status':status,'images':1,'result_sha256':entry.digest(online/'result.json')})
        output=root/'result/result';output.mkdir(parents=True)
        old=b'[{"prior":"preserve"}]\n';(output/'result.json').write_bytes(old)
        entry.write(output/'infer_time.json',{'infer_time':456})
        return work,output,old

    def test_publish_preserves_old_pair_and_keeps_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work,output,old=self.fixture(root)
            with patch.object(entry,'WORKSPACE_BOUNDARY',root),patch.object(entry,'WORK_BOUNDARY',root/'work'):
                result=entry.publish(work,output,1.2345)
            self.assertEqual(json.loads((output/'infer_time.json').read_text()),{'infer_time':1.2345})
            self.assertEqual(result['result_sha256'],entry.digest(output/'result.json'))
            backups=list((work/'previous_published_results').iterdir());self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'result.json').read_bytes(),old)
            self.assertEqual(entry.read(backups[0]/'infer_time.json'),{'infer_time':456})

    def test_end_clock_includes_result_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work,output,old=self.fixture(root)
            def end_clock():
                self.assertTrue((output/'result.json.pending').is_file())
                self.assertTrue((work/'previous_published_results').is_dir())
                return 114.125
            with patch.object(entry,'WORKSPACE_BOUNDARY',root),patch.object(entry,'WORK_BOUNDARY',root/'work'),patch.object(entry.time,'time',side_effect=end_clock):
                result=entry.publish(work,output,started_at=100.0)
            self.assertEqual(result['infer_time'],14.125)
            self.assertEqual(entry.read(output/'infer_time.json'),{'infer_time':14.125})

    def test_invalid_time_cannot_replace_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work,output,old=self.fixture(root)
            for value in [float('nan'),float('inf'),-1,0,True]:
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'计时无效'):
                    entry.publish(work,output,value)
            self.assertEqual((output/'result.json').read_bytes(),old)

    def test_incomplete_run_cannot_replace_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work,output,old=self.fixture(root,'incomplete')
            with patch.object(entry,'WORKSPACE_BOUNDARY',root),patch.object(entry,'WORK_BOUNDARY',root/'work'):
                with self.assertRaisesRegex(ValueError,'不发布'):entry.publish(work,output,1.0)
            self.assertEqual((output/'result.json').read_bytes(),old)
            self.assertFalse((work/'previous_published_results').exists())

if __name__=='__main__':unittest.main()
