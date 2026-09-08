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

    def test_publish_preserves_old_pair_and_converts_real_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work,output,old=self.fixture(root)
            with patch.object(entry,'WORKSPACE_BOUNDARY',root),patch.object(entry,'WORK_BOUNDARY',root/'work'):
                result=entry.publish(work,output,1.2345)
            self.assertEqual(json.loads((output/'infer_time.json').read_text()),{'infer_time':1234})
            self.assertEqual(result['result_sha256'],entry.digest(output/'result.json'))
            backups=list((work/'previous_published_results').iterdir());self.assertEqual(len(backups),1)
            self.assertEqual((backups[0]/'result.json').read_bytes(),old)
            self.assertEqual(entry.read(backups[0]/'infer_time.json'),{'infer_time':456})

    def test_incomplete_run_cannot_replace_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();work,output,old=self.fixture(root,'incomplete')
            with patch.object(entry,'WORKSPACE_BOUNDARY',root),patch.object(entry,'WORK_BOUNDARY',root/'work'):
                with self.assertRaisesRegex(ValueError,'不发布'):entry.publish(work,output,1.0)
            self.assertEqual((output/'result.json').read_bytes(),old)
            self.assertFalse((work/'previous_published_results').exists())

if __name__=='__main__':unittest.main()
