"""用真实短子进程验证完成事件和失败清理，防止固定等待及孤儿写入。"""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
if os.name=='nt':
    with patch.dict(sys.modules,{'fcntl':types.ModuleType('fcntl')}):
        pipeline=importlib.import_module('online_full_pipeline')
else:
    pipeline=importlib.import_module('online_full_pipeline')


class PipelineConcurrencyTests(unittest.TestCase):
    def test_parallel_children_finish_without_poll_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            scripts=[]
            for index in range(2):
                p=root/f'child_{index}.py'
                p.write_text('import time\nfrom pathlib import Path\ntime.sleep(.2)\nPath(__file__).with_suffix(".done").write_text("完成",encoding="utf-8")\n',encoding='utf-8')
                scripts.append(p)
            started=time.monotonic()
            with patch.object(pipeline,'ROOT',root),patch.object(pipeline,'ONLINE',root):pipeline.run_group(scripts)
            self.assertLess(time.monotonic()-started,5)
            self.assertTrue(all(p.with_suffix('.done').is_file() for p in scripts))

    def test_failure_stops_only_owned_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            bad=root/'fail.py';bad.write_text('import time\ntime.sleep(.2)\nraise SystemExit(7)\n',encoding='utf-8')
            slow=root/'slow.py';slow.write_text('import time\ntime.sleep(30)\n',encoding='utf-8')
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            children=[];original=subprocess.Popen
            def record(*args,**kwargs):
                child=original(*args,**kwargs);children.append(child);return child
            try:
                with patch.object(pipeline,'ROOT',root),patch.object(pipeline,'ONLINE',root),patch.object(pipeline.subprocess,'Popen',side_effect=record):
                    with self.assertRaisesRegex(RuntimeError,'退出码7'):pipeline.run_group([bad,slow])
                self.assertTrue(all(child.poll() is not None for child in children))
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate();unrelated.wait(timeout=10)

if __name__=='__main__':unittest.main()
