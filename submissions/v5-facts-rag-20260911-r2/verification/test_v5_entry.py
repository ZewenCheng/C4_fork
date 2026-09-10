"""v5独立扫描、来源防篡改及七字段输出边界。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from v5_entry import scan_inputs, verify_code, assemble

class V5Tests(unittest.TestCase):
    def test_scan_distinguishes_same_content_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['桥梁/a.jpg','轨道/a.jpg']:
                p=root/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'fixture')
            rows=scan_inputs(root)
            self.assertEqual(len(rows),2)
            self.assertEqual(len({r['sample_id'] for r in rows}),2)
            self.assertEqual(len({r['image_sha256'] for r in rows}),1)
    def test_scan_rejects_missing_or_ambiguous_domain(self):
        for name in ['a.jpg','桥梁/轨道/a.jpg']:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
                with self.assertRaises(ValueError):scan_inputs(root)
    def test_manifest_rejects_changed_or_extra_code(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'run';p.write_bytes(b'original')
            (root/'source_manifest.json').write_text(json.dumps({'run':hashlib.sha256(b'original').hexdigest()}))
            verify_code(root)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_code(root)
            p.write_bytes(b'original');(root/'unbound.py').write_bytes(b'extra')
            with self.assertRaises(ValueError):verify_code(root)
    def test_output_enforces_conditional_grade(self):
        from report_contract import locked_diagnosis, ContractError
        import test_report_contract
        h=test_report_contract.ProtocolTests();h.setUp()
        row,packet=h.sample('v5','完好');diag=locked_diagnosis(row,packet)
        self.assertEqual(assemble(diag,'',{'text':'当前分类预测为完好。'})['ratingScale(1-5)'],'')
        with self.assertRaises(ContractError):assemble(diag,'2',{'text':'当前分类预测为完好。'})

if __name__=='__main__':unittest.main()
