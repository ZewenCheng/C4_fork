"""平台CPU核验合法外置训练来源映射与篡改拒绝。"""
import json
from pathlib import Path
import tempfile
import unittest

from partner_semantic_tool import checked_training_source
from report_contract import ContractError, digest_file


class SemanticSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.runtime = self.base / 'runtime'; self.runtime.mkdir()
        self.external = self.base / 'original' / 'wemm_features'
        (self.external / 'adamw').mkdir(parents=True)
        self.source = self.external / 'adamw' / 'source.json'
        self.source.write_text('{"原始训练来源":"固定"}', encoding='utf-8')
        self.expected = digest_file(self.source)
        (self.runtime / 'wemm_features').symlink_to(self.external, target_is_directory=True)
        self.receipt = {'protocol': 'c4-submission-contract-v1', 'work': str(self.runtime),
                        'assets': {'wemm_features': str(self.external)}}
        self.write_receipt()

    def write_receipt(self):
        (self.runtime / 'submission_entry_receipt.json').write_text(json.dumps(self.receipt))

    def check(self):
        return checked_training_source(self.runtime, 'wemm_features/adamw', self.expected)

    def test_accepts_exact_bound_external_source_and_plain_source(self):
        self.assertEqual(self.check(), self.source.resolve())
        self.assertEqual(checked_training_source(self.external.parent, 'wemm_features/adamw',
                                                self.expected), self.source.resolve())

    def test_rejects_missing_or_changed_mapping(self):
        (self.runtime / 'submission_entry_receipt.json').unlink()
        with self.assertRaises(ContractError): self.check()
        self.receipt['assets']['wemm_features'] = str(self.base / 'wrong')
        self.write_receipt()
        with self.assertRaises(ContractError): self.check()

    def test_rejects_modified_source_even_with_valid_mapping(self):
        self.source.write_text('{"原始训练来源":"改变"}', encoding='utf-8')
        with self.assertRaises(ContractError): self.check()

    def test_rejects_escape_via_nested_symlink_or_relative_path(self):
        elsewhere = self.base / 'elsewhere'; elsewhere.mkdir()
        (elsewhere / 'source.json').write_bytes(self.source.read_bytes())
        (self.external / 'sgd_momentum').symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(ContractError):
            checked_training_source(self.runtime, 'wemm_features/sgd_momentum', self.expected)
        for rel in ['../original/wemm_features/adamw', str(self.external / 'adamw')]:
            with self.assertRaises(ContractError): checked_training_source(self.runtime, rel, self.expected)


if __name__ == '__main__':
    unittest.main()
