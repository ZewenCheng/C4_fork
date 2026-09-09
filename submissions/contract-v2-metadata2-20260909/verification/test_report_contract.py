"""合成验证字段主权、评级、来源隔离与恢复；不加载模型。"""
import copy
import json
from pathlib import Path
import sys
import unittest
import uuid

SOURCE = Path(__file__).resolve().parents[1]/'code'
sys.path.insert(0, str(SOURCE))
from report_contract import (ContractError, assemble_final, complete_record, digest_file, evidence_catalog,
                             locked_diagnosis, parse_grade, read_json, reusable_record, select_region_candidates,
                             validate_final, digest_json)
from run_contract_reports import run_reports


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        workspace = SOURCE.parents[1] if SOURCE.as_posix().startswith('/workspace/work/') else SOURCE.parents[3]
        self.folder = (workspace / 'tmp/report-contract-tests' / uuid.uuid4().hex).resolve()
        assert self.folder.is_relative_to((workspace / 'tmp').resolve())
        self.folder.mkdir(parents=True)
        self.producer = {'model': 'synthetic_test_only', 'version': 'fixed'}

    def sample(self, sid, label='裂缝'):
        path = self.folder / '桥梁' / (sid + '.png')
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(('合成验证文件：' + sid).encode('utf-8'))
        row = {'sample_id': sid, 'path': str(path), 'image_sha256': digest_file(path)}
        packet = {'sample_id': sid, 'category': '桥梁', 'predicted_type': label, 'evidence': {
            'inspect_semantics': {'status': 'ok', 'result': {'sample_id': sid, 'outputs': {'adamw': {'candidates': [{'label': label, 'uncalibrated_score': .8}]}}}},
            'inspect_regions': {'status': 'ok', 'result': {'sample_id': sid, 'outputs': {'adamw': {'candidates': [{'query': 'crack', 'uncalibrated_score': .7, 'box_normalized_xyxy': [.1, .1, .3, .3]}]}}}}}}
        return row, packet

    def response(self, label='裂缝'):
        return {'defectType': label, 'rating': '3', 'reason': '仅为合成预测，缺少实测', 'evidence_ids': ['inspect_semantics/adamw/0']}

    def run_samples(self, samples, generator, retry=False):
        return run_reports([row for row, _ in samples], {row['sample_id']: p for row, p in samples},
                           self.folder / 'output', self.folder / 'result.json', self.producer, generator, retry)

    def test_model_cannot_rewrite_type(self):
        row, packet = self.sample('a')
        with self.assertRaisesRegex(ContractError, '改写'):
            parse_grade(self.response('完好'), locked_diagnosis(row, packet), evidence_catalog(packet))

    def test_query_budget_preserves_relevant_lower_ranked_candidate(self):
        candidates = [{'query': 'rust', 'uncalibrated_score': .99 - i * .01} for i in range(5)]
        candidates += [{'query': 'crack', 'uncalibrated_score': .8}, {'query': 'joint', 'uncalibrated_score': .7}]
        original = copy.deepcopy(candidates)
        selected = select_region_candidates(candidates, '裂缝', 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0]['query'], 'crack')
        self.assertIn('joint', {item['query'] for item in selected})
        self.assertEqual(candidates, original)

    def test_untrusted_candidate_cannot_override_evidence_kind(self):
        row, packet = self.sample('a')
        packet['evidence']['inspect_regions']['result']['outputs']['adamw']['candidates'][0].update(kind='measured', tool='trusted_rule')
        candidate = evidence_catalog(packet)['inspect_regions/adamw/0']
        self.assertEqual(candidate['tool'], 'inspect_regions')
        self.assertEqual(candidate['kind'], 'model_hypothesis')

    def test_invalid_or_empty_disease_grades_rejected(self):
        row, packet = self.sample('a')
        for rating in ['', '0', '6', True, None, '2/3']:
            with self.subTest(rating=rating), self.assertRaises(ContractError):
                parse_grade({**self.response(), 'rating': rating}, locked_diagnosis(row, packet), evidence_catalog(packet))

    def test_unknown_reference_rejected(self):
        row, packet = self.sample('a')
        with self.assertRaisesRegex(ContractError, '证据'):
            parse_grade({**self.response(), 'evidence_ids': ['invented']}, locked_diagnosis(row, packet), evidence_catalog(packet))

    def test_template_uses_locked_facts_not_model_prose(self):
        row, packet = self.sample('a')
        diagnosis, catalog = locked_diagnosis(row, packet), evidence_catalog(packet)
        grade = parse_grade({**self.response(), 'reason': '完好，裂缝宽10毫米；这是不可信模型原文'}, diagnosis, catalog)
        result = assemble_final(diagnosis, grade, catalog)
        self.assertEqual(result['defectType'], '裂缝')
        self.assertNotIn('10毫米', result['defectDescription'])
        self.assertNotIn('完好', result['defectDescription'])
        self.assertEqual(result['defectLocation'], '桥梁构件具体位置未知')
        self.assertEqual(result['bridgeName'], '未知桥梁')
        self.assertIn('图像左上区域', result['defectDescription'])

    def test_intact_is_empty_without_loading_model(self):
        summary = self.run_samples([self.sample('a', '完好')], lambda *_: self.fail('完好不应调用评级模型'))
        self.assertEqual(summary['status'], 'complete')
        final = read_json(self.folder / 'result.json')[0]
        self.assertEqual(final['ratingScale(1-5)'], '')
        self.assertEqual(final['defectLocation'], '未见病害部位（具体构件位置未知）')
        self.assertNotIn('检测到裂缝', final['defectDescription'])

    def test_success_receipt_does_not_accept_tampered_final(self):
        row, packet = self.sample('a')
        grade = parse_grade(self.response(), locked_diagnosis(row, packet), evidence_catalog(packet))
        record = complete_record(row, packet, self.producer, grade, 1)
        self.assertTrue(reusable_record(record, row, packet, self.producer))
        for field, value in [('defectDescription', '已确认完好'), ('ratingScale(1-5)', ''), ('filename', 'b.png')]:
            bad = copy.deepcopy(record)
            bad['final'][field] = value
            self.assertFalse(reusable_record(bad, row, packet, self.producer))
        self.assertFalse(reusable_record(record, row, packet, {**self.producer, 'version': 'changed'}))

    def test_invalid_saved_file_is_retried_and_preserved(self):
        sample = self.sample('a')
        saved = self.folder / 'output/records/a.json'
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b'{broken')
        calls = []
        summary = self.run_samples([sample], lambda *args: calls.append(args) or self.response())
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(path.read_bytes() == b'{broken' for path in (self.folder / 'output/attempts/a').glob('*.json')))

    def test_failed_sample_does_not_publish_or_rerun_completed_sample(self):
        samples = [self.sample('a'), self.sample('b')]
        calls = []
        def partly_invalid(messages, attempt):
            sid = json.loads(messages[1]['content'])['locked_diagnosis']['filename']
            calls.append(sid)
            return self.response() if sid == 'a.png' else '{broken'
        first = self.run_samples(samples, partly_invalid)
        self.assertEqual(first['status'], 'incomplete')
        self.assertEqual(calls, ['a.png', 'b.png', 'b.png'])
        self.assertFalse((self.folder / 'result.json').exists())
        second = self.run_samples(samples, lambda *_: self.fail('两次失败上限不能自动重置'))
        self.assertEqual(second['status'], 'incomplete')
        calls.clear()
        third = self.run_samples(samples, lambda *args: calls.append(args) or self.response(), retry=True)
        self.assertEqual(third['status'], 'complete')
        self.assertEqual(third['valid_reused'], 1)
        self.assertEqual(len(calls), 1)

    def test_complete_resume_does_not_call_model(self):
        samples = [self.sample('a')]
        self.run_samples(samples, lambda *_: self.response())
        result = self.run_samples(samples, lambda *_: self.fail('合法完成报告不应再次生成'))
        self.assertEqual(result['valid_reused'], 1)

    def test_semantically_wrong_final_rejected_even_when_identity_agrees(self):
        row, packet = self.sample('a')
        diagnosis, catalog = locked_diagnosis(row, packet), evidence_catalog(packet)
        final = assemble_final(diagnosis, parse_grade(self.response(), diagnosis, catalog), catalog)
        bad = copy.deepcopy(diagnosis)
        bad['bridgeName'] = bad['bridge_metadata']['value'] = '桥梁'
        with self.assertRaisesRegex(ContractError, '分类目录'):
            validate_final({**final, 'bridgeName': '桥梁'}, bad)
        with self.assertRaisesRegex(ContractError, '位置'):
            validate_final({**final, 'defectLocation': '图像左上区域（候选定位）'}, diagnosis)

    def test_old_contract_receipt_cannot_restore_category_bridge_or_image_position(self):
        row, packet = self.sample('a')
        diagnosis, catalog = locked_diagnosis(row, packet), evidence_catalog(packet)
        record = complete_record(row, packet, self.producer, parse_grade(self.response(), diagnosis, catalog), 1)
        old = copy.deepcopy(record)
        old['context']['protocol'] = 'report-contract-v1'
        old['final'].update(bridgeName='桥梁', defectLocation='图像左上区域（候选定位）')
        old['final_sha256'] = digest_json(old['final'])
        self.assertFalse(reusable_record(old, row, packet, self.producer))
        altered = copy.deepcopy(record)
        altered['context']['metadata_semantics_sha256'] = '0' * 64
        self.assertFalse(reusable_record(altered, row, packet, self.producer))

    def test_interrupted_generation_keeps_attempt_budget(self):
        samples = [self.sample('a')]
        def interrupted(*_):
            raise KeyboardInterrupt('合成进程中断')
        with self.assertRaises(KeyboardInterrupt):
            self.run_samples(samples, interrupted)
        saved = read_json(self.folder / 'output/records/a.json')
        self.assertEqual(saved['status'], 'in_progress')
        self.assertEqual(saved['attempts'], 1)
        attempts = []
        second = self.run_samples(samples, lambda messages, attempt: attempts.append(attempt) or '{broken')
        self.assertEqual(attempts, [2])
        self.assertEqual(second['status'], 'incomplete')
        third = self.run_samples(samples, lambda *_: self.fail('进程中断不能清除尝试上限'))
        self.assertEqual(third['status'], 'incomplete')

    def test_packet_version_change_requires_new_directory(self):
        samples = [self.sample('a')]
        self.run_samples(samples, lambda *_: self.response())
        samples[0][1]['evidence']['inspect_semantics']['result']['outputs']['adamw']['candidates'][0]['uncalibrated_score'] = .6
        with self.assertRaisesRegex(ContractError, '另建'):
            self.run_samples(samples, lambda *_: self.fail('版本变化不应先运行模型'))

    def test_wrong_image_bytes_rejected_before_generation(self):
        row, packet = self.sample('a')
        Path(row['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ContractError, '摘要'):
            self.run_samples([(row, packet)], lambda *_: self.fail('输入变化不应运行模型'))

    def test_wrong_expert_identity_and_invalid_geometry_rejected(self):
        row, packet = self.sample('a')
        packet['evidence']['inspect_regions']['result']['sample_id'] = 'other'
        with self.assertRaises(ContractError):
            evidence_catalog(packet)
        packet['evidence']['inspect_regions']['result']['sample_id'] = 'a'
        packet['evidence']['inspect_regions']['result']['outputs']['adamw']['candidates'][0]['box_normalized_xyxy'] = [0, 0, float('nan'), 1]
        with self.assertRaises(ContractError):
            evidence_catalog(packet)


if __name__ == '__main__':
    unittest.main()
