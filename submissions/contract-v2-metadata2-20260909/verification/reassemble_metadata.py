"""在官方工作区核验旧凭证后，仅重装配桥名和位置；不加载任何模型。"""
import time
begin_time = time.time()

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

BASELINE_SHA = 'a09293f6fcb9efe043a32ed7feb8924fbcbfc41d03f1dc5be38fecfae3634473'
INPUT_SHA = '37e8e9d51b82ca5718881c2af8ad3b017aca87db6d0ff26e7aa5b222dfac6192'
BASELINE_ENTRY = Path('/workspace/work/c4-parallel2-entry-v1-20260909')
BASELINE_PACKAGE = Path('/workspace/work/c4-submission-versions/contract-v1-parallel2-20260909/submit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    code, output = args.code.resolve(), args.output.resolve()
    boundary = Path('/workspace/work').resolve()
    if not code.is_relative_to(boundary) or not output.is_relative_to(boundary) or output == boundary or output.exists():
        raise ValueError('源码与新输出须位于官方work独立目录，禁止覆盖已有结果')
    sys.path.insert(0, str(code))
    from report_contract import (FIELDS, atomic_json, digest_file, digest_json, read_json, reusable_record)
    from run_contract_reports import run_reports

    source = BASELINE_ENTRY / 'candidate_online_20260908'
    old_config = read_json(source / 'contract_reports/configuration.json')
    original = read_json(source / 'result.json')
    if digest_file(source / 'result.json') != BASELINE_SHA or digest_file(BASELINE_PACKAGE / 'result/result.json') != BASELINE_SHA:
        raise ValueError('原始结果与固定parallel2版本不一致')
    if digest_file(source / 'input_manifest.json') != INPUT_SHA:
        raise ValueError('输入集合改变')
    rows = read_json(source / 'input_manifest.json')
    if len(rows) != len(original) or len(rows) != 330 or len({r['sample_id'] for r in rows}) != 330:
        raise ValueError('输入规模或身份不符')
    legacy_path = BASELINE_PACKAGE / 'code/report_contract.py'
    if digest_file(legacy_path) != old_config['producer']['contract_sha256']:
        raise ValueError('旧协议不是原报告生产版本')
    spec = importlib.util.spec_from_file_location('verified_legacy_contract', legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    source_manifest = read_json(code / 'source_manifest.json')
    for name, expected in source_manifest['files'].items():
        if digest_file(code / name) != expected:
            raise ValueError('修复源码清单不符')

    def inspect(item):
        row, final = item
        sid = row['sample_id']
        packet = read_json(source / 'packets' / (sid + '.json'))
        path = source / 'contract_reports/records' / (sid + '.json')
        record = read_json(path)
        if digest_file(row['path']) != row['image_sha256'] or not legacy.reusable_record(record, row, packet, old_config['producer']) or record['final'] != final:
            raise ValueError('旧输入、证据或完成凭证校验失败')
        if final['filename'] != Path(row['path']).name or final['defectType'] != packet['predicted_type']:
            raise ValueError('旧结果身份或分类不一致')
        return row, packet, record, digest_file(path)

    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = list(pool.map(inspect, zip(rows, original)))
    packets = {row['sample_id']: packet for row, packet, _, _ in inspected}
    old_by_key = {(record['final']['questionCategory'], record['final']['filename']): record for _, _, record, _ in inspected}
    producer = {'mode': 'replay_verified_grade_for_metadata_reassembly',
                'contract_sha256': digest_file(code / 'report_contract.py'),
                'metadata_semantics_sha256': digest_file(code / 'metadata_semantics.py'),
                'runner_sha256': digest_file(code / 'run_contract_reports.py'),
                'parallel_sha256': digest_file(code / 'parallel_reports.py'),
                'reassembly_sha256': digest_file(Path(__file__)),
                'source_result_sha256': BASELINE_SHA, 'source_producer': old_config['producer'],
                'source_records_sha256': digest_json([{'sample_id': row['sample_id'], 'sha256': digest} for row, _, _, digest in inspected])}

    class VerifiedGradeReplay:
        calls = 0
        def generate_batch(self, requests):
            return [self(messages, attempt) for messages, attempt in requests]
        def __call__(self, messages, attempt):
            diagnosis = json.loads(messages[1]['content'])['locked_diagnosis']
            old = old_by_key[(diagnosis['questionCategory'], diagnosis['filename'])]
            if old['final']['defectType'] != diagnosis['defectType']:
                raise ValueError('分类改变时不可复用旧评级')
            self.calls += 1
            grade = old['grade_decision']
            return {'defectType': diagnosis['defectType'], **{key: grade[key] for key in ['rating', 'reason', 'evidence_ids']}}

    replay = VerifiedGradeReplay()
    state = run_reports(rows, packets, output / 'reports', output / 'result.json', producer, replay, batch_size=3, cpu_workers=4)
    if state['status'] != 'complete' or state['valid_generated'] != 330:
        raise ValueError('修复重装配尚未全部完成')
    repaired = read_json(output / 'result.json')
    changes = {key: sum(a[key] != b[key] for a, b in zip(original, repaired)) for key in FIELDS}
    if any(count for key, count in changes.items() if key not in {'bridgeName', 'defectLocation'}):
        raise ValueError('桥名和位置之外的字段发生改变')
    sources, location_sources, certainties = Counter(), Counter(), Counter()
    for row, packet, old, _ in inspected:
        if reusable_record(old, row, packet, producer):
            raise ValueError('旧凭证错误地通过新协议')
        record = read_json(output / 'reports/records' / (row['sample_id'] + '.json'))
        if not reusable_record(record, row, packet, producer):
            raise ValueError('新凭证无法复算')
        locked = record['locked_diagnosis']
        sources[locked['bridge_metadata']['source']] += 1
        location_sources[locked['location_metadata']['source']] += 1
        certainties[locked['location_metadata']['certainty']] += 1
    # 验证实际恢复路径，不向模型或回放边界发出第二次请求。
    def forbidden(*args):
        raise AssertionError('完整恢复不应再次请求评级')
    class NoGeneration:
        __call__ = staticmethod(forbidden)
        generate_batch = staticmethod(forbidden)
    resumed = run_reports(rows, packets, output / 'reports', output / 'result.json', producer, NoGeneration(), batch_size=3, cpu_workers=4)
    if resumed['valid_reused'] != 330 or resumed['valid_generated'] != 0:
        raise ValueError('新结果完整恢复未通过')
    if digest_file(source / 'result.json') != BASELINE_SHA:
        raise ValueError('旧结果发生变化')
    elapsed = time.time() - begin_time
    timing = {'repair_time_seconds': elapsed, 'source_infer_time_seconds': read_json(BASELINE_PACKAGE / 'result/infer_time.json')['infer_time'],
              'new_full_inference_time_seconds': None,
              '说明': '本次仅核验原始输入和旧专家凭证、重放既有评级并重装配两个元数据字段，含恢复核验；没有新的模型推理，不作为完整冷启动耗时。'}
    atomic_json(output / 'repair_timing.json', timing)
    summary = {'状态': '桥名与结构位置重装配及完整恢复通过', '图数': len(rows),
               'result_sha256': digest_file(output / 'result.json'), 'source_result_sha256': BASELINE_SHA,
               'source_manifest_sha256': digest_file(code / 'source_manifest.json'),
               '输入SHA256': INPUT_SHA, '逐字段变化数': changes,
               '桥名来源计数': dict(sources), '位置来源计数': dict(location_sources), '位置确定性计数': dict(certainties),
               '新模型调用次数': 0, '复用旧病害评级数': replay.calls, '新凭证恢复数': resumed['valid_reused'],
               '旧凭证被新协议拒绝数': 330, '计时': timing,
               '原始结果路径': str(output / 'result.json'), '逐图凭证': str(output / 'reports/records'),
               '正式作品目录已替换': False, '比赛提交或评测': False, '正式得分': None}
    atomic_json(output / 'repair_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
