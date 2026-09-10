"""P0/B1旁路入口：记录伙伴证据和实际评级请求，原七字段决策保持。"""
import time
begin_time = time.time()
from pathlib import Path
import json
from report_contract import (ContractError, atomic_json, digest_file, digest_json, locked_diagnosis,
                             make_context, read_json, run_lock)
from run_contract_reports import run_reports
from partner_protocol import packet_ledger, VERSION
from runtime_metrics import RuntimeMetrics


class ObservedGrader:
    def __init__(self, grader, rows, packets, directory, metrics):
        self.grader, self.directory, self.metrics = grader, Path(directory), metrics
        self.lookup = {}
        for row in rows:
            key = digest_json(locked_diagnosis(row, packets[row['sample_id']]))
            self.lookup.setdefault(key, []).append(row['sample_id'])

    def record(self, messages, attempt):
        payload = json.loads(messages[1]['content'])
        cases = self.lookup.get(digest_json(payload['locked_diagnosis']))
        if not cases:
            raise ContractError('评级请求不属于登记案件')
        data = {'case_ids': cases, 'attribution': 'exact' if len(cases) == 1 else 'ambiguous_same_diagnosis',
                'attempt': attempt, 'request_sha256': digest_json(messages),
                'evidence_ids': sorted(payload['evidence_by_id']),
                'evidence_payload_sha256': digest_json(payload['evidence_by_id']),
                '说明': '实际序列化传入评级的证据ID；不等于伙伴账本全部候选均被传入'}
        if len(cases) == 1:
            ledger = read_json(self.directory.parent / 'ledger' / (cases[0] + '.json'))
            data['stable_evidence_ids'] = [o['evidence_id'] for o in ledger['observations']
                                          if o['baseline_alias'] in payload['evidence_by_id']]
            data['summary_nested_evidence_ids'] = [o['evidence_id'] for o in ledger['observations']
                                                   if o['tool'] in payload['evidence_by_id']]
        else:
            data['stable_evidence_ids'] = None
            data['summary_nested_evidence_ids'] = None
        atomic_json(self.directory / (digest_json(data) + '.json'), data)

    def __call__(self, messages, attempt):
        self.record(messages, attempt)
        with self.metrics.span('grading_call', attempts=1):
            return self.grader(messages, attempt)

    def generate_batch(self, requests):
        if not callable(getattr(self.grader, 'generate_batch', None)):
            if len(requests) != 1:
                raise ContractError('生成器不支持真实批量')
            return [self(*requests[0])]
        for messages, attempt in requests:
            self.record(messages, attempt)
        with self.metrics.span('grading_batch', batch_size=len(requests)):
            return self.grader.generate_batch(requests)


def run_partner_reports(rows, packets, output, result_path, producer, generate, *, batch_size=1, cpu_workers=1):
    output = Path(output)
    metrics = RuntimeMetrics()
    source = Path(__file__).parent
    modules = ['run_partner_reports.py', 'partner_protocol.py', 'partner_coordinator.py',
               'runtime_metrics.py', 'input_quality.py', 'extent_evidence.py']
    version = {'base': producer, 'partner_version': VERSION,
               'source_sha256': {name: digest_file(source / name) for name in modules}, 'mode': 'shadow'}
    contexts = [make_context(row, packets[row['sample_id']], version) for row in rows]
    with run_lock(output / 'partner.lock'):
        cfg = output / 'partner_configuration.json'
        configuration = {'contexts': contexts, 'version': version, 'batch_size': batch_size, 'cpu_workers': cpu_workers}
        if cfg.exists() and read_json(cfg) != configuration:
            raise ContractError('伙伴旁路输入或版本变化，请使用新目录')
        atomic_json(cfg, configuration)
        with metrics.span('evidence_ledger', cases=len(rows)):
            for row in rows:
                if digest_file(row['path']) != row.get('image_sha256'):
                    raise ContractError('伙伴输入图片摘要变化')
                ledger = packet_ledger(row, packets[row['sample_id']])
                ledger['input_verification'] = 'file_sha256_checked'
                atomic_json(output / 'ledger' / (row['sample_id'] + '.json'), ledger)
        before = digest_json(packets)
        observer = ObservedGrader(generate, rows, packets, output / 'handoffs', metrics)
        try:
            with metrics.span('baseline_report_stage', cases=len(rows)):
                summary = run_reports(rows, packets, output / 'reports', result_path, version, observer,
                                      batch_size=batch_size, cpu_workers=cpu_workers)
            if digest_json(packets) != before:
                raise ContractError('旁路改变了原始证据')
            atomic_json(output / 'partner_summary.json', {'status': summary['status'], 'mode': 'shadow',
                        'decision_changes': 0, 'baseline_summary': summary,
                        '说明': '当前接通旁路，未启用类型改判；实际收益需独立对照'})
            return summary
        finally:
            metrics.save(output / 'runtime_metrics.json')


def main():
    import argparse
    import os
    from run_contract_reports import FrozenGrader, model_signature
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--packets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=range(1,9), default=3)
    parser.add_argument('--cpu-workers', type=int, choices=[1,2,4,8], default=4)
    args = parser.parse_args()
    for key, name in {'HF_HOME':'hf', 'TORCH_HOME':'torch', 'XDG_CACHE_HOME':'cache', 'TMPDIR':'tmp',
                      'TRITON_CACHE_DIR':'triton', 'CUDA_CACHE_PATH':'cuda', 'ALIPPU_CONFIG_PATH':'ppu',
                      'HGRTC_CACHE_PATH':'ppu/hgrtc'}.items():
        directory = args.output.resolve() / 'runtime' / name
        directory.mkdir(parents=True, exist_ok=True); os.environ[key] = str(directory)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    rows = read_json(args.manifest)
    from report_contract import safe_sample_id, preserve_attempt
    packets = {safe_sample_id(r['sample_id']): read_json(args.packets / (safe_sample_id(r['sample_id']) + '.json')) for r in rows}
    parent = Path(__file__).parent
    producer = {'model': model_signature('/model/Qwen3.6-27B'), 'attention_backend':'sdpa',
                'base_sources': {n:digest_file(parent/n) for n in ['report_contract.py','metadata_semantics.py',
                                                                'run_contract_reports.py','parallel_reports.py','qwen_tool_router.py']}}
    summary = run_partner_reports(rows, packets, args.output, args.result, producer, FrozenGrader('sdpa'),
                                  batch_size=args.batch_size, cpu_workers=args.cpu_workers)
    end_time = time.time()
    infer_time = end_time - begin_time
    timing = args.output / 'infer_time.json'
    preserve_attempt(timing, args.output / 'timing_history')
    atomic_json(timing, {'infer_time': infer_time, 'result_published': summary['status']=='complete',
                        'scope':'报告阶段入口含读包、模型加载和评级；复用专家缓存，不是全架构新推理计时'})
    print(json.dumps({'状态':summary['status'], '模式':'伙伴旁路', 'infer_time':infer_time}, ensure_ascii=False))
    return 0 if summary['status']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
