"""统一报告执行：先锁定事实、按需预测评级、确定性装配、校验完成凭证及限次恢复。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from report_contract import (VERSION, ContractError, assemble_final, atomic_json, complete_record, digest_file,
                             digest_json, evidence_catalog, grade_messages, locked_diagnosis, make_context,
                             parse_grade, preserve_attempt, read_json, reusable_record, run_lock, safe_sample_id,
                             validate_prediction_source)


def run_reports(rows, packets, output, result_path, producer, generate, retry_failed=False):
    """generate(messages, attempt)是唯一模型边界，测试可替换，生产固定使用冻结Qwen。"""
    output, result_path = Path(output), Path(result_path)
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ContractError('输入为空或样本ID重复')
    prepared = []
    for row in rows:
        sid = safe_sample_id(row['sample_id'])
        if digest_file(row['path']) != row.get('image_sha256'):
            raise ContractError('输入图像摘要不一致')
        packet = packets[sid]
        diagnosis = locked_diagnosis(row, packet)
        catalog = evidence_catalog(packet)
        validate_prediction_source(diagnosis, catalog)
        prepared.append((row, packet, diagnosis, catalog))
    configuration = {'protocol': VERSION, 'producer': producer,
                     'inputs': [make_context(row, packet, producer) for row, packet, _, _ in prepared],
                     'max_attempts_per_episode': 2, 'description_policy': '确定性事实模板，模型理由仅留内部证据'}
    with run_lock(output / 'run.lock'):
        cfg = output / 'configuration.json'
        if cfg.exists() and read_json(cfg) != configuration:
            raise ContractError('输入、证据或生成版本变化，请另建输出目录')
        atomic_json(cfg, configuration)
        state_path = output / 'status.json'
        previous_state = read_json(state_path) if state_path.exists() else {}
        started = previous_state.get('started_at', time.time())
        reused = generated = failed = 0
        failures = []
        records = output / 'records'
        for position, (row, packet, diagnosis, catalog) in enumerate(prepared):
            sid = row['sample_id']
            target = records / (sid + '.json')
            archive = output / 'attempts' / sid
            previous = None
            if target.exists():
                try:
                    previous = read_json(target)
                except (ValueError, UnicodeError):
                    pass
                if reusable_record(previous, row, packet, producer):
                    reused += 1
                    continue
                preserve_attempt(target, archive)
            attempts = 0
            if (isinstance(previous, dict) and previous.get('status') in {'failed', 'in_progress'}
                    and previous.get('context') == make_context(row, packet, producer) and not retry_failed):
                attempts = min(2, max(0, int(previous.get('attempts', 0))))
            error = None
            grade = parse_grade(None, diagnosis, catalog) if diagnosis['is_intact'] else None
            while grade is None and attempts < 2:
                attempts += 1
                raw = None
                # 先登记尝试再调用模型：进程中断也消耗本轮额度，恢复不能无限重置次数。
                atomic_json(target, {'status': 'in_progress', 'context': make_context(row, packet, producer),
                                     'attempts': attempts, 'attempt_started_at': time.time()})
                try:
                    raw = generate(grade_messages(diagnosis, catalog, error), attempts)
                    grade = parse_grade(raw, diagnosis, catalog)
                except (ContractError, RuntimeError, ValueError) as exc:
                    error = str(exc)
                trace = {'context': make_context(row, packet, producer), 'attempt': attempts,
                         'raw_output': raw, 'contract_error': error if grade is None else None,
                         'time': time.time()}
                atomic_json(archive / (digest_json(trace) + '.json'), trace)
                if grade is None:
                    atomic_json(target, {'status': 'failed', 'context': make_context(row, packet, producer),
                                         'attempts': attempts, 'error': error})
            if grade is None:
                failed += 1
                failures.append({'sample_id': sid, 'attempts': attempts, 'error': error or '本版本已达到两次尝试上限'})
            else:
                record = complete_record(row, packet, producer, grade, attempts)
                if not reusable_record(record, row, packet, producer):
                    raise ContractError('新报告未通过独立重建校验')
                atomic_json(target, record)
                generated += 1
            atomic_json(state_path, {'status': 'running', 'started_at': started, 'updated_at': time.time(),
                                    'pid': os.getpid(), 'processed': position + 1, 'total': len(rows),
                                    'valid_reused': reused, 'valid_generated': generated, 'failed': failed})
        if failed:
            summary = {'status': 'incomplete', 'started_at': started, 'updated_at': time.time(),
                       'total': len(rows), 'valid_reused': reused, 'valid_generated': generated,
                       'failed': failed, 'failures': failures, 'result_published': False}
            atomic_json(state_path, summary)
            return summary
        finals = []
        for row, packet, _, _ in prepared:
            record = read_json(records / (row['sample_id'] + '.json'))
            if not reusable_record(record, row, packet, producer):
                raise ContractError('聚合时发现无效或不同来源版本的报告')
            finals.append(record['final'])
        preserve_attempt(result_path, output / 'previous_results')
        atomic_json(result_path, finals)
        summary = {'status': 'complete', 'protocol': VERSION, 'started_at': started, 'updated_at': time.time(),
                   'images': len(rows), 'valid_reused': reused, 'valid_generated': generated, 'failed': 0,
                   'intact': sum(locked['is_intact'] for _, _, locked, _ in prepared),
                   'disease': sum(not locked['is_intact'] for _, _, locked, _ in prepared),
                   'result_sha256': digest_file(result_path), 'configuration_sha256': digest_file(cfg),
                   'elapsed_wall_seconds': time.time() - started,
                   'timing_scope': '报告阶段含恢复等待；完整入口另记区域、特征、报告与评级总耗时'}
        atomic_json(state_path, summary)
        return summary


class FrozenGrader:
    def __init__(self):
        self.router = None

    def __call__(self, messages, attempt):
        import torch
        if self.router is None:
            from qwen_tool_router import QwenToolRouter
            torch.set_num_threads(4)
            self.router = QwenToolRouter(None, [])
            self.router.load_frozen()
        tokenizer = self.router.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokens = None
        try:
            encoded = tokenizer(prompt, return_tensors='pt').to('cuda')
            if encoded.input_ids.shape[1] > 24000:
                raise ContractError('评级证据上下文超过24000词元')
            with torch.inference_mode():
                tokens = self.router.model.generate(**encoded, max_new_tokens=512 if attempt == 1 else 1024,
                                                    do_sample=False, temperature=None, top_p=None, top_k=None,
                                                    pad_token_id=tokenizer.pad_token_id)
            return tokenizer.decode(tokens[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
        finally:
            del encoded, tokens


def model_signature(directory):
    directory = Path(directory)
    names = ['config.json', 'generation_config.json', 'tokenizer_config.json', 'model.safetensors.index.json']
    signatures = {name: digest_file(directory / name) for name in names if (directory / name).is_file()}
    if 'config.json' not in signatures:
        raise ContractError('缺少冻结评级模型配置')
    return {'model': str(directory), 'metadata_sha256': signatures,
            'weight_boundary': '平台/model只读挂载；本凭证固定配置及分片索引，不冒充全量权重SHA'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--packets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true', help='明确开始新的失败项重试轮次，每轮至多两次')
    args = parser.parse_args()
    # 新入口不再导入训练模块来设置环境，运行缓存须在模型导入前显式定位。
    runtime = args.output.resolve() / 'runtime'
    for key, name in {'HF_HOME': 'hf', 'TORCH_HOME': 'torch', 'XDG_CACHE_HOME': 'cache', 'TMPDIR': 'tmp',
                      'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda',
                      'ALIPPU_CONFIG_PATH': 'ppu', 'HGRTC_CACHE_PATH': 'ppu/hgrtc'}.items():
        directory = runtime / name
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    rows = read_json(args.manifest)
    packets = {safe_sample_id(row['sample_id']): read_json(args.packets / (row['sample_id'] + '.json')) for row in rows}
    parent = Path(__file__).resolve().parent
    producer = {'runner_sha256': digest_file(Path(__file__)), 'contract_sha256': digest_file(parent / 'report_contract.py'),
                'router_sha256': digest_file(parent / 'qwen_tool_router.py'), 'model': model_signature('/model/Qwen3.6-27B')}
    summary = run_reports(rows, packets, args.output, args.result, producer, FrozenGrader(), args.retry_failed)
    print(json.dumps({'状态': '协议合格完成' if summary['status'] == 'complete' else '仍有失败报告',
                      '报告摘要': summary}, ensure_ascii=False))
    return 0 if summary['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
