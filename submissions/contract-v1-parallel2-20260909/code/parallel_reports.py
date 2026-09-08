"""共享模型批量调用及有界CPU读写；逐图尝试预算、身份和发布合同保持。"""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

from report_contract import (VERSION, ContractError, atomic_json, complete_record, digest_file,
                             digest_json, evidence_catalog, grade_messages, locked_diagnosis,
                             make_context, parse_grade, preserve_attempt, read_json, reusable_record,
                             run_lock, safe_sample_id, validate_prediction_source)


def run_reports_parallel(rows, packets, output, result_path, producer, generate, retry_failed=False,
                         *, batch_size=4, cpu_workers=4):
    if not 1 <= batch_size <= 8 or not 1 <= cpu_workers <= 8:
        raise ContractError('批量与CPU并发必须在1至8之间')
    if batch_size > 1 and not callable(getattr(generate, 'generate_batch', None)):
        raise ContractError('生成器未提供真实批量接口')
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ContractError('输入为空或样本ID重复')
    output, result_path = Path(output), Path(result_path)

    def prepare(row):
        sid = safe_sample_id(row['sample_id'])
        if digest_file(row['path']) != row.get('image_sha256'):
            raise ContractError('输入图像摘要不一致')
        packet = packets[sid]
        diagnosis, catalog = locked_diagnosis(row, packet), evidence_catalog(packet)
        validate_prediction_source(diagnosis, catalog)
        return {'row': row, 'packet': packet, 'diagnosis': diagnosis, 'catalog': catalog,
                'context': make_context(row, packet, producer), 'target': output / 'records' / (sid + '.json'),
                'archive': output / 'attempts' / sid, 'attempts': 0, 'error': None}

    with run_lock(output / 'run.lock'), ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        cases = list(pool.map(prepare, rows))
        configuration = {'protocol': VERSION, 'producer': producer,
                         'inputs': [c['context'] for c in cases], 'max_attempts_per_episode': 2,
                         'description_policy': '确定性事实模板，模型理由仅留内部证据',
                         'execution': {'batch_size': batch_size, 'cpu_workers': cpu_workers,
                                       'schedule': 'shared-model-batch-cpu-io-v1'}}
        cfg = output / 'configuration.json'
        if cfg.exists() and read_json(cfg) != configuration:
            raise ContractError('输入、证据或生成版本变化，请另建输出目录')
        atomic_json(cfg, configuration)
        state_path = output / 'status.json'
        previous_state = read_json(state_path) if state_path.exists() else {}
        started = previous_state.get('started_at', time.time())
        reused = generated = 0
        pending, failed, writes = [], [], []

        def store_complete(case, grade, trace=None):
            if trace is not None:
                atomic_json(case['archive'] / (digest_json(trace) + '.json'), trace)
            record = complete_record(case['row'], case['packet'], producer, grade, case['attempts'])
            if not reusable_record(record, case['row'], case['packet'], producer):
                raise ContractError('新报告未通过独立重建校验')
            atomic_json(case['target'], record)

        for case in cases:
            previous = None
            if case['target'].exists():
                try:
                    previous = read_json(case['target'])
                except (ValueError, UnicodeError):
                    pass
                if reusable_record(previous, case['row'], case['packet'], producer):
                    reused += 1
                    continue
                preserve_attempt(case['target'], case['archive'])
            if (isinstance(previous, dict) and previous.get('status') in {'failed', 'in_progress'}
                    and previous.get('context') == case['context'] and not retry_failed):
                case['attempts'] = min(2, max(0, int(previous.get('attempts', 0))))
            if case['diagnosis']['is_intact']:
                grade = parse_grade(None, case['diagnosis'], case['catalog'])
                writes.append(pool.submit(store_complete, case.copy(), grade))
                generated += 1
            elif case['attempts'] >= 2:
                failed.append(case)
            else:
                case['messages'] = grade_messages(case['diagnosis'], case['catalog'])
                pending.append(case)

        def drain_finished(wait=False):
            remaining = []
            for future in writes:
                if wait or future.done():
                    future.result()
                else:
                    remaining.append(future)
            writes[:] = remaining

        def mark_attempt(case):
            atomic_json(case['target'], {'status': 'in_progress', 'context': case['context'],
                                        'attempts': case['attempts'], 'attempt_started_at': time.time()})

        batch_count = 0
        while pending:
            drain_finished()
            # 重试使用独立1024输出预算；同一批不混入首次512预算请求。
            pending.sort(key=lambda c: (c['attempts'], len(c['messages'][1]['content']), c['row']['sample_id']))
            attempt = pending[0]['attempts']
            size = min(batch_size, sum(c['attempts'] == attempt for c in pending))
            batch, pending = pending[:size], pending[size:]
            for case in batch:
                case['attempts'] += 1
            # 同批每张图先持久化尝试，进程中断不能抹去已消费额度。
            list(pool.map(mark_attempt, batch))
            error = None
            try:
                requests = [(c['messages'], c['attempts']) for c in batch]
                if callable(getattr(generate, 'generate_batch', None)):
                    raw_values = generate.generate_batch(requests)
                else:
                    raw_values = [generate(*requests[0])]
                if not isinstance(raw_values, (list, tuple)) or len(raw_values) != len(batch):
                    raise ContractError('批量返回数量不符，拒绝位置错配')
            except (ContractError, RuntimeError, ValueError) as exc:
                raw_values = [None] * len(batch)
                error = str(exc)
            batch_count += 1
            for case, raw in zip(batch, raw_values):
                grade, item_error = None, error
                if item_error is None:
                    try:
                        grade = parse_grade(raw, case['diagnosis'], case['catalog'])
                    except (ContractError, RuntimeError, ValueError) as exc:
                        item_error = str(exc)
                trace = {'context': case['context'], 'attempt': case['attempts'], 'raw_output': raw,
                         'contract_error': item_error if grade is None else None, 'time': time.time()}
                if grade is not None:
                    writes.append(pool.submit(store_complete, case.copy(), grade, trace))
                    generated += 1
                else:
                    # 失败状态和重试同一文件，先完成写入，避免线程写入竞争。
                    atomic_json(case['archive'] / (digest_json(trace) + '.json'), trace)
                    case['error'] = item_error
                    atomic_json(case['target'], {'status': 'failed', 'context': case['context'],
                                                  'attempts': case['attempts'], 'error': item_error})
                    if case['attempts'] < 2:
                        case['messages'] = grade_messages(case['diagnosis'], case['catalog'], item_error)
                        pending.append(case)
                    else:
                        failed.append(case)
            atomic_json(state_path, {'status': 'running', 'started_at': started, 'updated_at': time.time(),
                                    'pid': os.getpid(), 'processed': generated + reused + len(failed), 'total': len(rows),
                                    'valid_generated': generated, 'valid_reused': reused, 'failed': len(failed),
                                    'batch_size': batch_size, 'cpu_workers': cpu_workers, 'batches': batch_count,
                                    'recent_batches': getattr(generate, 'batch_metrics', [])[-4:]})
        drain_finished(wait=True)
        if failed:
            summary = {'status': 'incomplete', 'started_at': started, 'updated_at': time.time(), 'total': len(rows),
                       'valid_reused': reused, 'valid_generated': generated, 'failed': len(failed),
                       'failures': [{'sample_id': c['row']['sample_id'], 'attempts': c['attempts'],
                                     'error': c['error'] or '本版本已达到两次尝试上限'} for c in failed],
                       'result_published': False}
        else:
            def final(case):
                record = read_json(case['target'])
                if not reusable_record(record, case['row'], case['packet'], producer):
                    raise ContractError('聚合时发现无效或不同来源版本的报告')
                return record['final']
            finals = list(pool.map(final, cases))
            preserve_attempt(result_path, output / 'previous_results')
            atomic_json(result_path, finals)
            summary = {'status': 'complete', 'protocol': VERSION, 'started_at': started, 'updated_at': time.time(),
                       'images': len(rows), 'valid_reused': reused, 'valid_generated': generated, 'failed': 0,
                       'intact': sum(c['diagnosis']['is_intact'] for c in cases),
                       'disease': sum(not c['diagnosis']['is_intact'] for c in cases),
                       'result_sha256': digest_file(result_path), 'configuration_sha256': digest_file(cfg),
                       'elapsed_wall_seconds': time.time() - started,
                       'timing_scope': '批量报告阶段含当前模型加载及CPU读写，完整入口另记秒数',
                       'execution': configuration['execution'], 'batches': batch_count,
                       'batch_metrics': getattr(generate, 'batch_metrics', []),
                       'model_load_seconds': getattr(generate, 'model_load_seconds', None)}
        atomic_json(state_path, summary)
        return summary
