"""平台CPU单折固定线性对照；训练只读取本折训练标签，不扫描参数。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np


CANDIDATES = ('full_view', 'multiview')
PARAMETERS = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-4}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_npz(path, **values):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(temp, path)


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def feature_view(x, candidate):
    if candidate == 'full_view':
        return normalize(x[:, 0])
    if candidate == 'multiview':
        return normalize(normalize(x).mean(axis=1))
    raise ValueError('未预冻结的候选')


def validate_job(job, x, sample_ids, y):
    required = {'version', 'fold_id', 'feature_path', 'feature_sha256', 'output_dir',
                'seed', 'train_indices', 'eval_indices', 'train_sample_ids',
                'eval_sample_ids', 'candidates', 'threads', 'fit_authorized', 'data_role'}
    if set(job) != required or job['version'] != 1 or job['threads'] != 2:
        raise ValueError('单折任务字段或版本/线程数不符')
    if job['candidates'] != list(CANDIDATES):
        raise ValueError('候选集合及顺序必须预冻结')
    if type(job['seed']) is not int or type(job['fold_id']) is not int:
        raise ValueError('种子和折号必须为整数')
    if x.ndim != 3 or min(x.shape) <= 0 or not np.isfinite(x).all():
        raise ValueError('特征必须为有限N×V×D张量')
    if sample_ids.ndim != 1 or len(sample_ids) != len(x) or len(set(sample_ids.tolist())) != len(x):
        raise ValueError('样本身份缺失或重复')
    if sample_ids.dtype.kind not in 'US':
        raise ValueError('样本身份须为无对象字符串数组')
    if y.ndim != 2 or y.shape[0] != len(x) or y.shape[1] < 1 or not np.isin(y, [0, 1]).all():
        raise ValueError('标签必须为N×C二元多热数组')
    index_sets = []
    for partition in ('train', 'eval'):
        indices = job[partition + '_indices']
        if (not isinstance(indices, list) or not indices or
                any(type(i) is not int or i < 0 or i >= len(x) for i in indices) or
                len(set(indices)) != len(indices)):
            raise ValueError('分区索引为空、重复或越界')
        if sample_ids[indices].tolist() != job[partition + '_sample_ids']:
            raise ValueError('分区身份与特征索引不一致')
        index_sets.append(set(indices))
    if index_sets[0] & index_sets[1]:
        raise ValueError('训练与验证分区重叠')


def fit_candidate(train_x, train_y, eval_x, seed):
    """只接受训练标签；API不传入任何验证标签。"""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    train_x = np.asarray(train_x, dtype=np.float64)
    eval_x = np.asarray(eval_x, dtype=np.float64)
    classes = train_y.shape[1]
    coef = np.zeros((classes, train_x.shape[1]), dtype=np.float64)
    intercept = np.zeros(classes, dtype=np.float64)
    constant = np.full(classes, -1, dtype=np.int8)
    probabilities = np.zeros((len(eval_x), classes), dtype=np.float64)
    iterations = np.zeros(classes, dtype=np.int64)
    converged = np.ones(classes, dtype=bool)
    started = time.time()
    with threadpool_limits(limits=2):
        for column in range(classes):
            target = train_y[:, column]
            unique = np.unique(target)
            if len(unique) == 1:
                constant[column] = int(unique[0])
                probabilities[:, column] = float(unique[0])
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model = LogisticRegression(**PARAMETERS, random_state=seed)
                model.fit(train_x, target)
            converged[column] = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            coef[column] = model.coef_[0]
            intercept[column] = model.intercept_[0]
            iterations[column] = int(model.n_iter_[0])
            probabilities[:, column] = model.predict_proba(eval_x)[:, list(model.classes_).index(1)]
    if not np.isfinite(probabilities).all():
        raise ValueError('预测概率非有限')
    return {'coef': coef, 'intercept': intercept, 'constant': constant,
            'probabilities': probabilities, 'iterations': iterations,
            'converged': converged, 'fit_seconds': time.time() - started}


def run_job(job, resume=False):
    # 默认仅评估；授权字段必须来自已核实的数据使用合同，不得由编排器自行推定。
    if job.get('fit_authorized') is not True or job.get('data_role') != 'training_authorized':
        raise PermissionError('数据未明确授权拟合；evaluation_only数据禁止训练或校准')
    import sklearn
    output = Path(job['output_dir'])
    feature = Path(job['feature_path'])
    if not output.is_absolute() or not feature.is_absolute():
        raise ValueError('输入输出必须为绝对路径')
    identity = {'job_sha256': digest(job), 'feature_sha256': sha256(feature),
                'source_sha256': sha256(__file__), 'numpy_version': np.__version__,
                'sklearn_version': sklearn.__version__, 'parameters': PARAMETERS}
    if identity['feature_sha256'] != job['feature_sha256']:
        raise ValueError('特征文件摘要不符')
    receipt_path = output / 'receipt.json'
    if receipt_path.exists():
        if not resume:
            raise ValueError('完成目录已存在，需显式恢复')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('status') != 'complete' or receipt.get('identity') != identity:
            raise ValueError('完成凭证身份不符')
        expected = {candidate + suffix for candidate in CANDIDATES
                    for suffix in ('_model.npz', '_eval.npz')}
        if set(receipt.get('outputs', {})) != expected:
            raise ValueError('完成凭证输出集合不符')
        for name, checksum in receipt['outputs'].items():
            if sha256(output / name) != checksum:
                raise ValueError('恢复输出摘要不符')
        return receipt
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('非空工作目录需显式恢复')
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        with np.load(feature, allow_pickle=False) as data:
            x, sample_ids, y = data['x'], data['sample_ids'], data['y']
        validate_job(job, x, sample_ids, y)
        # 在创建任何模型前，冻结训练标签副本；后续不传递验证标签。
        train_y = y[job['train_indices']].astype(np.int8, copy=True)
        del y
        details, outputs = {}, {}
        for candidate in CANDIDATES:
            features = feature_view(x, candidate)
            result = fit_candidate(features[job['train_indices']], train_y,
                                   features[job['eval_indices']], job['seed'])
            details[candidate] = {'fit_seconds': result['fit_seconds'],
                                  'converged': result['converged'].tolist(),
                                  'iterations': result['iterations'].tolist(),
                                  'constant': result['constant'].tolist()}
            if not result['converged'].all():
                raise RuntimeError('训练未收敛，拒绝标为完成：' + candidate)
            model_path = output / (candidate + '_model.npz')
            atomic_npz(model_path, coef=result['coef'], intercept=result['intercept'],
                       constant=result['constant'], iterations=result['iterations'],
                       converged=result['converged'], train_sample_ids=sample_ids[job['train_indices']])
            eval_path = output / (candidate + '_eval.npz')
            atomic_npz(eval_path, probabilities=result['probabilities'],
                       sample_ids=sample_ids[job['eval_indices']])
            outputs[model_path.name] = sha256(model_path)
            outputs[eval_path.name] = sha256(eval_path)
        receipt = {'status': 'complete', 'fold_id': job['fold_id'], 'identity': identity,
                   'train_sample_ids': job['train_sample_ids'], 'eval_sample_ids': job['eval_sample_ids'],
                   'details': details, 'outputs': outputs, 'wall_seconds': time.time() - started,
                   '说明': '本折训练标签独立拟合；概率未校准；不在单折程序计算验证指标。'}
        atomic_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        atomic_json(output / 'failure.json', {'status': 'failed', 'identity': identity,
                    'error_type': type(error).__name__, 'message': str(error),
                    'wall_seconds': time.time() - started, 'details': locals().get('details', {})})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run_job(json.loads(args.job.read_text(encoding='utf-8')), resume=args.resume)
    print(json.dumps({'状态': '单折完成', '折号': result['fold_id'],
                      '墙钟秒': result['wall_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
