"""平台上将预先选定的多视图头在全部授权初赛训练样本重拟合一次。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from cv_multiview_classifier import SCHEMA, VIEWS, PARAMETERS, sha, digest, CVMultiviewClassifier
from fold_linear_worker import feature_view, fit_candidate, atomic_json, atomic_npz

CV = Path('/workspace/work/c4-local-training-cv-20260910-v2')


def fit(output):
    output = Path(output).resolve()
    boundary = Path('/workspace/work').resolve()
    if output == boundary or not output.is_relative_to(boundary):
        raise ValueError('模型输出必须在平台独立工作目录')
    plan = json.loads((CV / 'fold_plan.json').read_text())
    from fold_protocol import validate_fold_plan
    validate_fold_plan(plan)
    if plan.get('mode') != 'cross_validation' or any(not r['labels'] for r in plan['rows']):
        raise ValueError('缺少已授权带真值的初赛训练计划')
    job = json.loads((CV / 'job-0.json').read_text())
    feature_path = CV / 'features.npz'
    if (job['fit_authorized'] is not True or job['data_role'] != 'training_authorized'
            or sha(feature_path) != job['feature_sha256']):
        raise ValueError('训练授权或固定特征摘要不符')
    origin = json.loads((CV / 'feature_source.json').read_text())['origin']
    if (origin.get('frozen_base') is not True or origin.get('views') != VIEWS
            or sha(Path(origin['base']) / 'model.safetensors') != origin['base_sha256']):
        raise ValueError('冻结视觉基底或视图发生变化')
    labels = sorted({v for r in plan['rows'] for v in r['labels']})
    ids = [r['sample_id'] for r in plan['rows']]
    with np.load(feature_path, allow_pickle=False) as z:
        x, actual_ids, y = z['x'], z['sample_ids'], z['y']
    expected_y = np.asarray([[v in r['labels'] for v in labels] for r in plan['rows']], dtype=np.int8)
    if (len(ids) != 3178 or len(set(ids)) != 3178 or len(labels) != 70
            or actual_ids.tolist() != ids or not np.array_equal(y, expected_y)
            or x.ndim != 3 or x.shape[:2] != (3178, 3) or not np.isfinite(x).all()):
        raise ValueError('训练特征、全部样本及逐图标签绑定不符')
    identity = {'feature_sha256': sha(feature_path), 'plan_sha256': plan['plan_sha256'],
                'authorized_local_manifest_sha256': sha(CV / 'authorized_local_manifest.json'),
                'feature_source_sha256': sha(CV / 'feature_source.json'),
                'source_sha256': {p.name: sha(p) for p in [Path(__file__),
                    Path(__file__).with_name('cv_multiview_classifier.py'),
                    Path(__file__).with_name('fold_linear_worker.py')]}}
    if output.exists():
        raise ValueError('最终头目录已存在，禁止覆盖或无意重复拟合')
    output.mkdir(parents=True)
    started = time.time()
    features = feature_view(x, 'multiview')
    # 验证矩阵仅占位让既有拟合函数导出权重；不使用测试集，不计算训练准确率。
    result = fit_candidate(features, y, features[:1], 20260910)
    if not result['converged'].all():
        raise RuntimeError('最终固定头未全部收敛')
    model = output / 'multiview_model.npz'
    atomic_npz(model, coef=result['coef'], intercept=result['intercept'],
               constant=result['constant'], iterations=result['iterations'],
               converged=result['converged'], train_sample_ids=actual_ids)
    manifest = {'schema': SCHEMA, 'candidate': 'multiview',
                'fit_scope': 'all_authorized_preliminary_training', 'training_count': len(ids),
                'training_ids_sha256': digest(ids), 'labels': labels, 'feature_width': x.shape[2],
                'model_sha256': sha(model), 'base_sha256': origin['base_sha256'],
                'frozen_base': True, 'views': VIEWS, 'parameters': PARAMETERS, 'seed': 20260910,
                'identity': identity, 'fit_seconds': result['fit_seconds'],
                'wall_seconds': time.time() - started,
                '说明': '五折选择后全3178样本重新拟合；不选单折，不计算训练准确率，不读取线上测试标签。'}
    atomic_json(output / 'manifest.json', manifest)
    classifier = CVMultiviewClassifier(model, output / 'manifest.json', labels, origin)
    for i in [0, len(x) - 1]:
        classifier.predict(x[i])
    receipt = {'状态': '全量固定多视图头拟合并读回通过', '训练数': len(ids), '类别数': len(labels),
               '训练秒': result['fit_seconds'], '模型摘要': sha(model),
               '清单摘要': sha(output / 'manifest.json'), 'identity': identity}
    atomic_json(output / 'complete.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(fit(args.output), ensure_ascii=False))
