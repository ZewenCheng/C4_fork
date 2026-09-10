"""加载全初赛训练集拟合的固定多视图头；只推理，不训练或校准。"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'c4-cv-multiview-classifier-v1'
VIEWS = ['full', 'horizontal_flip', 'center_square']
PARAMETERS = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 1e-4}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def multiview_features(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 3 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError('特征必须是有限的三视图矩阵')
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    x = x.mean(axis=0)
    return x / max(float(np.linalg.norm(x)), 1e-12)


class CVMultiviewClassifier:
    def __init__(self, model_path, manifest_path, expected_labels, feature_source):
        self.manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        m = self.manifest
        if (m.get('schema') != SCHEMA or m.get('candidate') != 'multiview'
                or m.get('fit_scope') != 'all_authorized_preliminary_training'
                or m.get('training_count') != 3178 or m.get('frozen_base') is not True
                or m.get('views') != VIEWS or m.get('parameters') != PARAMETERS):
            raise ValueError('分类头训练范围或预处理合同不符')
        self.labels = m.get('labels')
        if (not isinstance(self.labels, list) or len(self.labels) != 70
                or len(set(self.labels)) != 70 or set(self.labels) != set(expected_labels)
                or any(not isinstance(v, str) or not v.strip() for v in self.labels)):
            raise ValueError('分类头有序类别或调用词表不符')
        if (feature_source.get('base_sha256') != m.get('base_sha256')
                or feature_source.get('frozen_base') is not True
                or feature_source.get('views') != VIEWS):
            raise ValueError('线上特征基底或视图与训练来源不一致')
        self.model_sha256 = sha(model_path)
        if self.model_sha256 != m.get('model_sha256'):
            raise ValueError('分类头权重摘要不符')
        with np.load(model_path, allow_pickle=False) as z:
            self.coef = z['coef'].astype(np.float64)
            self.intercept = z['intercept'].astype(np.float64)
            self.constant = z['constant']
            ids = z['train_sample_ids'].tolist()
            converged = z['converged']
        if (len(ids) != 3178 or len(set(ids)) != 3178
                or digest(ids) != m.get('training_ids_sha256')):
            raise ValueError('分类头未覆盖全部授权训练样本')
        if (self.coef.ndim != 2 or self.coef.shape != (70, m.get('feature_width'))
                or self.intercept.shape != (70,) or self.constant.shape != (70,)
                or converged.shape != (70,) or not converged.all()
                or not np.isfinite(self.coef).all() or not np.isfinite(self.intercept).all()
                or not np.isin(self.constant, [-1, 0, 1]).all()):
            raise ValueError('分类头形状、收敛或数值异常')

    def predict(self, features):
        x = multiview_features(features)
        if x.shape != (self.coef.shape[1],):
            raise ValueError('线上特征维度不符')
        logits = self.coef @ x + self.intercept
        # 稳定 sigmoid；常数标签列与五折训练器完全一致。
        exp_neg = np.exp(-np.abs(logits))
        scores = np.where(logits >= 0, 1 / (1 + exp_neg), exp_neg / (1 + exp_neg))
        scores = np.where(self.constant >= 0, self.constant, scores)
        order = np.argsort(-scores, kind='stable')[:5]
        return {'checkpoint_sha256': self.model_sha256,
                'candidates': [{'label': self.labels[int(i)], 'label_id': int(i),
                                'uncalibrated_score': float(scores[i])} for i in order],
                'candidate': 'multiview', 'fit_scope': self.manifest['fit_scope'],
                'training_count': 3178, 'base_sha256': self.manifest['base_sha256'],
                '说明': '初赛训练集全量固定头；分数未校准，五折小幅分类收益不代表七字段提分。'}
