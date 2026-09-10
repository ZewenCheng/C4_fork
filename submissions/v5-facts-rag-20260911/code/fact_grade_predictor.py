"""事实评级研究候选加载器：绑定模型、编码器与来源，不赋予正式准入。"""
import hashlib
import json
from pathlib import Path

import fact_protocol

SCHEMA = 'final-fact-grade-v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


class FactGradePredictor:
    def __init__(self, model_path, manifest_path, *, expected_manifest_sha256):
        if sha(manifest_path) != expected_manifest_sha256:
            raise ValueError('事实评级模型清单摘要不符')
        self.manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        m = self.manifest
        template = fact_protocol.encode_tabular_features(fact_protocol.normalize_facts({}))
        if (m.get('schema') != SCHEMA or m.get('research_only') is not True
                or m.get('promotion_eligible') is not False
                or m.get('training_count') != 632 or m.get('authorized_source_count') != 3178
                or m.get('encoder_version') != template['encoder_version']
                or m.get('feature_order_sha256') != template['feature_order_sha256']
                or m.get('feature_count') != len(template['values'])
                or m.get('encoder_source_sha256') != sha(fact_protocol.__file__)
                or m.get('model_sha256') != sha(model_path)):
            raise ValueError('事实评级模型、训练范围或编码器版本不匹配')
        if not isinstance(m.get('training_source_identity'), dict) or not m['training_source_identity']:
            raise ValueError('事实评级模型缺少训练来源')
        import joblib
        import sklearn
        if m.get('sklearn_version') != sklearn.__version__:
            raise ValueError('事实评级训练与推理sklearn版本不一致')
        self.model = joblib.load(model_path)
        if [int(x) for x in self.model.classes_] != m.get('classes'):
            raise ValueError('实际评级模型类别与清单不符')
        if self.model.n_features_in_ != m['feature_count']:
            raise ValueError('实际评级模型输入维度不符')
        if any(type(x) is not int or not 1 <= x <= 5 for x in m['classes']):
            raise ValueError('评级模型包含非法等级')
        self.manifest_sha256 = expected_manifest_sha256
        self.model_sha256 = m['model_sha256']
        self.manifest_path = str(Path(manifest_path))

    def prepare(self, fact_package):
        """事实编码和OneHot预处理均在纯模型计时之外。"""
        encoded = fact_protocol.encode_tabular_features(fact_package)
        if (encoded['feature_order_sha256'] != self.manifest['feature_order_sha256']
                or sha(fact_protocol.__file__) != self.manifest['encoder_source_sha256']):
            raise ValueError('评级调用时事实编码器发生变化')
        import numpy as np
        return self.model.named_steps['encode'].transform(np.asarray([encoded['values']], dtype=object))

    def predict_values(self, prepared):
        """仅执行拟合后的逻辑回归predict，供外层精确包围计时。"""
        return self.model.named_steps['grade'].predict(prepared)

    def format_prediction(self, raw_prediction, fact_package):
        grade = int(raw_prediction[0])
        if grade not in self.manifest['classes']: raise ValueError('模型返回未登记等级')
        evidence = sorted({sid for f in fact_package['facts'].values()
            if f['state'] in {'observed', 'estimated'} for sid in f['source_ids']})
        return {'predicted_grade': str(grade), 'model_version': self.manifest['model_sha256'],
                'evidence_ids': evidence, 'possible_grades': list(map(str, self.manifest['classes'])),
                'research_only': True, 'promotion_eligible': False,
                'manifest_sha256': self.manifest_sha256,
                '说明': '真实事实编码后的模型预测；嵌套验证未优于全2基线，不代表规范实测或正式准入。'}

    def predict_prepared(self, prepared, fact_package):
        return self.format_prediction(self.predict_values(prepared), fact_package)

    def __call__(self, fact_package):
        return self.predict_prepared(self.prepare(fact_package), fact_package)

    predict = __call__
