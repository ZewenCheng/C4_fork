"""研究评级加载器绑定与真实调用合同；不生成合成准确率结论。"""
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import sklearn
import fact_protocol
from fact_protocol import normalize_facts, encode_tabular_features
from fact_grade_predictor import FactGradePredictor, SCHEMA, sha


class IdentityTransformer:
    def transform(self, x): return x


class SyntheticDecision:
    def predict(self, x): return [3] * len(x)


class SyntheticPipeline:
    classes_ = [2, 3, 4]
    n_features_in_ = 110
    named_steps = {'encode': IdentityTransformer(), 'grade': SyntheticDecision()}
    def predict(self, x): return [3] * len(x)


class FactGradePredictorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.model = self.root / 'model.joblib'; self.manifest = self.root / 'manifest.json'
        joblib.dump(SyntheticPipeline(), self.model)
        template = encode_tabular_features(normalize_facts({}))
        self.m = {'schema': SCHEMA, 'research_only': True, 'promotion_eligible': False,
            'training_count': 632, 'authorized_source_count': 3178, 'encoder_version': template['encoder_version'],
            'feature_order_sha256': template['feature_order_sha256'], 'feature_count': 110,
            'encoder_source_sha256': sha(fact_protocol.__file__), 'model_sha256': sha(self.model),
            'training_source_identity': {'fixture': 'synthetic'}, 'classes': [2, 3, 4], 'sklearn_version': sklearn.__version__}
        self.save()
    def save(self): self.manifest.write_text(json.dumps(self.m))
    def load(self): return FactGradePredictor(self.model, self.manifest, expected_manifest_sha256=sha(self.manifest))
    def test_model_prediction_not_hardcoded_majority(self):
        result = self.load()(normalize_facts({}))
        self.assertEqual(result['predicted_grade'], '3'); self.assertFalse(result['promotion_eligible'])
        self.assertEqual(result['possible_grades'], ['2', '3', '4'])
    def test_rejects_unbound_manifest_and_changed_encoder(self):
        with self.assertRaises(ValueError): FactGradePredictor(self.model, self.manifest, expected_manifest_sha256='wrong')
        self.m['encoder_source_sha256'] = 'wrong'; self.save()
        with self.assertRaises(ValueError): self.load()
    def test_rejects_changed_model_and_wrong_features(self):
        self.m['model_sha256'] = 'wrong'; self.save()
        with self.assertRaises(ValueError): self.load()
        self.m['model_sha256'] = sha(self.model); self.m['feature_count'] = 8; self.save()
        with self.assertRaises(ValueError): self.load()


if __name__ == '__main__': unittest.main()
