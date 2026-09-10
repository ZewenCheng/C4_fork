"""在当前输入特征上重算两套融合头，不把历史开发缓存当成新输入。"""
from pathlib import Path
from threading import Lock
import time
from report_contract import ContractError, read_json, digest_file


def checked_training_source(asset_root, relative, expected):
    """仅接受原根内文件或提交入口已绑定的外置资产，摘要始终精确匹配。"""
    root = Path(asset_root).resolve()
    rel = Path(relative)
    allowed = {'hplus_distilled_features', 'wemm_features', 'legacy_convnext_features'}
    if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] not in allowed:
        raise ContractError('融合训练来源路径不合法')
    source = (root / rel / 'source.json').resolve()
    top = root / rel.parts[0]
    mapped_root = top.resolve()
    # 内部文件仍按原资产根限制；外置链接须匹配入口已固定的映射凭证。
    if not source.is_relative_to(root):
        receipt_path = root / 'submission_entry_receipt.json'
        if not receipt_path.is_file():
            raise ContractError('融合外置训练来源缺少入口映射凭证')
        receipt = read_json(receipt_path)
        declared = receipt.get('assets', {}).get(rel.parts[0])
        if (receipt.get('protocol') != 'c4-submission-contract-v1'
                or Path(receipt.get('work', '')).resolve() != root
                or not isinstance(declared, str) or not Path(declared).is_absolute()
                or Path(declared).resolve() != mapped_root
                or not source.is_relative_to(mapped_root)):
            raise ContractError('融合外置训练来源与入口映射不符')
    if digest_file(source) != expected:
        raise ContractError('融合训练来源发生变化')
    return source


class SemanticPartner:
    def __init__(self, rows, feature_root, asset_root):
        import torch
        from train_fusion import Fusion
        self.rows = {r['sample_id']: r for r in rows}
        if len(self.rows) != len(rows): raise ContractError('语义工具清单存在重复案件')
        self.features = Path(feature_root); self.assets = Path(asset_root)
        self.labels = read_json(self.assets/'data/vocabulary.json')['raw_classes']
        self.models = {}; self.weights = {}; self.sources = {}; self.widths = {}; self.lock = Lock()
        for opt in ['adamw', 'sgd_momentum']:
            folder = self.assets/'checkpoints/semantic_fusion'/opt
            cfg = read_json(folder/'configuration.json'); receipt = read_json(folder/'complete.json')
            weight = folder/'last.pt'; checksum = digest_file(weight)
            if checksum != receipt['checkpoint_sha256'] or cfg['vocabulary_sha256'] != digest_file(self.assets/'data/vocabulary.json'):
                raise ContractError('融合头权重或词表不符')
            if cfg['classes'] != len(self.labels): raise ContractError('融合类别数不符')
            for relative, expected in cfg['sources'].items():
                source = checked_training_source(self.assets, relative, expected)
                self.sources[str(source)] = expected
            model = Fusion(cfg['widths'], cfg['classes'])
            model.load_state_dict(torch.load(weight, map_location='cpu', weights_only=False)['model'])
            self.models[opt] = model.eval(); self.weights[opt] = checksum; self.widths[opt] = cfg['widths']
            for family, prefix in [('hplus_distilled_features', 'dinov3_hplus_distilled_'), ('wemm_features', 'wemm9b_')]:
                path = self.features/family/opt/'source.json'; data = read_json(path)
                ck = self.assets/'checkpoints'/(prefix+opt)
                actual = digest_file(ck/'last.pt')
                if data['checkpoint_sha256'] != actual or read_json(ck/'complete.json')['checkpoint_sha256'] != actual:
                    raise ContractError('当前输入特征适配器不符')
                self.sources[str(path)] = digest_file(path)
        path = self.features/'legacy_convnext_features/source.json'; data = read_json(path)
        if digest_file(Path(data['base'])/'model.safetensors') != data['base_sha256']:
            raise ContractError('当前ConvNeXt基底不符')
        self.sources[str(path)] = digest_file(path)

    def __call__(self, arguments):
        import numpy as np
        import torch
        from torch.nn import functional as F
        if set(arguments) != {'sample_id'} or arguments['sample_id'] not in self.rows:
            raise ContractError('语义请求案件未登记')
        row = self.rows[arguments['sample_id']]; sid = row['sample_id']; started = time.time()
        if digest_file(row['path']) != row['image_sha256']: raise ContractError('语义输入内容变化')
        with self.lock, torch.inference_mode():
            for source, expected in self.sources.items():
                if digest_file(source) != expected: raise ContractError('当前输入特征来源发生变化')
            outputs = {}
            for opt, model in self.models.items():
                features = []; hashes = []
                for family, width in zip(['hplus_distilled_features', 'wemm_features', 'legacy_convnext_features'], self.widths[opt]):
                    folder = self.features/family if family == 'legacy_convnext_features' else self.features/family/opt
                    file = folder/(sid+'.npz')
                    with np.load(file, allow_pickle=False) as data: array = data['features'].astype(np.float32)
                    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != width or not np.isfinite(array).all():
                        raise ContractError('语义特征形状或数值无效')
                    f = torch.from_numpy(array)
                    features.append(F.normalize(F.normalize(f, dim=-1).mean(0), dim=-1).unsqueeze(0))
                    hashes.append(digest_file(file))
                logits, gates = model(features); values, indices = logits[0].softmax(-1).topk(min(10, len(self.labels)))
                outputs[opt] = {'checkpoint_sha256': self.weights[opt], 'feature_sha256': hashes,
                    'expert_order': ['hplus', 'wemm', 'convnext'], 'expert_weights': gates[0].tolist(),
                    'candidates': [{'label': self.labels[int(i)], 'label_id': int(i), 'uncalibrated_score': float(v)} for i, v in zip(indices, values)]}
        return {'status': 'ok', 'result': {'sample_id': sid, 'outputs': outputs}, 'cache_mode': 'current_input_head_recompute',
            'model_calls': 2, 'new_visual_observation': False, 'runtime_seconds': time.time()-started,
            'limitations': ['只重算分类头，不产生新视觉特征；分数与门控权重未校准']}
