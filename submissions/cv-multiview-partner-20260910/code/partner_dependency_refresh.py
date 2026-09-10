"""改判后的依赖重建：重新选区域、查询原文，删除旧类型历史与评级。"""
import copy
from pathlib import Path
from report_contract import (ContractError, digest_file, digest_json, evidence_catalog,
    select_region_candidates, locked_diagnosis, QUERY_MAP)


class StandardRetriever:
    """复用现有检索实现，显式绑定索引目录，避免改写模块全局 ROOT。"""
    def __init__(self, index):
        self.index = Path(index)
        self.assets = {name: digest_file(self.index / name) for name in
                       ['vectorizer.joblib', 'matrix.npz', 'chunks.json']}

    def __call__(self, label):
        from query_standard import load_index
        for name, expected in self.assets.items():
            if digest_file(self.index / name) != expected:
                raise ContractError('规范索引在依赖刷新期间变化')
        vectorizer, matrix, chunks = load_index(self.index, tuple(sorted(self.assets.items())))
        scores = (matrix @ vectorizer.transform([label]).T).toarray().ravel()
        choices = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0][:2]
        return {'tool': 'query_standard', 'evidence_type': 'public_standard_text',
                'query': label, 'results': [dict(chunks[i], retrieval_similarity=float(scores[i]))
                                           for i in choices],
                'rating_decision': None, 'index_sha256': self.assets,
                'reason': '重新检索的公开原文；相关度不代表条款适用或规范等级'}


class DependencyRefresh:
    """供 commit_revision 的 refresh(diagnosis, proposal) 回调；自身不开放采用门禁。

    regions/extent 为已核验输入和模型来源的 sample_id 工具；缺少程度工具明确未知。
    standards 为 StandardRetriever 或同协议的原文检索器，None 明确未知。
    此对象不写入原 packet；任一现有工具损坏则抛 ContractError，由上层整案回退。
    """
    def __init__(self, row, packet, *, regions, extent=None, standards=None, limit=5):
        if not callable(regions) or type(limit) is not int or not 1 <= limit <= 20:
            raise ContractError('依赖刷新缺少区域工具或候选预算非法')
        self.row = copy.deepcopy(row)
        self.packet = copy.deepcopy(packet)
        self.regions, self.extent, self.standards, self.limit = regions, extent, standards, limit
        self.version = {'implementation': digest_file(__file__), 'packet': digest_json(packet),
                        'limit': limit, 'standards': getattr(standards, 'assets', None),
                        'history_policy': 'remove_old_type_no_validated_recompute'}

    def __call__(self, diagnosis, proposal):
        try:
            return self._refresh(diagnosis, proposal)
        except (OSError, TypeError, KeyError, ValueError, RuntimeError) as exc:
            raise ContractError('改判依赖刷新失败：' + type(exc).__name__) from exc

    def _refresh(self, diagnosis, proposal):
        label = proposal['label']
        if digest_file(self.row['path']) != self.row['image_sha256']:
            raise ContractError('改判输入内容变化')
        changed = copy.deepcopy(self.packet)
        changed['predicted_type'] = label
        expected = locked_diagnosis(self.row, changed)
        if any(diagnosis.get(k) != v for k, v in expected.items() if k != 'source'):
            raise ContractError('改判元数据没有依据新类型重新计算')
        if diagnosis.get('source') != 'partner_decision/revision1':
            raise ContractError('改判来源不是伙伴事务')
        evidence, audit = {}, {}
        for tool, callback in [('inspect_regions', self.regions), ('inspect_extent', self.extent)]:
            if callback is None:
                audit[tool] = {'status': 'unknown', 'reason': '未配置可核验程度重建器'}
                continue
            envelope = copy.deepcopy(callback({'sample_id': self.row['sample_id']}))
            if envelope.get('status') != 'ok':
                raise ContractError('改判工具没有可用完整回应：' + tool)
            if envelope.get('result', {}).get('sample_id') != self.row['sample_id']:
                raise ContractError('改判工具回应缺少当前案件身份：' + tool)
            original = evidence_catalog({'sample_id': self.row['sample_id'], 'evidence': {tool: envelope}})
            retained = {}
            for variant, output in envelope['result'].get('outputs', {}).items():
                candidates = output.get('candidates', [])
                selected = select_region_candidates(candidates, label, self.limit)
                output['candidates'] = selected
                retained[variant] = {'received': len(candidates), 'retained': len(selected),
                                     'selected_sha256': digest_json(selected)}
            evidence[tool] = envelope
            audit[tool] = {'status': 'refreshed', 'received_catalog_sha256': digest_json(original),
                           'selection': retained, 'response_sha256': digest_json(envelope)}
        catalog = evidence_catalog({'sample_id': self.row['sample_id'], 'evidence': evidence})
        # 不复制任何旧语义摘要、旧类型规范、旧历史等级或旧评级理由。
        for item in catalog.values():
            item['dependency_type'] = label
            item['decision_revision'] = 1
        queries = sorted({q for term, q in QUERY_MAP if term in label})
        audit['matching_queries'] = queries
        if self.standards is None:
            standard = {'status': 'unknown', 'reason': '未配置可核验规范检索索引', 'results': []}
        else:
            standard = self.standards(label)
            if (not isinstance(standard, dict) or standard.get('query') != label or
                    standard.get('rating_decision') is not None or not isinstance(standard.get('results'), list)):
                raise ContractError('新规范查询身份或等级边界不符')
            standard = {'status': 'refreshed', **copy.deepcopy(standard)}
        catalog['query_standard/revision1'] = {'tool': 'query_standard', 'kind': 'retrieval_context',
            'dependency_type': label, 'decision_revision': 1, 'result': standard}
        catalog['assess_history/revision1'] = {'tool': 'assess_history', 'kind': 'unknown',
            'dependency_type': label, 'decision_revision': 1, 'status': 'unknown',
            'reason': '旧类型历史预测已移除；尚无经验证的新类型历史重算，不能沿用旧等级'}
        catalog['partner_decision/revision1'] = {'tool': 'partner_decision', 'kind': 'adopted_model_prediction',
            'label': label, 'dependency_type': label, 'decision_revision': 1,
            'support_evidence_ids': list(proposal['support']), 'reason': proposal.get('reason', ''),
            'limitation': '采用门禁由上层独立验证；该记录不是新视觉观测或规范实测'}
        return {'decision_revision': 1, 'defectType': label, 'catalog': catalog, 'diagnosis': diagnosis,
                'audit': audit, 'version': self.version,
                'grade_source': 'partner_decision/revision1' if diagnosis['is_intact'] else 'refreshed_model_prediction',
                'removed_dependencies': ['old_semantics', 'old_standard', 'old_history', 'old_rating']}


def intact_revision_grade(diagnosis, catalog):
    """完好沿用空评级合同，引用真正的新决策，不能冒用旧分类首选。"""
    source = 'partner_decision/revision1'
    if not diagnosis['is_intact'] or catalog.get(source, {}).get('label') != diagnosis['defectType']:
        raise ContractError('完好修订缺少对应伙伴决策来源')
    return {'rating': '', 'reason': '按完好评级留空合同执行', 'evidence_ids': [source], 'kind': 'intact_policy'}
