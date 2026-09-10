"""伙伴旁路证据合同：稳定身份、来源家族和逐阶段候选传递。"""
import copy
import math
from collections import Counter
from report_contract import ContractError, digest_json, safe_sample_id, select_region_candidates

VERSION = 'partner-r4-v1'
FAMILIES = {'inspect_cv_multiview': 'convnext_shared', 'inspect_semantics': 'fusion_with_hplus_wemm_convnext', 'inspect_hplus': 'hplus_shared',
            'inspect_local': 'convnext_shared', 'inspect_regions': 'rtdetr_soft_teachers',
            'inspect_masks': 'sam_soft_teachers', 'inspect_grounding': 'grounding_soft_teachers'}
FAMILIES.update(inspect_extent='sam_soft_teachers',inspect_crops='rtdetr_soft_teachers',
                inspect_v2_rail='v2_shared_convnext_and_multibackbone')


def packet_ledger(row, packet):
    sid = safe_sample_id(row['sample_id'])
    if packet.get('sample_id') != sid:
        raise ContractError('伙伴案件串图')
    if not isinstance(packet.get('evidence'), dict):
        raise ContractError('缺少伙伴证据映射')
    try:
        packet_digest = digest_json(packet)
    except (ValueError, TypeError) as exc:
        raise ContractError('伙伴证据包含非有限数值或不可序列化对象') from exc
    ledger = {'version': VERSION, 'case_id': sid, 'input_sha256': row['image_sha256'],
              'packet_sha256': packet_digest, 'decision_revision': 0,
              'input_verification': 'declared_manifest_digest',
              'observations': [], 'sources': [], 'limitations': []}
    for tool, envelope in sorted(packet['evidence'].items()):
        if not isinstance(envelope, dict) or envelope.get('status') not in {'ok', 'partial', 'invalid', 'unavailable'}:
            raise ContractError('未知伙伴响应状态')
        status = envelope['status']
        ledger['sources'].append({'tool': tool, 'status': status, 'source_family': FAMILIES.get(tool, tool)})
        if status not in {'ok', 'partial'}:
            ledger['limitations'].append({'tool': tool, 'reason': status})
            continue
        result = envelope.get('result')
        if not isinstance(result, dict) or result.get('sample_id', sid) != sid:
            raise ContractError('伙伴响应身份无效')
        outputs = result.get('outputs', {})
        if not isinstance(outputs, dict):
            raise ContractError('伙伴分支必须为映射')
        for variant, output in sorted(outputs.items()):
            # H+原生分支是列表；稳定ID不能依赖其在列表中的名次。
            branch = {'candidates': output} if isinstance(output, list) else output
            if not isinstance(branch, dict) or branch.get('sample_id', sid) != sid:
                raise ContractError('伙伴分支身份无效')
            counts = Counter()
            for index, candidate in enumerate(branch.get('candidates', [])):
                if not isinstance(candidate, dict):
                    raise ContractError('伙伴候选必须为映射')
                score = candidate.get('uncalibrated_score')
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ContractError('伙伴候选分数无效')
                if 'box_normalized_xyxy' in candidate:
                    from input_quality import checked_box
                    checked_box(candidate['box_normalized_xyxy'])
                identity = {'case_id': sid, 'input_sha256': row['image_sha256'], 'tool': tool,
                            'variant': variant, 'checkpoint': branch.get('checkpoint_sha256'),
                            'preprocessing': branch.get('preprocessing_sha256'), 'candidate': candidate}
                base = digest_json(identity); counts[base] += 1
                ledger['observations'].append({'evidence_id': base + '-' + str(counts[base]),
                    'baseline_alias': f'{tool}/{variant}/{index}',
                    'tool': tool, 'variant': variant, 'source_family': FAMILIES.get(tool, tool),
                    'status': status, 'kind': 'model_hypothesis', 'checkpoint_sha256': identity['checkpoint'],
                    'checkpoint_known': identity['checkpoint'] is not None, 'candidate': copy.deepcopy(candidate)})
            total = branch.get('total_candidates')
            if total is not None:
                ledger['limitations'].append({'tool': tool, 'variant': variant, 'cached_count': total,
                                              'packet_count': len(branch.get('candidates', [])),
                                              'internal_query_count': None})
    return ledger


def select_with_trace(candidates, label, *, limit=5):
    selected = select_region_candidates(candidates, label, limit)
    remaining = Counter(digest_json(c) for c in selected)
    trace = []
    for index, c in enumerate(candidates):
        key = digest_json(c); kept = remaining[key] > 0
        if kept:
            remaining[key] -= 1
        trace.append({'source_index': index, 'candidate_sha256': key, 'retained': kept,
                      'reason': '类型与查询覆盖筛选保留' if kept else '外层证据预算截断'})
    return copy.deepcopy(selected), {'received': len(candidates), 'sent': len(selected),
        'limit': limit, 'internal_query_count': None, 'items': trace,
        'coverage_kind': '候选保留，非真实病害召回'}
