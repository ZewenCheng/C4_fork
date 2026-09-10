"""确定性事实描述及逐断言语义校验；无支持内容回退，不生成物理量。"""
from copy import deepcopy
import math

VERSION = 'grounded-description-v1'
SAFE_TEXT = '当前证据不足以形成可核验的病害描述。'
LABELS = {'component': '构件', 'structure_location': '位置', 'defect_type': '病害类型',
          'crack_width': '裂缝宽度', 'crack_length': '裂缝长度', 'defect_area': '病害面积',
          'area_ratio': '病害面积占比', 'through_crack': '贯通裂缝'}
PHYSICAL = {'crack_width', 'crack_length', 'defect_area'}


def _supported(name, fact, sources):
    ids = fact.get('source_ids')
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        return False
    for sid in ids:
        source = sources.get(sid)
        if not isinstance(source, dict):
            return False
        bound = source.get('facts', {}).get(name)
        if (not isinstance(bound, dict) or type(bound.get('value')) is not type(fact.get('value'))
                or bound.get('value') != fact.get('value') or bound.get('unit') != fact.get('unit')):
            return False
        if name != 'defect_type' and source.get('kind') in ('cv_top1', 'classification', 'cv_classification'):
            return False
        if source.get('kind') in ('cv_top1', 'classification', 'cv_classification') and fact.get('state') != 'estimated':
            return False
        semantics = (('measurement_method', 'scale_source') if name in PHYSICAL else
                     ('denominator',) if name == 'area_ratio' else
                     ('observation_scope',) if name == 'through_crack' else ())
        if any(bound.get(key) != fact.get(key) for key in semantics):
            return False
    if name in PHYSICAL and (not fact.get('measurement_method') or not fact.get('scale_source')):
        return False
    if name in PHYSICAL and (type(fact.get('value')) not in (int, float) or not fact.get('unit')):
        return False
    if name == 'area_ratio' and not fact.get('denominator'):
        return False
    if name == 'through_crack' and not fact.get('observation_scope'):
        return False
    value = fact.get('value')
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return fact.get('state') in ('observed', 'estimated') and value is not None


def _render(claim, fact):
    name = claim['fact_name']; label = LABELS[name]
    if claim['state'] in ('unknown', 'conflict'):
        return label + ('存在证据冲突' if claim['state'] == 'conflict' else '未知')
    value = claim['value']
    prefix = '预测' if claim['state'] == 'estimated' else '记录的'
    if name == 'through_crack':
        if type(value) is not bool:
            raise ValueError('贯通事实必须为布尔值')
        return ('预测在' if claim['state'] == 'estimated' else '在') + str(fact['observation_scope']) + ('范围内检出贯通裂缝' if value else '范围内未检出贯通裂缝')
    suffix = '' if claim['unit'] is None else str(claim['unit'])
    text = prefix + label + '为' + str(value) + suffix
    if name == 'area_ratio':
        text += '（分母：' + str(fact['denominator']) + '）'
    return text


def _text(claims, facts):
    return '；'.join(_render(claim, facts[claim['fact_name']]) for claim in claims) + '。' if claims else SAFE_TEXT


def validate_description(output, fact_package, sources):
    if fact_package.get('schema_version') != 'facts-v1' or not isinstance(sources, dict):
        raise ValueError('描述事实或来源schema无效')
    if not isinstance(output, dict) or not isinstance(output.get('claims'), list):
        raise ValueError('描述缺少断言列表')
    facts = fact_package['facts']; seen = set()
    for claim in output['claims']:
        if not isinstance(claim, dict) or set(claim) != {'fact_name', 'value', 'unit', 'state', 'source_ids'}:
            raise ValueError('断言字段不符合模板')
        name = claim['fact_name']
        if name not in LABELS or name in seen or name not in facts:
            raise ValueError('未知、重复或无依据的断言')
        seen.add(name); fact = facts[name]
        if any(claim[key] != fact.get(key) for key in ('value', 'unit', 'state', 'source_ids')):
            raise ValueError('断言值、单位、状态或来源与事实不符')
        if fact.get('state') in ('unknown', 'conflict'):
            if fact.get('state') == 'unknown' and fact.get('value') is not None:
                raise ValueError('未知状态不得携带确定值')
        elif not _supported(name, fact, sources):
            raise ValueError('引用未支持该事实语义、量纲或测量依据')
    if output.get('text') != _text(output['claims'], facts):
        raise ValueError('文本包含模板断言之外的内容或状态改写')
    return True


def describe_facts(fact_package, *, sources, candidate=None):
    if fact_package.get('schema_version') != 'facts-v1' or not isinstance(fact_package.get('facts'), dict):
        raise ValueError('描述事实schema不兼容')
    facts = fact_package['facts']; claims = []
    defect = str(facts.get('defect_type', {}).get('value') or '')
    if not defect:
        relevant_unknown = {'defect_type'}
    elif any(word in defect.lower() for word in ('裂缝', '裂纹', 'crack')):
        relevant_unknown = {'crack_width'}
    elif any(word in defect for word in ('完好', '无病害', '正常')):
        relevant_unknown = set()
    else:
        relevant_unknown = {'component'}
    for name in LABELS:
        fact = facts.get(name)
        if not isinstance(fact, dict):
            continue
        if _supported(name, fact, sources) or (name in relevant_unknown and fact.get('state') == 'unknown' and fact.get('value') is None):
            claims.append({'fact_name': name, **{key: deepcopy(fact.get(key)) for key in ('value', 'unit', 'state', 'source_ids')}})
    baseline = {'schema_version': VERSION, 'text': _text(claims, facts), 'claims': claims}
    validate_description(baseline, fact_package, sources)
    if candidate is not None:
        try:
            validate_description(candidate, fact_package, sources)
            return deepcopy(candidate)
        except (ValueError, TypeError, KeyError):
            baseline['fallback_reason'] = '候选描述未通过逐断言值、单位、来源与文本核验'
    return baseline
