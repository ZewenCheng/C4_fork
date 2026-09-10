"""统一事实和有版本的表格编码；不从描述文本推断量测或阴性事实。"""
import copy
import hashlib
import json
import math

SCHEMA_VERSION = 'facts-v1'
ENCODER_VERSION = 'facts-tabular-v1'
STATES = ('observed', 'estimated', 'unknown', 'not_applicable', 'conflict')
TEXT_FIELDS = ('domain', 'component', 'defect_type', 'structure_location')
NUMBER_UNITS = {'crack_width': 'mm', 'crack_length': 'mm', 'defect_area': 'mm2',
                'image_area_ratio': 'ratio', 'component_area_ratio': 'ratio', 'defect_count': 'count'}
BOOL_FIELDS = ('through_crack', 'exposed_rebar', 'water_seepage', 'corrosion')
FIELD_ORDER = (*TEXT_FIELDS, *NUMBER_UNITS, *BOOL_FIELDS)
RECOVERABILITY = {'requires_observation', 'requires_metadata', 'requires_external_measurement',
                  'requires_adjudication', 'not_recoverable', 'not_needed', 'unknown'}
_FACT_KEYS = {'value', 'state', 'source_ids', 'unit', 'missing_reason', 'recoverability',
              'measurement_method', 'scale_source', 'target_level', 'denominator',
              'observation_scope', 'applicability_basis', 'candidates'}


class FactError(ValueError):
    """事实来源、量纲或缺失状态不满足合同。"""


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _convert(name, value, unit):
    expected = NUMBER_UNITS.get(name)
    conversions = {'mm': {'mm': 1, 'cm': 10, 'm': 1000},
                   'mm2': {'mm2': 1, 'cm2': 100, 'm2': 1000000},
                   'ratio': {'ratio': 1, '%': .01}, 'count': {'count': 1}}
    if expected is None:
        if unit is not None: raise FactError('非数值事实不得附加物理单位：' + name)
        return value, None
    if unit not in conversions[expected]: raise FactError('事实量纲不匹配：' + name)
    if value is not None:
        if not _number(value) or value < 0: raise FactError('事实数值必须有限且非负：' + name)
        value *= conversions[expected][unit]
        if not math.isfinite(value): raise FactError('单位换算溢出')
        if expected == 'ratio' and value > 1: raise FactError('面积比例超过1')
        if expected == 'count' and int(value) != value: raise FactError('数量必须为整数')
    return value, expected


def _source(source_id, sources):
    if not _text(source_id): raise FactError('来源ID必须为非空文本')
    s = sources.get(source_id)
    if not isinstance(s, dict) or not _text(s.get('kind')) or not _text(s.get('evidence_type')):
        raise FactError('来源不存在或缺少证据性质：' + str(source_id))
    if s['kind'] == 'model' and not _text(s.get('model_version')):
        raise FactError('模型来源缺少版本：' + source_id)
    return s


def _normalize_one(name, raw, sources, *, allow_conflict=True):
    if raw is None: raw = {}
    if not isinstance(raw, dict) or set(raw) - _FACT_KEYS: raise FactError('事实结构或字段非法：' + name)
    state = raw.get('state', 'unknown')
    if state not in STATES: raise FactError('事实状态非法：' + name)
    value, unit = _convert(name, raw.get('value'), raw.get('unit', NUMBER_UNITS.get(name)))
    known = state in {'observed', 'estimated'}
    if not known and value is not None: raise FactError('缺失、不适用和冲突不得附带单一事实值：' + name)
    ids = raw.get('source_ids', [])
    if not isinstance(ids, list) or any(not _text(i) for i in ids) or len(ids) != len(set(ids)):
        raise FactError('来源ID应为不重复字符串数组')
    for sid in ids: _source(sid, sources)
    missing = raw.get('missing_reason', None if known else 'not_provided')
    recovery = raw.get('recoverability', 'not_needed' if known else 'unknown')
    if recovery not in RECOVERABILITY: raise FactError('事实恢复类型非法')
    if known:
        if not ids or value is None or missing is not None: raise FactError('已知事实缺值、来源或错误附带缺失原因')
        if name in TEXT_FIELDS and not _text(value): raise FactError('文本事实必须为非空文本')
        if name in BOOL_FIELDS and type(value) is not bool: raise FactError('布尔事实必须为显式true或false')
    elif not _text(missing): raise FactError('非已知事实必须说明原因')
    fact = {'value': value, 'state': state, 'source_ids': list(ids), 'unit': unit,
            'missing_reason': missing, 'recoverability': recovery,
            **{k: copy.deepcopy(raw.get(k)) for k in ('measurement_method', 'scale_source',
                'target_level', 'denominator', 'observation_scope', 'applicability_basis')},
            'candidates': []}
    for key in ['measurement_method', 'scale_source', 'target_level', 'observation_scope', 'applicability_basis']:
        if fact[key] is not None and not _text(fact[key]): raise FactError('事实方法或范围应为非空文本')
    if state == 'not_applicable' and (not ids or not _text(fact['applicability_basis'])):
        raise FactError('不适用必须有来源与适用性依据')
    if state == 'conflict':
        candidates = raw.get('candidates', [])
        if not allow_conflict or not isinstance(candidates, list) or len(candidates) < 2:
            raise FactError('冲突必须保留至少两项已知候选')
        fact['candidates'] = [_normalize_one(name, c, sources, allow_conflict=False) for c in candidates]
        if any(c['state'] not in {'observed', 'estimated'} for c in fact['candidates']):
            raise FactError('冲突候选必须包含值与来源')
        if len({_digest(c['value']) for c in fact['candidates']}) < 2: raise FactError('相同值不能标为值冲突')
        if set(ids) != {sid for c in fact['candidates'] for sid in c['source_ids']}:
            raise FactError('冲突来源必须覆盖全部候选')
    elif raw.get('candidates'): raise FactError('非冲突事实不能夹带冲突候选')
    if known and name in NUMBER_UNITS:
        if not _text(fact['measurement_method']) or not _text(fact['target_level']):
            raise FactError('数值事实缺少测量方法或目标层级')
        if unit in {'mm', 'mm2'}:
            scale = fact['scale_source']
            if not _text(scale): raise FactError('物理量缺少尺度或仪器来源')
            s = _source(scale, sources)
            if s['kind'] not in {'calibration', 'measurement'} or s['evidence_type'] not in {'metric_calibration', 'physical_measurement'}:
                raise FactError('图像模型分数不能充当物理尺度')
        if unit == 'ratio':
            d = fact['denominator']; expected_kind = 'image' if name == 'image_area_ratio' else 'component_surface'
            if (not isinstance(d, dict) or set(d) != {'kind', 'value', 'unit', 'source_ids'}
                    or d['kind'] != expected_kind or not _number(d['value']) or d['value'] <= 0
                    or d['unit'] not in {'px2', 'mm2', 'cm2', 'm2'}
                    or not isinstance(d['source_ids'], list) or not d['source_ids']):
                raise FactError('面积比例分母缺失或类型混用')
            if name == 'image_area_ratio' and (d['unit'] != 'px2' or fact['target_level'] != 'image'):
                raise FactError('图像面积比例必须使用图像像素分母')
            if name == 'component_area_ratio' and fact['target_level'] != 'component':
                raise FactError('构件面积比例目标必须为构件')
            for sid in d['source_ids']: _source(sid, sources)
    if known and name in BOOL_FIELDS and value is False:
        if not _text(fact['observation_scope']) or not _text(fact['measurement_method']):
            raise FactError('阴性事实必须记录检查范围和方法')
    if known:
        for sid in ids:
            source = sources[sid]
            if 'facts' in source:
                if not isinstance(source['facts'], dict): raise FactError('来源事实绑定必须为映射')
                binding = source['facts'].get(name, {})
                if not isinstance(binding, dict): raise FactError('来源事实绑定项必须为映射')
                supported, supported_unit = _convert(name, binding.get('value'), binding.get('unit', NUMBER_UNITS.get(name)))
                if type(supported) is not type(value) and name in BOOL_FIELDS:
                    raise FactError('来源布尔语义不一致')
                if supported != value or supported_unit != unit: raise FactError('来源事实绑定不支持当前值：' + name)
    return fact


def normalize_facts(payload, *, source_catalog=None):
    """补齐未知字段、同语义单位换算并验证来源；不解析自然语言描述。"""
    if not isinstance(payload, dict) or set(payload) - {'schema_version', 'sample_id', 'facts', 'sources', 'prediction_candidates'}:
        raise FactError('事实包字段非法，标注与描述不能自动作为线上事实')
    if payload.get('schema_version', SCHEMA_VERSION) != SCHEMA_VERSION: raise FactError('事实协议版本不匹配')
    supplied = payload.get('facts', {})
    if not isinstance(supplied, dict) or set(supplied) - set(FIELD_ORDER): raise FactError('未知事实名称')
    sources = copy.deepcopy(payload.get('sources', {}))
    if not isinstance(sources, dict) or (source_catalog is not None and not isinstance(source_catalog, dict)):
        raise FactError('来源目录必须为映射')
    for key, item in (source_catalog or {}).items():
        if key in sources and sources[key] != item: raise FactError('内外来源目录发生冲突')
        sources[key] = copy.deepcopy(item)
    candidates = copy.deepcopy(payload.get('prediction_candidates', []))
    if not isinstance(candidates, list): raise FactError('预测候选必须为数组')
    for c in candidates:
        if (not isinstance(c, dict) or set(c) != {'label', 'uncalibrated_score', 'source_ids'}
                or not _text(c['label']) or not _number(c['uncalibrated_score'])
                or not isinstance(c['source_ids'], list) or not c['source_ids']):
            raise FactError('预测候选仅接受有来源的未校准分数')
        for sid in c['source_ids']: _source(sid, sources)
    result = {'schema_version': SCHEMA_VERSION, 'sample_id': payload.get('sample_id'),
              'sources': sources, 'facts': {name: _normalize_one(name, supplied.get(name), sources) for name in FIELD_ORDER},
              'prediction_candidates': candidates}
    if result['sample_id'] is not None and not _text(result['sample_id']): raise FactError('样本身份非法')
    try: json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc: raise FactError('事实包不可安全序列化') from exc
    return result


def validate_facts(packet, *, source_catalog=None):
    """验证已规范化包，拒绝未显式规范化的数据和旧模型输入。"""
    normalized = normalize_facts(packet, source_catalog=source_catalog)
    comparison = copy.deepcopy(packet)
    if source_catalog is not None: comparison['sources'] = normalized['sources']
    if normalized != comparison: raise FactError('事实包尚未规范化')
    return True


def encode_tabular_features(packet):
    """固定数值哨兵与缺失/状态掩码；不拟合插补器，不兼容旧八维模型。"""
    validate_facts(packet)
    names, values, categorical = [], [], []
    def add(name, value, category=False):
        if category: categorical.append(len(values))
        names.append(name); values.append(value)
    for name in FIELD_ORDER:
        f = packet['facts'][name]; known = f['state'] in {'observed', 'estimated'}
        value = f['value'] if known else ('__missing__' if name in TEXT_FIELDS else 0)
        if name in BOOL_FIELDS and known: value = int(value)
        add(name + '_value', value, name in TEXT_FIELDS)
        add(name + '_known', int(known))
        for state in STATES: add(name + '_state_' + state, int(f['state'] == state))
        if name in NUMBER_UNITS:
            add(name + '_measurement_method', f['measurement_method'] or '__missing__', True)
            add(name + '_target_level', f['target_level'] or '__missing__', True)
    return {'schema_version': SCHEMA_VERSION, 'encoder_version': ENCODER_VERSION,
            'feature_names': names, 'values': values, 'categorical_indices': categorical,
            'feature_order_sha256': _digest(names), 'missing_numeric_sentinel': 0,
            '说明': '哨兵0仅用于模型编码；必须连同known及五状态列读取，不代表物理阴性。'}


def facts_from_prediction(prediction, *, category=None, metadata=None, source_catalog=None, sample_id=None):
    """从CV候选及显式可信元数据建旁路事实；不读取标签、描述或评级。"""
    if not isinstance(prediction, dict): raise FactError('CV预测必须为映射')
    candidates = prediction.get('candidates', [])
    version = prediction.get('checkpoint_sha256')
    if (not _text(version) or not isinstance(candidates, list) or not candidates
            or any(not isinstance(c, dict) for c in candidates)):
        raise FactError('CV预测缺少冻结模型来源或候选')
    first = candidates[0]
    if not isinstance(first, dict) or not _text(first.get('label')): raise FactError('CV首选无效')
    sid = 'cv:' + version
    sources = {sid: {'kind': 'model', 'model_version': version, 'evidence_type': 'model_inference',
                     'facts': {'defect_type': {'value': first['label'], 'unit': None}}}}
    facts = {'defect_type': {'value': first['label'], 'state': 'estimated', 'source_ids': [sid]}}
    if category is not None:
        if category not in {'桥梁', '轨道'}: raise FactError('结构域元数据无效')
        sources['metadata:domain'] = {'kind': 'metadata', 'evidence_type': 'declared_input_domain',
                                     'facts': {'domain': {'value': category, 'unit': None}}}
        facts['domain'] = {'value': category, 'state': 'observed', 'source_ids': ['metadata:domain']}
    metadata = metadata or {}
    if not isinstance(metadata, dict): raise FactError('元数据必须为映射')
    for name in ['component', 'structure_location']:
        item = metadata.get(name)
        if not isinstance(item, dict): continue
        if item.get('state') in {'observed', 'estimated'} and item.get('source_ids'):
            facts[name] = copy.deepcopy(item)
        elif item.get('certainty') in {'filename_metadata', 'verified_metadata', 'type_inference'} and _text(item.get('value')):
            source_id = 'metadata:' + name + ':' + _digest(item)[:16]
            sources[source_id] = {'kind': 'derived' if item['certainty'] == 'type_inference' else 'metadata',
                'evidence_type': item['certainty'], 'facts': {name: {'value': item['value'], 'unit': None}}}
            facts[name] = {'value': item['value'], 'state': 'estimated' if item['certainty'] == 'type_inference' else 'observed',
                           'source_ids': [source_id]}
    for name in NUMBER_UNITS:
        facts[name] = {'state': 'unknown', 'missing_reason': 'no_metric_scale' if NUMBER_UNITS[name] in {'mm', 'mm2'} else 'not_observed',
                       'recoverability': 'requires_external_measurement' if NUMBER_UNITS[name] in {'mm', 'mm2'} else 'requires_observation'}
    return normalize_facts({'schema_version': SCHEMA_VERSION, 'sample_id': sample_id,
        'sources': sources, 'facts': facts, 'prediction_candidates': [
            {'label': c.get('label'), 'uncalibrated_score': c.get('uncalibrated_score'), 'source_ids': [sid]} for c in candidates]},
        source_catalog=source_catalog)
