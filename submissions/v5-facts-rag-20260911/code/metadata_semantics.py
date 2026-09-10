"""桥名沿用v2的EXIF近邻规则；结构位置与图像方位分开处理。"""
from __future__ import annotations

import math
from pathlib import Path
import re

VERSION = 'metadata-semantics-v1'
MAX_BRIDGE_DISTANCE_METRES = 2_000.0
# 直接沿用v2的公开数据聚合中心，不读取旧测试预测作为推理输入。
BRIDGE_REFERENCE_CENTRES = (
    ('东溪河特大桥', 31.6648392, 109.5862483),
    ('后溪河大桥', 31.4644505, 109.5945136),
    ('巫溪互通大桥', 31.4012816, 109.5898490),
    ('柏杨河大桥', 31.3990713, 109.5875495),
    ('白沙溪2号桥', 31.6700011, 109.5681713),
    ('白鹤路跨线桥', 31.4042911, 109.5942710),
    ('范家坪1号大桥', 31.5208190, 109.6190008),
    ('西溪河特大桥', 31.5378395, 109.6227319),
    ('青树湾1号大桥', 31.5087426, 109.6162838),
)
COMPONENTS = ('跨中', '梁底', '梁端', '梁体', '盖梁', '横梁', '腹板', '底板', '翼缘',
              '桥墩', '桥台', '墩柱', '墩顶', '台身', '支座', '伸缩缝', '桥面', '护栏',
              '挡墙', '轨枕', '道床', '扣件', '轨道板', '隧道衬砌')


def filename_side(filename):
    stem = Path(filename).stem
    if '左右幅' in stem:
        return None
    return next((side for side in ('左幅', '右幅') if side in stem), None)


def image_gps(path):
    """只读图像元数据；无有效EXIF时保留未知，不由目录猜桥名。"""
    from PIL import ExifTags, Image
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return None
            tag = getattr(getattr(ExifTags, 'IFD', object()), 'GPSInfo', 34853)
            gps = exif.get_ifd(tag) if hasattr(exif, 'get_ifd') else exif.get(tag)
            if not gps:
                return None
            coordinates = []
            for key, reference, negative in ((2, 1, 'S'), (4, 3, 'W')):
                degrees, minutes, seconds = (float(value) for value in gps[key])
                coordinate = degrees + minutes / 60.0 + seconds / 3600.0
                ref = gps.get(reference, '')
                ref = ref.decode('ascii') if isinstance(ref, bytes) else str(ref)
                coordinates.append(-coordinate if ref.upper().endswith(negative) else coordinate)
            latitude, longitude = coordinates
            if math.isfinite(latitude) and math.isfinite(longitude) and -90 <= latitude <= 90 and -180 <= longitude <= 180:
                return latitude, longitude
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        pass
    return None


def _distance_metres(first, second):
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000.0 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, value))))


def nearest_bridge_name(gps, filename):
    if gps is None:
        base, source, distance = '未知桥梁', '无EXIF坐标', None
    else:
        if len(gps) != 2 or not all(math.isfinite(value) for value in gps) or not (-90 <= gps[0] <= 90 and -180 <= gps[1] <= 180):
            raise ValueError('桥名坐标无效')
        base, latitude, longitude = min(BRIDGE_REFERENCE_CENTRES, key=lambda item: _distance_metres(gps, item[1:]))
        distance = _distance_metres(gps, (latitude, longitude))
        if distance > MAX_BRIDGE_DISTANCE_METRES:
            base, source = '未知桥梁', '超出公开坐标阈值'
        else:
            source = '公开坐标近邻'
    side = filename_side(filename)
    return (f'{base}（{side}）' if side else base), source, distance


def bridge_metadata(category, path):
    if category == '轨道':
        return {'value': '不适用', 'source': '轨道不适用桥名', 'distance_metres': None, 'certainty': 'not_applicable'}
    if category != '桥梁':
        raise ValueError('桥名结构域无效')
    name, source, distance = nearest_bridge_name(image_gps(Path(path)), Path(path).name)
    return {'value': name, 'source': source, 'distance_metres': distance,
            'certainty': 'candidate' if source == '公开坐标近邻' else 'unknown'}


def filename_location(filename):
    """只保留可辨部位或编号；不把拍摄序号、时间戳及任意文件名当成构件。"""
    stem = re.sub(r'^(?:左右幅|左幅|右幅)[_-]?', '', Path(filename).stem)
    stem = re.sub(r'\s+', ' ', stem.replace('_', '-')).strip()
    if not stem or len(stem) > 120 or re.match(r'^(?:DJI|IMG|DSC|DSCN|IMAGE|FRAME)[-\d]', stem, re.I):
        return None
    component = any(term in stem for term in COMPONENTS) or bool(re.search(r'第?[0-9一二三四五六七八九十]+(?:号)?(?:跨|墩|台)', stem))
    if component and re.fullmatch(r'[\u4e00-\u9fffA-Za-z0-9（）()#＃、.\- ]+', stem):
        return {'value': stem, 'source': '文件名结构部位', 'certainty': 'filename_metadata'}
    if re.fullmatch(r'\d{1,3}(?:-\d{1,3}){1,4}', stem):
        return {'value': stem + '（文件名编号，构件含义未核验）', 'source': '文件名编号', 'certainty': 'identifier_only'}
    return None


def structure_location(category, filename, defect_type, is_intact):
    if category not in {'桥梁', '轨道'}:
        raise ValueError('位置结构域无效')
    named = filename_location(filename)
    if named:
        return named
    if is_intact:
        return {'value': '未见病害部位（具体构件位置未知）', 'source': '完好预测且缺少构件元数据', 'certainty': 'unknown'}
    components = []
    # 沿用v2的粗粒度类型映射，但显式保留推断性质，不能升级为实测位置。
    for marker, component in (('裂缝', '混凝土构件'), ('钢筋锈蚀', '钢筋外露部位'),
                              ('钢结构锈蚀', '钢结构'), ('支座锈蚀', '支座'),
                              ('渗水泛碱', '混凝土表面'), ('渗水/泛碱', '混凝土表面'),
                              ('破损', '结构表面')):
        # 泛化的“裂缝”不说明材质；只有标签明确混凝土时才使用该材质。
        if marker == '裂缝' and '混凝土' not in defect_type:
            continue
        if marker in defect_type and component not in components:
            components.append(component)
    if components:
        return {'value': '、'.join(components) + '（按病害类型推断，具体位置未知）',
                'source': '病害类型关联构件', 'certainty': 'type_inference'}
    return {'value': category + '构件具体位置未知', 'source': '缺少构件元数据', 'certainty': 'unknown'}


def validate_metadata(category, bridge, location):
    if not isinstance(bridge, dict) or not isinstance(location, dict):
        raise ValueError('缺少桥名或位置来源凭证')
    value = bridge.get('value')
    if not isinstance(value, str) or not value or value in {'桥梁', '轨道'}:
        raise ValueError('分类目录不能作为桥名')
    if category == '轨道' and (value != '不适用' or bridge.get('source') != '轨道不适用桥名'):
        raise ValueError('轨道桥名必须为不适用')
    if category == '桥梁':
        base = re.sub(r'（[左右]幅）$', '', value)
        source, distance = bridge.get('source'), bridge.get('distance_metres')
        if source == '公开坐标近邻':
            if base not in {item[0] for item in BRIDGE_REFERENCE_CENTRES} or type(distance) not in (int, float) or not math.isfinite(distance) or not 0 <= distance <= MAX_BRIDGE_DISTANCE_METRES:
                raise ValueError('桥名近邻缺少阈值内来源')
        elif base != '未知桥梁' or source not in {'无EXIF坐标', '超出公开坐标阈值'}:
            raise ValueError('未知桥名来源不合法')
    text = location.get('value')
    if not isinstance(text, str) or not text or '图像' in text or text in {'全桥', '桥梁结构'}:
        raise ValueError('结构位置不能使用图像方位或无依据的全桥范围')
