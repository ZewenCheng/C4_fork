"""原图视图合同与回映；坐标基于EXIF校正后图像，未知质量不猜测。"""
import math
from report_contract import ContractError, digest_file, digest_json


def checked_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        raise ContractError('坐标必须为四个有限数值')
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ContractError('坐标越界或面积无效')
    return list(box)


def load_views(path, crops=(), *, max_crops=2):
    from PIL import Image, ImageOps
    if type(max_crops) is not int or not 0 <= max_crops <= 2 or len(crops) > max_crops:
        raise ContractError('原图裁块预算超限')
    digest = digest_file(path)
    with Image.open(path) as source:
        raw_size = list(source.size)
        orientation = source.getexif().get(274, 1)
        original = ImageOps.exif_transpose(source).convert('RGB')
    width, height = original.size
    result = []
    for requested in [[0, 0, 1, 1], *crops]:
        b = checked_box(requested)
        pixel = [math.floor(b[0] * width), math.floor(b[1] * height),
                 math.ceil(b[2] * width), math.ceil(b[3] * height)]
        actual = [pixel[0] / width, pixel[1] / height, pixel[2] / width, pixel[3] / height]
        record = {'version': '原图裁块-v1', 'image_sha256': digest, 'raw_size_wh': raw_size,
                  'oriented_size_wh': [width, height], 'exif_orientation': orientation,
                  'coordinate_space': 'exif_oriented_original', 'crop_normalized_xyxy': actual,
                  'crop_pixels_xyxy': pixel, 'requested_crop': b, 'processor': None,
                  'tensor_shape': None, 'quality': {'blur': None, 'occlusion': None, 'scale': None}}
        record['view_id'] = digest_json(record)
        result.append((original.crop(pixel), record))
    return result


def map_box_to_original(box, view):
    # 输入是去除处理器填充后的视图归一化坐标；填充尚未还原时拒绝调用。
    b = checked_box(box)
    if view.get('coordinate_space') != 'exif_oriented_original':
        raise ContractError('未知坐标空间')
    x1, y1, x2, y2 = checked_box(view['crop_normalized_xyxy'])
    return [x1 + b[0] * (x2-x1), y1 + b[1] * (y2-y1),
            x1 + b[2] * (x2-x1), y1 + b[3] * (y2-y1)]
