#!/usr/bin/env python3
"""为赛题四七字段结果补全非模型输出字段。

桥梁名称只在测试图 EXIF 坐标距离公开标注桥梁中心不超过 2 km 时回填；
超过阈值或没有坐标时明确写为“未知桥梁”，避免把弱线索伪装成真值。
其余字段由文件名、问题类别和现有 ``defectType`` 确定性生成。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import re
from typing import Any

from PIL import ExifTags, Image

from common import FIELDS, atomic_json, load_manifest, resolve_image_path


TARGET_FIELDS = ("bridgeName", "defectLocation", "defectDescription", "ratingScale(1-5)")
MAX_BRIDGE_DISTANCE_METRES = 2_000.0

# 由公开训练集与初赛测试集 EXIF GPS 按桥名聚合得到的中心点；只保留紧凑聚合值，
# 不把图片、文件名或逐图坐标带入作品代码。
BRIDGE_REFERENCE_CENTRES = (
    ("东溪河特大桥", 31.6648392, 109.5862483),
    ("后溪河大桥", 31.4644505, 109.5945136),
    ("巫溪互通大桥", 31.4012816, 109.5898490),
    ("柏杨河大桥", 31.3990713, 109.5875495),
    ("白沙溪2号桥", 31.6700011, 109.5681713),
    ("白鹤路跨线桥", 31.4042911, 109.5942710),
    ("范家坪1号大桥", 31.5208190, 109.6190008),
    ("西溪河特大桥", 31.5378395, 109.6227319),
    ("青树湾1号大桥", 31.5087426, 109.6162838),
)

DESCRIPTION_BY_DEFECT = {
    "完好": "未见明显病害",
    "支座锈蚀": "支座表面存在锈蚀",
    "裂缝(混凝土裂缝)": "混凝土构件可见裂缝",
    "破损": "结构表面存在破损",
    "钢结构锈蚀": "钢结构表面存在锈蚀",
    "钢筋锈蚀": "外露钢筋存在锈蚀",
    "渗水泛碱": "混凝土表面存在渗水泛碱",
    "钢筋锈蚀、破损": "外露钢筋存在锈蚀，结构表面存在破损",
    "破损、支座锈蚀": "结构表面存在破损，支座表面存在锈蚀",
}


def _blank(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def _dms_to_degrees(value: Any) -> float:
    parts = list(value)
    if len(parts) != 3:
        raise ValueError("GPS 度分秒必须包含三个值。")
    degrees, minutes, seconds = (float(item) for item in parts)
    return degrees + minutes / 60.0 + seconds / 3600.0


def image_gps(path: Path) -> tuple[float, float] | None:
    """读取 JPEG EXIF GPS；无坐标或坐标损坏时返回 None。"""

    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return None
            gps_tag = getattr(getattr(ExifTags, "IFD", object()), "GPSInfo", 34853)
            if hasattr(exif, "get_ifd"):
                gps = exif.get_ifd(gps_tag)
            else:
                gps = exif.get(gps_tag)
            if not gps:
                return None
            latitude = _dms_to_degrees(gps[2])
            longitude = _dms_to_degrees(gps[4])
            latitude_ref = str(gps.get(1, "N")).upper()
            longitude_ref = str(gps.get(3, "E")).upper()
            if latitude_ref.endswith("S"):
                latitude = -latitude
            if longitude_ref.endswith("W"):
                longitude = -longitude
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                return None
            return latitude, longitude
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError):
        return None


def _distance_metres(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = (radians(value) for value in first)
    lat2, lon2 = (radians(value) for value in second)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * asin(sqrt(value))


def filename_side(filename: str) -> str | None:
    stem = Path(filename).stem
    if "左右幅" in stem:
        return None
    if "左幅" in stem:
        return "左幅"
    if "右幅" in stem:
        return "右幅"
    return None


def nearest_bridge_name(
    gps: tuple[float, float] | None,
    filename: str,
) -> tuple[str, str, float | None]:
    """返回桥名、来源和最近距离；远距离不冒充已知桥梁。"""

    side = filename_side(filename)
    if gps is None:
        base, source, distance = "未知桥梁", "无EXIF坐标", None
    else:
        base, latitude, longitude = min(
            BRIDGE_REFERENCE_CENTRES,
            key=lambda item: _distance_metres(gps, (item[1], item[2])),
        )
        distance = _distance_metres(gps, (latitude, longitude))
        if distance > MAX_BRIDGE_DISTANCE_METRES:
            base, source = "未知桥梁", "超出公开坐标阈值"
        else:
            source = "公开坐标近邻"
    if side:
        base = f"{base}（{side}）"
    return base, source, distance


def defect_location(category: str, filename: str, defect_type: str) -> str:
    if category == "桥梁":
        stem = Path(filename).stem
        stem = re.sub(r"^(?:左右幅|左幅|右幅)[_-]?", "", stem)
        if stem and not stem.upper().startswith("DJI_"):
            return stem.replace("_", "-")
        return "全桥" if defect_type == "完好" else "桥梁结构"

    if defect_type == "完好":
        return "未见病害部位"
    locations: list[str] = []
    for marker, location in (
        ("混凝土裂缝", "混凝土构件"),
        ("裂缝", "混凝土构件"),
        ("钢筋锈蚀", "钢筋外露部位"),
        ("钢结构锈蚀", "钢结构"),
        ("支座锈蚀", "支座"),
        ("渗水泛碱", "混凝土表面"),
        ("破损", "结构表面"),
    ):
        if marker in defect_type and location not in locations:
            locations.append(location)
    return "、".join(locations) if locations else "结构表面"


def defect_description(defect_type: str) -> str:
    return DESCRIPTION_BY_DEFECT.get(defect_type, f"检测到{defect_type}")


def rating_scale(defect_type: str) -> str:
    if defect_type == "完好":
        return "1"
    return "3" if any(separator in defect_type for separator in ("、", "，", ",", "/", "+")) else "2"


def complete_rows(rows: list[dict[str, str]], manifest: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    """仅填补空字段，保留模板或模型已经提供的非空值。"""

    manifest_by_filename = {item["filename"]: item for item in manifest}
    if {row.get("filename") for row in rows} != set(manifest_by_filename):
        raise RuntimeError("结果与清单的文件名集合不同。")

    before_empty = Counter()
    bridge_name_sources = Counter()
    bridge_distance_values: list[float] = []
    for row in rows:
        if list(row) != FIELDS:
            raise RuntimeError(f"字段名称或顺序不符：{list(row)}")
        item = manifest_by_filename[row["filename"]]
        category, defect_type = row["questionCategory"], row["defectType"]
        if category not in ("桥梁", "轨道") or _blank(defect_type):
            raise RuntimeError(f"类别或缺损类型无效：{row}")
        for field in TARGET_FIELDS:
            before_empty[field] += int(_blank(row.get(field)))

        if _blank(row.get("bridgeName")):
            if category == "轨道":
                row["bridgeName"] = "不适用"
            else:
                path = resolve_image_path(manifest_path, item)
                name, source, distance = nearest_bridge_name(image_gps(path), row["filename"])
                row["bridgeName"] = name
                bridge_name_sources[source] += 1
                if distance is not None:
                    bridge_distance_values.append(distance)
        if _blank(row.get("defectLocation")):
            row["defectLocation"] = defect_location(category, row["filename"], defect_type)
        if _blank(row.get("defectDescription")):
            row["defectDescription"] = defect_description(defect_type)
        if _blank(row.get("ratingScale(1-5)")):
            row["ratingScale(1-5)"] = rating_scale(defect_type)

    after_empty = {field: sum(_blank(row.get(field)) for row in rows) for field in TARGET_FIELDS}
    return {
        "此前空值": dict(before_empty),
        "补全后空值": after_empty,
        "桥名来源": dict(bridge_name_sources),
        "桥名最近距离最大值_米": round(max(bridge_distance_values), 1) if bridge_distance_values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("result.json 必须是 JSON 数组。")
    report = complete_rows(rows, manifest, manifest_path)
    atomic_json(args.output, rows)
    report.update({"状态": "RESULT_FIELDS_FILLED", "记录数": len(rows), "输出": str(args.output)})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
