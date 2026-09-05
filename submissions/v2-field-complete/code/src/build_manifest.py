#!/usr/bin/env python3
"""把决赛云端数据目录转换成与机器无关的JSONL清单。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import IMAGE_SUFFIXES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="官方元数据数组；至少包含filename、questionCategory，可包含bridgeName和defectType",
    )
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    image_map: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.name in image_map:
                raise RuntimeError(f"Duplicate image filename in dataset: {path.name}")
            image_map[path.name] = path

    rows = []
    if args.metadata_json:
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        for item in metadata:
            filename = item["filename"]
            path = image_map.get(filename)
            if path is None:
                raise FileNotFoundError(f"Metadata image missing: {filename}")
            row = {
                "filename": filename,
                "questionCategory": item["questionCategory"],
                "bridgeName": item.get("bridgeName", ""),
                "image_path": os.path.relpath(path, args.output.parent.resolve()),
            }
            if "defectType" in item:
                row["defectType"] = item["defectType"]
            rows.append(row)
    else:
        for filename, path in sorted(image_map.items()):
            folder = path.parent.name
            category = "轨道" if folder == "轨道" else "桥梁"
            rows.append({
                "filename": filename,
                "questionCategory": category,
                "bridgeName": "" if category == "轨道" else folder,
                "image_path": os.path.relpath(path, args.output.parent.resolve()),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"status": "MANIFEST_CREATED", "rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
