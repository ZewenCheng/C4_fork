#!/usr/bin/env python3
"""把云端测试特征转换为七字段result.json。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import FIELDS, PACKAGE_ROOT, l2, load_manifest, load_portable_lr, portable_lr_predict
from result_fields import complete_rows


def validate_rows(rows: list[dict]) -> None:
    filenames = []
    for row in rows:
        if list(row) != FIELDS or any(not isinstance(value, str) for value in row.values()):
            raise RuntimeError(f"Invalid seven-field row: {row}")
        if row["questionCategory"] not in ("轨道", "桥梁"):
            raise RuntimeError(f"Invalid category: {row}")
        filenames.append(row["filename"])
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("Duplicate filename in result")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, default=PACKAGE_ROOT / "models/classifier/expq_lr.npz")
    parser.add_argument("--template-json", type=Path, help="官方样例结果；提供时保留桥梁及非预测字段")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    feature_data = np.load(args.features, allow_pickle=False)
    features = np.hstack([feature_data["xbase"].astype(np.float32), l2(feature_data["expq_ft"])]).astype(np.float32)
    classifier = load_portable_lr(args.classifier)
    labels, probability = portable_lr_predict(classifier, features)
    prediction = dict(zip(feature_data["filenames"].astype(str).tolist(), labels.tolist()))

    if args.template_json:
        rows = json.loads(args.template_json.read_text(encoding="utf-8"))
        if {row["filename"] for row in rows} != {row["filename"] for row in manifest}:
            raise RuntimeError("Template and manifest filename sets differ")
        output_rows = []
        for row in rows:
            new = dict(row)
            if row["questionCategory"] == "轨道":
                new["defectType"] = prediction[row["filename"]]
            output_rows.append(new)
    else:
        output_rows = []
        for item in manifest:
            category = item["questionCategory"]
            output_rows.append({
                "questionCategory": category,
                "bridgeName": item.get("bridgeName", "") if category == "桥梁" else "",
                "defectLocation": "",
                "filename": item["filename"],
                "defectType": prediction[item["filename"]] if category == "轨道" else "完好",
                "defectDescription": "",
                "ratingScale(1-5)": "",
            })

    completion = complete_rows(output_rows, manifest, args.manifest.resolve())
    validate_rows(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".probabilities.npz"),
        filenames=feature_data["filenames"].astype(str),
        class_names=np.asarray(classifier["class_names"]).astype(str),
        probability=probability,
        prediction=labels,
    )
    print(json.dumps({
        "status": "CLOUD_INFERENCE_PASS",
        "rows": len(output_rows),
        "rail_rows": len(prediction),
        "output": str(args.output),
        "field_completion": completion,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
