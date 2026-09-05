#!/usr/bin/env python3
"""验证决赛 result.json 结构、文件覆盖和七字段完整性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import FIELDS, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    rows = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("result.json must be a JSON array")
    filenames = []
    for row in rows:
        if list(row) != FIELDS:
            raise RuntimeError(f"Field names/order mismatch: {list(row)}")
        if any(not isinstance(value, str) for value in row.values()):
            raise RuntimeError(f"Non-string result value: {row}")
        if row["questionCategory"] not in ("桥梁", "轨道"):
            raise RuntimeError(f"Invalid category: {row}")
        empty_fields = [key for key, value in row.items() if not value.strip()]
        if empty_fields:
            raise RuntimeError(f"Empty result fields {empty_fields}: {row['filename']}")
        if row["ratingScale(1-5)"] not in {"1", "2", "3", "4", "5"}:
            raise RuntimeError(f"Invalid ratingScale(1-5): {row}")
        filenames.append(row["filename"])
    expected = {row["filename"] for row in manifest}
    if len(filenames) != len(set(filenames)) or set(filenames) != expected:
        raise RuntimeError("Result filename coverage/uniqueness mismatch")
    print(json.dumps({
        "status": "RESULT_VALIDATION_PASS",
        "rows": len(rows),
        "bridge_rows": sum(row["questionCategory"] == "桥梁" for row in rows),
        "rail_rows": sum(row["questionCategory"] == "轨道" for row in rows),
        "empty_values": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
