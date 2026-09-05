#!/usr/bin/env python3
"""验证云端部署包全部文件的SHA-256。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checked = 0
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Package file mismatch: {relative}")
        checked += 1
    print(json.dumps({"status": "PACKAGE_MANIFEST_PASS", "files_checked": checked}, ensure_ascii=False))


if __name__ == "__main__":
    main()
