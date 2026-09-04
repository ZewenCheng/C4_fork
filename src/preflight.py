#!/usr/bin/env python3
"""决赛云端启动前的离线文件、依赖、设备和分类器合同检查。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from common import PACKAGE_ROOT, choose_device, load_config, resolve_package_path, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config/deploy_config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-qwen", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = resolve_package_path("models/MODEL_MANIFEST.json")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    checked = 0
    for record in records:
        path = resolve_package_path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Model artifact mismatch: {path}")
        checked += 1
    classifier = np.load(resolve_package_path(config["classifier"]["checkpoint"]), allow_pickle=False)
    if int(classifier["feature_dim"]) != int(config["feature_contract"]["classifier_dim"]):
        raise RuntimeError("Classifier/config feature dimension mismatch")
    qwen_ready = bool(
        os.environ.get(config["qwen"]["base_url_env"])
        or os.environ.get(config["qwen"]["provider_module_env"])
    )
    if args.require_qwen and not qwen_ready:
        raise RuntimeError("Official Qwen provider is not configured")
    local_qwen = None
    if os.environ.get(config["qwen"]["provider_module_env"], "").strip() == "qwen_local_provider":
        import importlib
        local_qwen = importlib.import_module("qwen_local_provider").preflight()
    print(json.dumps({
        "status": "CLOUD_PREFLIGHT_PASS",
        "device": choose_device(args.device),
        "model_files_checked": checked,
        "qwen_provider_configured": qwen_ready,
        "local_qwen_files": local_qwen,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
