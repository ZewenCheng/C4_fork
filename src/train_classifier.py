#!/usr/bin/env python3
"""使用云端训练集特征拟合可移植的22类expQ逻辑回归分类器。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import PACKAGE_ROOT, l2, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, default=PACKAGE_ROOT / "models/classifier/class_names.json")
    parser.add_argument("--c", type=float, default=20.0)
    args = parser.parse_args()

    class_names = np.asarray(json.loads(args.class_names.read_text(encoding="utf-8")), dtype=str)
    class_index = {name: index for index, name in enumerate(class_names)}
    manifest = load_manifest(args.manifest)
    label_map = {
        Path(row["filename"]).stem: row["defectType"]
        for row in manifest
        if row["questionCategory"] == "轨道" and isinstance(row.get("defectType"), str)
    }
    data = np.load(args.features, allow_pickle=False)
    stems = data["stems"].astype(str)
    if set(stems) != set(label_map):
        raise RuntimeError(f"Training feature/label stem mismatch: features={len(stems)} labels={len(label_map)}")
    unknown = sorted(set(label_map.values()) - set(class_names.tolist()))
    if unknown:
        raise RuntimeError(f"Cloud labels contain combinations outside the locked 22 classes: {unknown}")
    y = np.asarray([class_index[label_map[stem]] for stem in stems], dtype=np.int64)
    features = np.hstack([data["xbase"].astype(np.float32), l2(data["expq_ft"])]).astype(np.float32)
    model = LogisticRegression(C=args.c, solver="lbfgs", tol=1e-4, max_iter=2000, random_state=0).fit(features, y)
    coef = model.coef_.astype(np.float64)
    intercept = model.intercept_.astype(np.float64)
    # scikit-learn二分类只保存一个对数几率行；扩展成[0, z]后可继续使用统一softmax推理。
    if len(model.classes_) == 2 and coef.shape[0] == 1:
        coef = np.vstack([np.zeros_like(coef), coef])
        intercept = np.concatenate([np.zeros_like(intercept), intercept])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        coef=coef,
        intercept=intercept,
        sklearn_classes=model.classes_.astype(np.int64),
        class_names=class_names[model.classes_],
        feature_dim=np.asarray(features.shape[1], dtype=np.int64),
        c=np.asarray(args.c, dtype=np.float64),
        train_stems=stems,
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "CLASSIFIER_TRAINING_PASS",
        "rows": len(stems),
        "feature_dim": features.shape[1],
        "classes": len(model.classes_),
        "iterations": model.n_iter_.tolist(),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
