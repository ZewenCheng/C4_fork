#!/usr/bin/env python3
"""从复现参考包导出不依赖scikit-learn对象反序列化的expQ LR参数。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


LOCAL_TAGS = ("siglip2", "dinov2L", "eva02", "convnextv2")


def l2(value):
    value = np.asarray(value, dtype=np.float32)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def three_view(orig, flip, crop):
    return l2(l2(orig) + l2(flip) + l2(crop))


def stem_from_key(key):
    return Path(str(key).split("::")[-1]).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.reference_root / "Claude Science 精选归档_2026-08-23"
    feature_root = archive / "05_关键特征与模型产物"
    labels_path = archive / "11_训练数据集/赛题四/训练集/轨道数据.json"
    source_package = archive / "07_安全提交包/submit_expQ_85.34_当前最佳.tar.gz"
    official_orig_path = args.reference_root / "external_assets/official_embeddings_rail_768.npz"
    tta_path = feature_root / "embeddings_rail_tta.npz"
    local_test_path = feature_root / "feat_test_all_五编码器测试特征.npz"
    xtr_path = feature_root / "Xtr_concat_五编码器训练特征.npy"
    ft_path = feature_root / "expQ_最佳线上/ft_siglip2_final.npz"

    label_rows = json.loads(labels_path.read_text(encoding="utf-8"))
    label_map = {Path(row["filename"]).stem: row["defectType"] for row in label_rows}
    train_stems = np.asarray(sorted(label_map), dtype=str)
    class_names = np.asarray(sorted(set(label_map.values())), dtype=str)
    class_index = {name: index for index, name in enumerate(class_names)}
    y = np.asarray([class_index[label_map[stem]] for stem in train_stems], dtype=np.int64)

    original_npz = np.load(official_orig_path, allow_pickle=True)
    tta_npz = np.load(tta_path, allow_pickle=True)
    original = {str(key): np.asarray(value, dtype=np.float32) for key, value in zip(original_npz["keys"], original_npz["embs"])}
    flip, crop = {}, {}
    for key, value in zip(tta_npz["keys"], np.asarray(tta_npz["embs"], dtype=np.float32)):
        view, path = str(key).split("::", 1)
        (flip if view == "flip" else crop)[path] = value
    train_paths = sorted((path for path in original if path.startswith("train_rail/")), key=stem_from_key)
    test_paths = sorted((path for path in original if path.startswith("test_rail/")), key=stem_from_key)
    test_stems = np.asarray([stem_from_key(path) for path in test_paths], dtype=str)

    def official_matrix(paths):
        stacked = np.stack([l2(original[path]) + l2(flip[path]) + l2(crop[path]) for path in paths]) / np.float32(3.0)
        return l2(stacked)

    train_parts = [official_matrix(train_paths)]
    for tag in LOCAL_TAGS:
        data = np.load(feature_root / f"feat_{tag}_train.npz", allow_pickle=True)
        index = {str(stem): i for i, stem in enumerate(data["stems"])}
        order = [index[stem] for stem in train_stems]
        train_parts.append(three_view(data["orig"][order], data["flip"][order], data["crop"][order]))
    xtr = np.hstack(train_parts).astype(np.float32)
    if not np.array_equal(xtr, np.load(xtr_path, mmap_mode="r")):
        raise RuntimeError("Xtr reconstruction mismatch")

    local_test = np.load(local_test_path, allow_pickle=True)
    index = {str(stem): i for i, stem in enumerate(local_test["stems"])}
    order = [index[stem] for stem in test_stems]
    test_parts = [official_matrix(test_paths)]
    for tag in LOCAL_TAGS:
        test_parts.append(three_view(
            local_test[f"{tag}_orig"][order],
            local_test[f"{tag}_flip"][order],
            local_test[f"{tag}_crop"][order],
        ))
    xte = np.hstack(test_parts).astype(np.float32)
    ft = np.load(ft_path, allow_pickle=False)
    x_train = np.hstack([xtr, l2(ft["Ftr"])]).astype(np.float32)
    x_test = np.hstack([xte, l2(ft["Fte"])]).astype(np.float32)
    model = LogisticRegression(C=20.0, solver="lbfgs", tol=1e-4, max_iter=2000, random_state=0).fit(x_train, y)
    prediction = class_names[model.predict(x_test)]

    with tarfile.open(source_package, "r:gz") as archive_file:
        rows = json.load(archive_file.extractfile("result/result.json"))
    reference = {
        Path(row["filename"]).stem: row["defectType"]
        for row in rows if row["questionCategory"] == "轨道"
    }
    matched = int(np.sum(prediction == np.asarray([reference[stem] for stem in test_stems], dtype=str)))
    if matched != 300:
        raise RuntimeError(f"Exported classifier regression failed: {matched}/300")
    manual_logits = x_test.astype(np.float64) @ model.coef_.astype(np.float64).T + model.intercept_.astype(np.float64)
    manual_prediction = class_names[np.argmax(manual_logits, axis=1)]
    if not np.array_equal(manual_prediction, prediction):
        raise RuntimeError("Portable coefficient inference differs from scikit-learn prediction")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        coef=model.coef_.astype(np.float64),
        intercept=model.intercept_.astype(np.float64),
        sklearn_classes=model.classes_.astype(np.int64),
        class_names=class_names,
        feature_dim=np.asarray(x_train.shape[1], dtype=np.int64),
        c=np.asarray(20.0, dtype=np.float64),
        train_stems=train_stems,
        reference_test_stems=test_stems,
        reference_test_prediction=prediction,
    )
    os.replace(temporary, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {
        "status": "INITIAL_EXPQ_CLASSIFIER_EXPORT_PASS",
        "matched_reference": matched,
        "feature_dim": x_train.shape[1],
        "classes": len(class_names),
        "iterations": model.n_iter_.tolist(),
        "sha256": digest,
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_name("class_names.json").write_text(json.dumps(class_names.tolist(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
