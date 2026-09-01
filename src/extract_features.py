#!/usr/bin/env python3
"""从决赛云端新图片提取Qwen、四骨干和expQ微调特征。"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import timm
import torchvision.transforms as T
from PIL import Image

from common import (
    PACKAGE_ROOT,
    choose_device,
    l2,
    load_config,
    load_expq_model,
    load_manifest,
    load_safetensor_backbone,
    prepare_local_image,
    resolve_image_path,
    resolve_package_path,
    three_view,
)
from qwen_adapter import OfficialQwenClient


def local_views(image: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return (
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
        image.crop((left, top, left + side, top + side)),
    )


def extract_three_view_model(model, paths: list[Path], device: str, batch_images: int, config: dict) -> np.ndarray:
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    rows = []
    with torch.no_grad():
        for offset in range(0, len(paths), batch_images):
            batch_paths = paths[offset:offset + batch_images]
            tensors = []
            for path in batch_paths:
                image = prepare_local_image(path, config)
                tensors.extend(transform(view) for view in local_views(image))
            values = model(torch.stack(tensors).to(device)).float().cpu().numpy()
            values = values.reshape(len(batch_paths), 3, -1)
            rows.append(three_view(values[:, 0], values[:, 1], values[:, 2]))
    return np.concatenate(rows, axis=0).astype(np.float32)


def extract_transformers_three_view(model_dir: Path, paths: list[Path], device: str, batch_images: int, config: dict) -> np.ndarray:
    """严格复刻初赛extract_multi.py的Transformers特征提取路径。"""

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    wrapper = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model = getattr(wrapper, "vision_model", wrapper).to(device).eval()
    rows = []
    with torch.no_grad():
        for offset in range(0, len(paths), batch_images):
            batch_paths = paths[offset:offset + batch_images]
            images = []
            for path in batch_paths:
                images.extend(local_views(prepare_local_image(path, config)))
            inputs = processor(images=images, return_tensors="pt").to(device)
            output = model(**inputs)
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state.mean(1)
            values = features.float().cpu().numpy().reshape(len(batch_paths), 3, -1)
            rows.append(three_view(values[:, 0], values[:, 1], values[:, 2]))
    del model, wrapper, processor
    gc.collect()
    return np.concatenate(rows, axis=0).astype(np.float32)


def extract_expq(model, paths: list[Path], device: str, batch_images: int, config: dict) -> np.ndarray:
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = T.Compose([
        T.Resize((378, 378)),
        T.ToTensor(),
        T.Normalize(data_config["mean"], data_config["std"]),
    ])
    rows = []
    with torch.no_grad():
        for offset in range(0, len(paths), batch_images):
            batch_paths = paths[offset:offset + batch_images]
            images = torch.stack([transform(prepare_local_image(path, config)) for path in batch_paths]).to(device)
            features = model.forward_head(model.forward_features(images), pre_logits=True)
            rows.append(features.float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def release_model(model, device: str) -> None:
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config/deploy_config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-images", type=int, default=4)
    parser.add_argument("--expq-delta", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(args.device)
    manifest = load_manifest(args.manifest)
    rail_rows = [row for row in manifest if row["questionCategory"] == "轨道"]
    if not rail_rows:
        raise RuntimeError("Manifest has no rail images")
    paths = [resolve_image_path(args.manifest, row) for row in rail_rows]
    stems = np.asarray([Path(row["filename"]).stem for row in rail_rows], dtype=str)

    qwen_client = OfficialQwenClient(config["qwen"])
    qwen_views = qwen_client.embed_images(paths)
    if qwen_views.shape[2] != int(config["feature_contract"]["qwen_dim"]):
        raise RuntimeError(f"Unexpected Qwen dimension: {qwen_views.shape}")
    qwen = three_view(qwen_views[:, 0], qwen_views[:, 1], qwen_views[:, 2])

    frozen_parts = []
    for name in ("siglip2", "dinov2", "eva02", "convnextv2"):
        spec = config["models"][name]
        if spec.get("provider") == "transformers":
            feature = extract_transformers_three_view(
                resolve_package_path(spec["model_dir"]), paths, device, args.batch_images, config
            )
            model = None
        else:
            model = load_safetensor_backbone(
                spec["timm_name"],
                resolve_package_path(spec["checkpoint"]),
                device,
                num_classes=0,
            )
            feature = extract_three_view_model(model, paths, device, args.batch_images, config)
        if feature.shape[1] != int(spec["expected_dim"]):
            raise RuntimeError(f"Unexpected {name} feature shape: {feature.shape}")
        frozen_parts.append(feature)
        if model is not None:
            release_model(model, device)

    xbase = np.hstack([qwen, *frozen_parts]).astype(np.float32)
    if xbase.shape[1] != int(config["feature_contract"]["xbase_dim"]):
        raise RuntimeError(f"Unexpected Xbase shape: {xbase.shape}")

    expq = load_expq_model(config, device, args.expq_delta)
    ft = extract_expq(expq, paths, device, args.batch_images, config)
    release_model(expq, device)
    if ft.shape[1] != int(config["feature_contract"]["expq_ft_dim"]):
        raise RuntimeError(f"Unexpected expQ feature shape: {ft.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        stems=stems,
        filenames=np.asarray([row["filename"] for row in rail_rows], dtype=str),
        xbase=xbase,
        expq_ft=ft,
        qwen_model=np.asarray(qwen_client.model),
        feature_contract=np.asarray(json.dumps(config["feature_contract"], ensure_ascii=False)),
    )
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "FEATURE_EXTRACTION_PASS",
        "device": device,
        "rail_rows": len(rail_rows),
        "xbase_shape": list(xbase.shape),
        "expq_ft_shape": list(ft.shape),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
