"""云端训练与推理共用的路径、模型、数值和数据合同函数。"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELDS = [
    "questionCategory", "bridgeName", "defectLocation", "filename",
    "defectType", "defectDescription", "ratingScale(1-5)",
]


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or PACKAGE_ROOT / "config/deploy_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_package_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "models":
        external_root = os.environ.get("CQAIP_MODELS_DIR", "").strip()
        if external_root:
            relative = Path(*path.parts[1:])
            if ".." in relative.parts:
                raise ValueError("模型相对路径不能越界。")
            return Path(external_root).resolve() / relative
    return PACKAGE_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def l2(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Cannot L2-normalize a zero vector")
    return value / norm


def three_view(orig: np.ndarray, flip: np.ndarray, crop: np.ndarray) -> np.ndarray:
    return l2(l2(orig) + l2(flip) + l2(crop))


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("Manifest must be a JSON array or JSONL records")
    filenames = []
    for row in rows:
        for key in ("filename", "questionCategory", "image_path"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Manifest row missing string {key}: {row}")
        if row["questionCategory"] not in ("轨道", "桥梁"):
            raise ValueError(f"Invalid questionCategory: {row['questionCategory']}")
        filenames.append(row["filename"])
    if len(filenames) != len(set(filenames)):
        raise ValueError("Manifest filename values must be unique")
    return rows


def resolve_image_path(manifest_path: Path, row: dict[str, Any]) -> Path:
    image_path = Path(row["image_path"])
    path = image_path if image_path.is_absolute() else manifest_path.parent / image_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare_local_image(path: Path, config: dict[str, Any]) -> Image.Image:
    """复刻初赛rail_1024：最长边1024、Lanczos、JPEG质量92并重新解码。"""

    spec = config["local_image_preprocess"]
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = int(spec["max_side"]) / max(width, height)
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(spec["jpeg_quality"]))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def load_safetensor_backbone(
    model_name: str,
    checkpoint: Path,
    device: str,
    *,
    num_classes: int = 0,
    img_size: int | None = None,
):
    """离线创建timm模型，仅加载名称和形状匹配的SafeTensors参数。"""

    import timm
    from safetensors.torch import load_file

    kwargs: dict[str, Any] = {"pretrained": False, "num_classes": num_classes}
    if img_size is not None:
        kwargs["img_size"] = img_size
    model = timm.create_model(model_name, **kwargs)
    source = load_file(str(checkpoint), device="cpu")
    target = model.state_dict()
    compatible = {name: tensor for name, tensor in source.items() if name in target and target[name].shape == tensor.shape}
    loaded_parameters = sum(tensor.numel() for tensor in compatible.values())
    total_parameters = sum(tensor.numel() for name, tensor in target.items() if not name.startswith("head."))
    if loaded_parameters < int(total_parameters * 0.95):
        raise RuntimeError(
            f"Too few compatible parameters for {model_name}: {loaded_parameters}/{total_parameters}"
        )
    model.load_state_dict(compatible, strict=False)
    return model.to(device).eval()


def load_expq_model(config: dict[str, Any], device: str, delta_override: Path | None = None):
    """加载原始FP32 expQ全量权重，可选再覆盖云端继续训练产生的delta。"""

    import timm
    import torch

    spec = config["models"]["expq"]
    model = timm.create_model(
        spec["timm_name"],
        pretrained=False,
        num_classes=int(spec["num_classes"]),
        img_size=int(spec["img_size"]),
    )
    full_state = torch.load(
        resolve_package_path(spec["full_checkpoint"]), map_location="cpu", weights_only=True
    )
    model.load_state_dict(full_state, strict=True)
    if delta_override is not None:
        payload = torch.load(delta_override, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected expQ delta keys: {unexpected[:10]}")
        trainable_prefixes = tuple(
            [f"blocks.{index}." for index in payload.get("last_blocks", [23, 24, 25, 26])]
            + ["norm.", "head."]
        )
        missing_trainable = [name for name in missing if name.startswith(trainable_prefixes)]
        if missing_trainable:
            raise RuntimeError(f"Missing expQ trainable keys: {missing_trainable[:10]}")
    return model.to(device).eval()


def load_portable_lr(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = ("coef", "intercept", "class_names", "feature_dim")
    if any(key not in data for key in required):
        raise RuntimeError(f"Portable LR missing keys: {required}")
    return {key: data[key] for key in data.files}


def portable_lr_predict(model: dict[str, np.ndarray], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    if features.shape[1] != int(np.asarray(model["feature_dim"]).item()):
        raise RuntimeError(f"Classifier feature dimension mismatch: {features.shape}")
    probability = softmax(features @ coef.T + intercept)
    labels = np.asarray(model["class_names"]).astype(str)[np.argmax(probability, axis=1)]
    return labels, probability


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
