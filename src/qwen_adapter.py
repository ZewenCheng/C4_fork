"""决赛官方Qwen3-VL Embedding适配层。

默认调用OpenAI兼容的官方 ``/embeddings`` 接口，但不提供任何外部默认URL，
避免在决赛环境中把图片误发到非官方服务。通过 ``QWEN_PROVIDER_MODULE`` 可指定
官方 SDK 或用户授权的本地原始权重实现；本地权重不会将图片发送至外部服务。
"""

from __future__ import annotations

import base64
import importlib
import io
import os
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def image_variants(path: Path, max_side: int, quality: int) -> list[str]:
    """生成与初赛一致的Qwen原图、翻转、中心80%裁剪JPEG data URL。"""

    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    width, height = image.size
    crop = image.crop((int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9)))
    result = []
    for variant in (image, flip, crop):
        buffer = io.BytesIO()
        variant.save(buffer, "JPEG", quality=quality)
        result.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return result


class OfficialQwenClient:
    def __init__(self, config: dict):
        self.config = config
        self.model = os.environ.get(config["model_env"], config["default_model"])
        self.provider_module = os.environ.get(config["provider_module_env"], "").strip()
        self.base_url = os.environ.get(config["base_url_env"], "").strip().rstrip("/")
        self.api_key = os.environ.get(config["api_key_env"], "").strip()
        if not self.provider_module and not self.base_url:
            raise RuntimeError(
                f"Set {config['base_url_env']} for the official endpoint, or "
                f"{config['provider_module_env']} for the official local provider module"
            )

    def _embed_items(self, items: list[dict]) -> np.ndarray:
        if self.provider_module:
            module = importlib.import_module(self.provider_module)
            values = module.embed_items(items=items, model=self.model)
            return np.asarray(values, dtype=np.float32)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": items}
        retries = int(self.config["retries"])
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=int(self.config["timeout_seconds"]),
                )
                if response.status_code == 200:
                    data = response.json()["data"]
                    return np.asarray([item["embedding"] for item in data], dtype=np.float32)
                message = f"Qwen HTTP {response.status_code}: {response.text[:300]}"
            except Exception as exc:  # 网络错误只能在官方云内重试，不改变输入或模型
                message = repr(exc)
            if attempt + 1 < retries:
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"Official Qwen embedding failed after {retries} attempts: {message}")

    def close(self) -> None:
        """可选本地 provider 生命周期；远程接口和旧 SDK 保持原行为。"""
        if self.provider_module:
            module = importlib.import_module(self.provider_module)
            release = getattr(module, "release_model", None)
            if callable(release):
                release()

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        """返回N×3×D矩阵，第二维顺序固定为orig/flip/crop。"""

        all_values = []
        batch_images = int(self.config["batch_images"])
        for offset in range(0, len(paths), batch_images):
            batch = paths[offset:offset + batch_images]
            items = []
            for path in batch:
                items.extend(
                    {"image": value}
                    for value in image_variants(path, int(self.config["max_side"]), int(self.config["jpeg_quality"]))
                )
            embedded = self._embed_items(items)
            if embedded.shape[0] != len(batch) * 3:
                raise RuntimeError(f"Qwen returned unexpected rows: {embedded.shape}")
            all_values.append(embedded.reshape(len(batch), 3, -1))
        return np.concatenate(all_values, axis=0)
