"""仅用于离线回归测试的官方Qwen接口模拟器。

通过 ``QWEN_FIXTURE_NPZ`` 指定预先封存的data URL哈希到Embedding映射。
决赛正式运行不得设置本模块，应连接官方Qwen服务。
"""

from __future__ import annotations

import hashlib
import os

import numpy as np


def embed_items(items: list[dict], model: str):
    fixture_path = os.environ["QWEN_FIXTURE_NPZ"]
    data = np.load(fixture_path, allow_pickle=False)
    mapping = {key: value for key, value in zip(data["keys"].astype(str), data["embeddings"])}
    result = []
    for item in items:
        key = hashlib.sha256(item["image"].encode("utf-8")).hexdigest()
        if key not in mapping:
            raise KeyError(f"Qwen fixture has no payload hash {key}")
        result.append(mapping[key])
    return np.asarray(result, dtype=np.float32)
