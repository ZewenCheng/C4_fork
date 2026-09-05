"""官方 Qwen3-VL-Embedding-8B 的离线图像 provider，不启动网络服务。

输入沿用 C4 的 JPEG/PNG data URL，输出按原顺序为 N×4096 float32。
模型封装、默认提示和最后有效 token 池化参考 Qwen 官方 Apache-2.0 实现：
https://github.com/QwenLM/Qwen3-VL-Embedding
本适配不吞掉图像错误、不退化为 NULL 文本，也不执行下载模型内的 Python。
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"
MODEL_REVISION = "2c4565515e0f265c6511776e7193b22c0968ddc7"
DIMENSION = 4096
INSTRUCTION = "Represent the user's input."
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 1800 * 32 * 32
MAX_LENGTH = 8192
_LOCK = threading.RLock()
_ENGINE = None


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    offload_dir: Path
    device: str
    gpu_gib: float
    cpu_gib: float
    micro_batch: int

    @classmethod
    def from_environment(cls) -> "Settings":
        value = cls(
            model_dir=Path(os.environ.get("QWEN_LOCAL_MODEL_DIR", str(PROJECT_ROOT / "models/Qwen3-VL-Embedding-8B"))).resolve(),
            offload_dir=Path(os.environ.get("QWEN_LOCAL_OFFLOAD_DIR", str(PROJECT_ROOT / "tmp/qwen-local/offload"))).resolve(),
            device=os.environ.get("QWEN_LOCAL_DEVICE", "auto"),
            gpu_gib=float(os.environ.get("QWEN_LOCAL_GPU_GIB", "5")),
            cpu_gib=float(os.environ.get("QWEN_LOCAL_CPU_GIB", "10")),
            micro_batch=int(os.environ.get("QWEN_LOCAL_BATCH_SIZE", "1")),
        )
        if value.device not in ("auto", "cpu"):
            raise ValueError("QWEN_LOCAL_DEVICE 只能为 auto 或 cpu。")
        if not np.isfinite([value.gpu_gib, value.cpu_gib]).all() or min(value.gpu_gib, value.cpu_gib) <= 0:
            raise ValueError("本地内存预算必须为有限正数。")
        if value.micro_batch < 1:
            raise ValueError("QWEN_LOCAL_BATCH_SIZE 必须为正整数。")
        return value


def preflight(*, verify_hashes: bool = False) -> dict:
    """文件核验不等同于前向成功；真实前向由 smoke 入口单独证明。"""
    settings = Settings.from_environment()
    marker = settings.model_dir / "LOCAL_MODEL_MANIFEST.json"
    if not marker.is_file():
        raise FileNotFoundError("未找到已核验模型清单，请先完成 tools/download_qwen_model.py。")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("模型名称或固定版本与本适配不匹配。")
    records = manifest.get("files", [])
    names = {record["path"] for record in records}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "model.safetensors.index.json"}
    required.update(f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5))
    if not required.issubset(names) or len(names) != len(records):
        raise ValueError("模型清单缺少必要文件或包含重复条目。")
    for record in records:
        path = (settings.model_dir / record["path"]).resolve()
        if not path.is_relative_to(settings.model_dir):
            raise ValueError("模型清单包含越界路径。")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"模型文件缺失或大小不符：{record['path']}")
        if verify_hashes:
            with path.open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != record["sha256"]:
                raise ValueError(f"模型文件 SHA-256 不符：{record['path']}")
    config = json.loads((settings.model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or config.get("text_config", {}).get("hidden_size") != DIMENSION:
        raise ValueError("模型架构或特征维数不匹配。")
    if config["text_config"].get("num_hidden_layers") != 36:
        raise ValueError("当前配置不是所需的 8B 模型。")
    return {
        "类型": "本地官方原始权重",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_dir": str(settings.model_dir),
        "files_checked": len(records),
        "sha256_checked": verify_hashes,
        "dimension": DIMENSION,
        "前向运行已验证": False,
    }


def decode_image(item: dict) -> Image.Image:
    if not isinstance(item, dict) or set(item) != {"image"}:
        raise ValueError("本地 C4 provider 的每项输入必须且只能包含 image。")
    value = item["image"]
    if not isinstance(value, str) or not value.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
        raise ValueError("仅接受 JPEG/PNG base64 data URL；不访问网络 URL 或任意文件路径。")
    encoded = value.split(",", 1)[1]
    if not encoded or len(encoded) > 16 * 1024 * 1024:
        raise ValueError("图片内容为空或超过 16 MiB 编码限制。")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ("JPEG", "PNG") or source.width * source.height > 16_000_000:
                raise ValueError("图片格式或像素数不符合本地输入限制。")
            source.load()
            return source.convert("RGB")
    except (binascii.Error, OSError) as exc:
        raise ValueError("图片 base64 或图像内容无效。") from exc


def conversation(image: Image.Image) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": [{"type": "image", "image": image, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}]},
    ]


class _LocalEngine:
    def __init__(self, settings: Settings):
        import psutil
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        preflight()
        self.settings = settings
        self.torch = torch
        available = psutil.virtual_memory().available
        cpu_budget = min(int(settings.cpu_gib * 2**30), available - 2 * 2**30)
        if cpu_budget < 2 * 2**30:
            raise RuntimeError("可用内存不足，未开始加载；请释放至少数 GiB 内存后重试。")
        use_cuda = settings.device == "auto" and torch.cuda.is_available()
        self.input_device = torch.device("cuda:0" if use_cuda else "cpu")
        memory = {"cpu": cpu_budget}
        if use_cuda:
            free, _ = torch.cuda.mem_get_info(0)
            gpu_budget = min(int(settings.gpu_gib * 2**30), free - int(1.5 * 2**30))
            if gpu_budget < 1 * 2**30:
                raise RuntimeError("可用显存不足以安全运行，未挤占其他进程。")
            memory[0] = gpu_budget

        class QwenEmbeddingBackbone(Qwen3VLPreTrainedModel):
            # 与官方无生成头封装相同的参数层级，不创建随机 lm_head。
            _checkpoint_conversion_mapping = {}
            accepts_loss_kwargs = False

            def __init__(self, config):
                super().__init__(config)
                self.model = Qwen3VLModel(config)
                self.post_init()

            def get_input_embeddings(self):
                return self.model.get_input_embeddings()

            def forward(self, **inputs):
                return self.model(**inputs)

        settings.offload_dir.mkdir(parents=True, exist_ok=True)
        self.processor = Qwen3VLProcessor.from_pretrained(
            str(settings.model_dir), local_files_only=True, trust_remote_code=False, padding_side="right",
        )
        print("正在加载本地 Qwen 原始 BF16 权重；启用显存／内存／磁盘分层加载。", flush=True)
        self.model, self.loading_info = QwenEmbeddingBackbone.from_pretrained(
            str(settings.model_dir), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
            max_memory=memory, offload_folder=str(settings.offload_dir),
            offload_state_dict=True, output_loading_info=True,
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if self.loading_info.get(key):
                raise RuntimeError(f"权重加载并非严格匹配：{key}={self.loading_info[key][:5]}")
        # 不能再调用 model.to(cuda)：这会破坏 accelerate 的分层加载。
        self.model.eval()
        self.model.config.use_cache = False
        self.memory_budget = {str(key): value for key, value in memory.items()}

    def prepare(self, images: list[Image.Image]):
        from qwen_vl_utils.vision_process import process_vision_info

        conversations = [conversation(image) for image in images]
        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images_in, videos, kwargs = process_vision_info(
            conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True,
        )
        if videos is not None:
            raise ValueError("图像 provider 不应产生视频输入。")
        inputs = self.processor(
            text=text, images=images_in, videos=None, video_metadata=None,
            truncation=False, padding=True, do_resize=False, return_tensors="pt", **kwargs,
        )
        if inputs["input_ids"].shape[1] > MAX_LENGTH:
            raise ValueError("图像序列超长；拒绝截断视觉 token 或返回替代向量。")
        return {name: value.to(self.input_device) for name, value in inputs.items()}

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        results = []
        for offset in range(0, len(images), self.settings.micro_batch):
            batch = images[offset:offset + self.settings.micro_batch]
            print(f"本地特征前向：{offset + 1}–{offset + len(batch)}/{len(images)}", flush=True)
            inputs = self.prepare(batch)
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
                hidden = output.last_hidden_state
                mask = inputs["attention_mask"].to(hidden.device)
                last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
                values = torch.nn.functional.normalize(pooled, p=2, dim=-1).float().cpu().numpy()
            if values.shape != (len(batch), DIMENSION) or not np.isfinite(values).all():
                raise RuntimeError("本地模型输出形状错误或包含非有限值。")
            if np.any(np.linalg.norm(values, axis=1) < 1e-6):
                raise RuntimeError("本地模型返回零向量。")
            results.append(values)
            del inputs, output, hidden, mask, pooled
        return np.concatenate(results, axis=0)

    def runtime_info(self) -> dict:
        raw_device_map = getattr(self.model, "hf_device_map", None)
        device_map = (
            {key: str(value) for key, value in raw_device_map.items()}
            if isinstance(raw_device_map, dict) else None
        )
        parameter_devices = sorted({str(parameter.device) for parameter in self.model.parameters()})
        missing_keys = sorted(str(key) for key in (self.loading_info.get("missing_keys") or []))
        unexpected_keys = sorted(str(key) for key in (self.loading_info.get("unexpected_keys") or []))
        return {
            "device_map": device_map,
            "parameter_devices": parameter_devices,
            "device_map说明": "分层加载映射" if device_map is not None else "框架未生成 hf_device_map；记录实际参数设备",
            "memory_budget_bytes": self.memory_budget,
            "dtype": str(next(self.model.parameters()).dtype),
            "micro_batch": self.settings.micro_batch,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }


def embed_items(items: list[dict], model: str) -> np.ndarray:
    """直接接入既有 QWEN_PROVIDER_MODULE，不需要 API Key。"""
    global _ENGINE
    if model not in (MODEL_ID, MODEL_ID.split("/", 1)[1]):
        raise ValueError("本地适配仅允许固定的 Qwen3-VL-Embedding-8B。")
    if not isinstance(items, list):
        raise TypeError("items 必须是列表。")
    images = [decode_image(item) for item in items]
    if not images:
        return np.empty((0, DIMENSION), dtype=np.float32)
    settings = Settings.from_environment()
    with _LOCK:
        if _ENGINE is None:
            _ENGINE = _LocalEngine(settings)
        elif _ENGINE.settings != settings:
            raise RuntimeError("模型已加载时不能改变本地设置；请先调用 release_model。")
        return _ENGINE.encode(images)


def runtime_info() -> dict:
    with _LOCK:
        return {"loaded": _ENGINE is not None, **(_ENGINE.runtime_info() if _ENGINE else {})}


def release_model() -> None:
    """在其他视觉骨干开始前释放模型和分层加载钩子持有的显存。"""
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            return
        torch = _ENGINE.torch
        _ENGINE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
