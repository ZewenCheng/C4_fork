"""本地 Qwen 文件检查、单图编码和合成图片冒烟入口。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

# 离线开关必须先于 transformers/huggingface_hub 的首次导入。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["QWEN_PROVIDER_MODULE"] = "qwen_local_provider"
os.environ.setdefault("QWEN_MODEL", "Qwen3-VL-Embedding-8B")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageDraw

from common import PACKAGE_ROOT, atomic_json, load_config, sha256_file, three_view
from qwen_adapter import OfficialQwenClient, image_variants

sys.path.insert(0, str(PACKAGE_ROOT / "adapters"))
import qwen_local_provider as provider


def make_fixtures(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    image = Image.new("RGB", (192, 160), (245, 240, 225))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 18, 82, 135), fill=(210, 30, 25))
    draw.ellipse((105, 35, 180, 115), fill=(20, 80, 220))
    path = directory / "合成几何图.png"
    image.save(path)
    paths.append(path)
    image = Image.new("RGB", (192, 160), (10, 35, 40))
    draw = ImageDraw.Draw(image)
    for index in range(6):
        draw.line((5, 10 + index * 24, 180, 28 + index * 20), fill=(240, 210, 15), width=7)
    path = directory / "合成条纹图.png"
    image.save(path)
    paths.append(path)
    return paths


def compare_official_reference(path: Path, actual: np.ndarray) -> dict:
    """只在验收时执行已审核、已校验的官方参考源码，复用同一权重避免双份显存。"""
    import torch

    model_dir = provider.Settings.from_environment().model_dir
    script = model_dir / "scripts/qwen3_vl_embedding.py"
    manifest = json.loads((model_dir / "LOCAL_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    record = next(record for record in manifest["files"] if record["path"] == "scripts/qwen3_vl_embedding.py")
    if sha256_file(script) != record["sha256"]:
        raise RuntimeError("官方参考源码摘要不匹配，拒绝执行。")
    # 固定版本官方脚本仍导入一个已注释装饰器；Transformers 5.2.0 已移除该符号。
    # 仅在导入期间补回恒等装饰器，不修改官方源码或平台包，并在导入后立即撤销。
    generic = importlib.import_module("transformers.utils.generic")
    compatibility_added = not hasattr(generic, "check_model_inputs")
    if compatibility_added:
        generic.check_model_inputs = lambda function: function
    spec = importlib.util.spec_from_file_location("qwen_official_verification_reference", script)
    reference_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_module
    try:
        spec.loader.exec_module(reference_module)
    finally:
        if compatibility_added:
            del generic.check_model_inputs
    reference = object.__new__(reference_module.Qwen3VLEmbedder)
    engine = provider._ENGINE
    reference.model = engine.model
    reference.processor = engine.processor
    for key, value in {
        "max_length": provider.MAX_LENGTH, "min_pixels": provider.MIN_PIXELS,
        "max_pixels": provider.MAX_PIXELS, "total_pixels": reference_module.MAX_TOTAL_PIXELS,
        "fps": 1, "num_frames": 64, "max_frames": 64, "default_instruction": provider.INSTRUCTION,
    }.items():
        setattr(reference, key, value)
    config = load_config()["qwen"]
    url = image_variants(path, int(config["max_side"]), int(config["jpeg_quality"]))[0]
    image = provider.decode_image({"image": url})
    expected_inputs = reference._preprocess_inputs([reference.format_model_input(image=image)])
    actual_inputs = engine.prepare([image])
    inputs_equal = set(expected_inputs) == set(actual_inputs) and all(
        torch.equal(expected_inputs[key].cpu(), actual_inputs[key].cpu()) for key in expected_inputs
    )
    if not inputs_equal:
        raise AssertionError("本地预处理与固定版本官方参考不一致。")
    print("正在与固定版本官方参考实现比较真实向量。", flush=True)
    expected = reference.process([{"image": image}]).float().cpu().numpy()[0]
    difference = float(np.max(np.abs(expected - actual)))
    if difference > 1e-5:
        raise AssertionError(f"与官方参考实现的最大绝对误差超过 1e-5：{difference}")
    return {
        "源码_SHA256": record["sha256"],
        "预处理逐元素一致": inputs_equal,
        "向量最大绝对误差": difference,
        "Transformers_5_2兼容处理": "仅为官方脚本中未使用的 check_model_inputs 导入临时提供恒等装饰器，导入后撤销；官方源码及平台包未修改。" if compatibility_added else "无需处理",
        "比较范围": "同权重、同设备分层、同软件环境的官方参考；不是赛事远程接口或历史特征对齐。",
    }


def smoke(output: Path) -> None:
    import torch

    started = time.monotonic()
    fixture_dir = Path(os.environ.get("QWEN_LOCAL_RUNTIME_DIR", str(PACKAGE_ROOT / ".runtime/qwen-local"))) / "smoke"
    paths = make_fixtures(fixture_dir)
    report = {
        "状态": "未完成", "验证时间_UTC": datetime.now(timezone.utc).isoformat(),
        "任务": "2026-09-03-local-qwen-embedding",
        "model_id": provider.MODEL_ID, "revision": provider.MODEL_REVISION,
        "数据": "两张本机合成几何图；未使用比赛数据。",
        "输入文件": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
        "运行版本": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils", "numpy", "Pillow")},
        "Python": sys.version,
        "比赛准确率": None,
        "赛事官方接口数值一致性": "未验证",
    }
    client = OfficialQwenClient(load_config()["qwen"])
    gpu_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        with patch("socket.socket.connect", side_effect=AssertionError("离线验收禁止网络连接")) as network, patch("socket.create_connection", side_effect=AssertionError("离线验收禁止网络连接")) as connections:
            report["模型文件"] = provider.preflight()
            values = client.embed_images(paths)
            assert values.shape == (2, 3, 4096), values.shape
            assert np.isfinite(values).all(), "输出存在非有限值"
            norms = np.linalg.norm(values, axis=2)
            assert np.max(np.abs(norms - 1)) < 0.01, "BF16 单位向量误差超限"
            merged = three_view(values[:, 0], values[:, 1], values[:, 2])
            np.testing.assert_allclose(np.linalg.norm(merged, axis=1), 1, atol=1e-6)
            image_similarity = float(merged[0] @ merged[1])
            assert image_similarity < 0.999, "不同图片返回了近乎相同的向量"
            repeat = client.embed_images(paths[:1])
            repeat_difference = float(np.max(np.abs(values[:1] - repeat)))
            assert repeat_difference <= 1e-5, "重复调用误差超限"
            report["官方参考对照"] = compare_official_reference(paths[0], values[0, 0])
            report["本地运行配置"] = provider.runtime_info()
            assert network.call_count + connections.call_count == 0, "离线过程中尝试访问网络"
            report["网络连接尝试次数"] = network.call_count + connections.call_count
        np.savez_compressed(fixture_dir / "合成图片特征.npz", views=values, aggregated=merged)
        report.update({"三视图形状": list(values.shape), "聚合形状": list(merged.shape), "输出类型": str(values.dtype), "范数范围": [float(norms.min()), float(norms.max())], "不同图片余弦相似度": image_similarity, "重复最大绝对误差": repeat_difference})
        report["峰值torch已分配显存_字节"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        report["状态"] = "通过"
    except Exception as exc:
        report["失败类别"] = type(exc).__name__
        report["失败说明"] = str(exc)
        raise
    finally:
        client.close()
        released = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        report["释放后torch已分配显存_字节"] = released
        report["模型引用已释放"] = not provider.runtime_info()["loaded"]
        report["耗时_秒"] = round(time.monotonic() - started, 2)
        if report["状态"] == "通过" and released > gpu_before + 16 * 2**20:
            report["状态"] = "未通过"
            report["失败说明"] = "关闭 provider 后仍有异常显存占用。"
        atomic_json(output, report)
    if report["状态"] != "通过":
        raise RuntimeError(report.get("失败说明", "本地验收未通过"))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check", help="验证本地模型文件，不加载模型")
    check.add_argument("--verify-hashes", action="store_true")
    test = sub.add_parser("smoke", help="使用合成图像完成真实离线验收")
    test.add_argument("--output", type=Path, default=PACKAGE_ROOT / ".runtime/qwen-local/Qwen真实前向核验.json")
    embed = sub.add_parser("embed", help="编码本地图片，保存三视图及聚合向量")
    embed.add_argument("images", nargs="+", type=Path)
    embed.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "check":
        print(json.dumps(provider.preflight(verify_hashes=args.verify_hashes), ensure_ascii=False, indent=2))
    elif args.action == "smoke":
        smoke(args.output)
    else:
        if args.output.exists():
            raise FileExistsError("输出文件已存在，未覆盖。")
        client = OfficialQwenClient(load_config()["qwen"])
        try:
            values = client.embed_images(args.images)
        finally:
            client.close()
        aggregated = three_view(values[:, 0], values[:, 1], values[:, 2])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, views=values, aggregated=aggregated, filenames=np.asarray([str(path) for path in args.images]))
        print(f"已保存真实三视图 {values.shape} 和聚合向量 {aggregated.shape}：{args.output}")


if __name__ == "__main__":
    main()
