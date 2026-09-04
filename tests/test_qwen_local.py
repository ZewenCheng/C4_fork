"""本地 provider 的输入、安全边界与既有三视图接口测试；不伪装模型效果。"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / "adapters"), str(PACKAGE / "src")]
from common import load_config, three_view
from qwen_adapter import OfficialQwenClient
import qwen_local_provider as provider


def image_item(fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 120, 200)).save(buffer, format=fmt)
    return {"image": f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")}


class LocalProviderTests(unittest.TestCase):
    def test_jpeg_and_png_decode(self):
        for fmt in ("JPEG", "PNG"):
            image = provider.decode_image(image_item(fmt))
            self.assertEqual(image.size, (64, 48))
            self.assertEqual(image.mode, "RGB")

    def test_network_and_files_are_rejected(self):
        for value in ("https://example.com/image.jpg", "http://localhost/a", "file:///tmp/a", "C:/secret.png", "oss://bucket/a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                provider.decode_image({"image": value})

    def test_invalid_content_has_no_null_fallback(self):
        for item in ({}, {"text": "hello"}, {"image": "data:image/jpeg;base64,not-base64"}, {"image": "data:image/png;base64,AAAA"}, {**image_item(), "instruction": "override"}):
            with self.subTest(item=list(item)), self.assertRaises(ValueError):
                provider.decode_image(item)

    def test_wrong_model_does_not_load(self):
        with patch.object(provider, "_LocalEngine") as engine:
            with self.assertRaises(ValueError):
                provider.embed_items([image_item()], "Qwen3-VL-Embedding-2B")
            engine.assert_not_called()

    def test_invalid_input_does_not_load(self):
        with patch.object(provider, "_LocalEngine") as engine:
            with self.assertRaises(ValueError):
                provider.embed_items([{"image": "https://example.com/a.jpg"}], provider.MODEL_ID)
            engine.assert_not_called()

    def test_empty_input_shape_without_loading(self):
        with patch.object(provider, "_LocalEngine") as engine:
            result = provider.embed_items([], provider.MODEL_ID)
            self.assertEqual(result.shape, (0, 4096))
            self.assertEqual(result.dtype, np.float32)
            engine.assert_not_called()

    def test_settings_validate_resource_limits(self):
        cases = ({"QWEN_LOCAL_BATCH_SIZE": "0"}, {"QWEN_LOCAL_GPU_GIB": "nan"}, {"QWEN_LOCAL_CPU_GIB": "-1"}, {"QWEN_LOCAL_DEVICE": "remote"})
        for case in cases:
            with self.subTest(case=case), patch.dict(os.environ, case), self.assertRaises(ValueError):
                provider.Settings.from_environment()

    def test_prompt_and_pixel_budget(self):
        data = provider.conversation(Image.new("RGB", (64, 64)))
        self.assertEqual(data[0]["content"][0]["text"], "Represent the user's input.")
        self.assertEqual(data[1]["content"][0]["min_pixels"], 4096)
        self.assertEqual(data[1]["content"][0]["max_pixels"], 1843200)

    def test_missing_local_model_is_not_ready(self):
        with patch.dict(os.environ, {"QWEN_LOCAL_MODEL_DIR": str(PACKAGE / "tests/nonexistent-local-model")}):
            with self.assertRaises(FileNotFoundError):
                provider.preflight()

    def test_runtime_info_without_accelerate_device_map(self):
        parameter = types.SimpleNamespace(device="cuda:0", dtype="torch.bfloat16")
        engine = object.__new__(provider._LocalEngine)
        engine.model = types.SimpleNamespace(parameters=lambda: iter([parameter]))
        engine.memory_budget = {"0": 80 * 2**30, "cpu": 8 * 2**30}
        engine.settings = types.SimpleNamespace(micro_batch=1)
        engine.loading_info = {"missing_keys": {"b", "a"}, "unexpected_keys": set()}
        info = engine.runtime_info()
        self.assertIsNone(info["device_map"])
        self.assertEqual(info["parameter_devices"], ["cuda:0"])
        self.assertIn("未生成", info["device_map说明"])
        self.assertEqual(info["missing_keys"], ["a", "b"])
        self.assertEqual(info["unexpected_keys"], [])


class ExistingInterfaceTests(unittest.TestCase):
    def test_three_view_order_and_batching(self):
        observed = []
        module = types.ModuleType("unit_only_qwen_provider")

        def fake_embed(items, model):
            observed.extend(item["image"] for item in items)
            result = np.ones((len(items), 4096), dtype=np.float32)
            for index, item in enumerate(items):
                result[index, 0] = int(item["image"])
            return result

        module.embed_items = fake_embed
        module.release_model = Mock()
        config = load_config()["qwen"]
        config["batch_images"] = 1
        env = {"QWEN_PROVIDER_MODULE": module.__name__, "QWEN_MODEL": provider.MODEL_ID}
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, env), patch("qwen_adapter.requests.post") as http:
            with patch("qwen_adapter.image_variants", side_effect=[["0", "1", "2"], ["3", "4", "5"]]):
                client = OfficialQwenClient(config)
                result = client.embed_images([Path("a.jpg"), Path("b.jpg")])
                client.close()
            self.assertEqual(result.shape, (2, 3, 4096))
            np.testing.assert_array_equal(result[:, :, 0], [[0, 1, 2], [3, 4, 5]])
            self.assertEqual(observed, [str(index) for index in range(6)])
            http.assert_not_called()
            module.release_model.assert_called_once()
        aggregated = three_view(result[:, 0], result[:, 1], result[:, 2])
        self.assertEqual(aggregated.shape, (2, 4096))
        np.testing.assert_allclose(np.linalg.norm(aggregated, axis=1), 1, atol=1e-6)

    def test_legacy_sdk_without_release_still_works(self):
        module = types.ModuleType("unit_only_legacy_provider")
        with patch.dict(sys.modules, {module.__name__: module}), patch.dict(os.environ, {"QWEN_PROVIDER_MODULE": module.__name__}):
            OfficialQwenClient(load_config()["qwen"]).close()

    def test_remote_client_does_not_import_local_provider_on_close(self):
        env = {"QWEN_PROVIDER_MODULE": "", "QWEN_BASE_URL": "http://127.0.0.1:9999"}
        with patch.dict(os.environ, env), patch("qwen_adapter.importlib.import_module") as importer:
            OfficialQwenClient(load_config()["qwen"]).close()
            importer.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
