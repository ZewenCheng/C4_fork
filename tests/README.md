# 离线回归测试

`mock_official_qwen_provider.py`只用于使用封存Embedding验证图片预处理和本地骨干特征抽取，不是Qwen-free模型，也不能用于决赛正式预测。

正式环境应取消：

```bash
unset QWEN_PROVIDER_MODULE
unset QWEN_FIXTURE_NPZ
```

并按官方接口设置 `QWEN_BASE_URL`、`QWEN_MODEL` 和必要时的 `QWEN_API_KEY`。
