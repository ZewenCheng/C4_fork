# Qwen官方接口适配

初赛实际使用三个Qwen视图：

1. 最长边缩放至不超过768的原图；
2. 水平翻转；
3. 在缩放后图像上裁剪中心80%区域。

三个视图均编码为JPEG质量85，分别获取4,096维Embedding，然后执行：

```text
L2(L2(orig) + L2(flip) + L2(crop))
```

`src/qwen_adapter.py`已经固定上述预处理。正式环境只需要适配官方调用方式，不应改变视图、JPEG质量或Embedding顺序。

OpenAI兼容接口的请求体：

```json
{
  "model": "Qwen3-VL-Embedding-8B",
  "input": [
    {"image": "data:image/jpeg;base64,..."}
  ]
}
```

若官方不是该协议，应通过 `QWEN_PROVIDER_MODULE` 适配，不要修改特征提取主程序。适配函数必须保证输入与返回顺序一致。
