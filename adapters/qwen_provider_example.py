"""官方本地Qwen服务适配模板。

将本文件复制为官方要求的模块名并实现 ``embed_items``。输入items中的每个元素
形如 ``{"image": "data:image/jpeg;base64,..."}``，返回顺序一致的二维浮点数组。
不要在这里连接非官方外部服务。
"""


def embed_items(items: list[dict], model: str):
    raise NotImplementedError("Replace with the final cloud's official Qwen SDK call")
