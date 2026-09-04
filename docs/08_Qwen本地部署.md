# Qwen3-VL-Embedding-8B 自托管适配

## 当前状态

当前已完成 provider、三视图接口、释放钩子、离线验收入口和 13 项接口测试。指定比赛 PPU 上的固定权重、项目内依赖、真实前向、官方固定源码对照和 C4 严格预检均已通过。

实际运行目标为既有官方 PPU 实例，本机保存源码和缓存；Windows CUDA 环境未完成。Chrome 和 SSH 已恢复，11 个初始适配文件及运行中发现的兼容修复均在云端核对前版本后受控部署，原源码已备份。21 个固定版本模型文件共 16,305,674,095 字节通过完整摘要检查；`qwen-vl-utils==0.0.14` 仅安装到项目 `.runtime/qwen_deps`，没有升级或覆盖平台 PPU/PyTorch。

本机 Windows 独立缓存的第 1、2、4 分片已完成官方摘要校验，第 3 分片仅有 272,629,760 字节断点；下载进程已结束，全部缓存保留。本机缓存没有完整模型清单，本机预检会按预期拒绝运行；这不影响已经完成的云端 PPU 部署。可复查结果见 [适配与缓存核验](../../.cairn/evidence/Qwen适配与缓存核验.json)。

上述断点状态仅指本机缓存。云端另外使用 ModelScope 官方 Qwen 源下载，并对照同一固定提交的 HuggingFace 摘要；模型清单、依赖和真实前向回执已只读复核。证据见 [PPU 真实前向核验](../../.cairn/evidence/QwenPPU真实前向核验.json)。

这一步是推理部署，不需要训练。用户已经允许使用 PPU 训练，但后续训练仍需独立固定数据、切分、基线、指标和预算。本次不训练、不使用官方测试集、不发起评测。

## 模型和实现

- 官方模型：`Qwen/Qwen3-VL-Embedding-8B`，Apache-2.0。
- 固定版本：`2c4565515e0f265c6511776e7193b22c0968ddc7`。
- 四个权重分片共约 16.29 GB；加处理器等共约 16.31 GB。
- 本机缓存：项目根目录的 `models/Qwen3-VL-Embedding-8B`。
- PPU 计划目录：`/workspace/work/road-infrastructure-finals-cloud/models/transformers/qwen3_vl_embedding`。
- 新模型不覆盖原来的 `MODEL_MANIFEST.json` 或分类器；使用独立 `LOCAL_MODEL_MANIFEST.json`。
- 图像处理和池化参考固定版本附带的 `scripts/qwen3_vl_embedding.py`；provider 本身不执行下载模型中的 Python，也不使用 `trust_remote_code=True`。

来源：[官方模型](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)、[官方实现](https://github.com/QwenLM/Qwen3-VL-Embedding)。

## 接口不变项

现有 `OfficialQwenClient` → `QWEN_PROVIDER_MODULE=qwen_local_provider` → `embed_items(items, model)`。

每项输入仅为 `{"image": "data:image/jpeg;base64,..."}`，兼容 PNG data URL。拒绝外部 URL、任意文件路径、错误模型、非法图像、额外字段及超长图像序列；不会退化为 NULL 文本或模拟向量。

原图、水平翻转、中心 80% 裁剪与 JPEG 设置仍由既有 `image_variants` 生成。输出为 `N×3×4096`，按原方法聚合成 `N×4096`。原分类器、其他视觉分支及预测公式未修改。`extract_features.py` 使用 `finally` 调用 `close()`，让其他视觉模型运行前释放 Qwen 显存；没有释放钩子的旧 SDK 和远程模式继续兼容。

模型保持 BF16 原始权重；默认官方提示为 `Represent the user's input.`，使用最后有效 token 和官方 BF16 L2 归一化，再转 float32 传给原流程。BF16 单个向量的 float32 范数允许小量舍入误差，三视图聚合仍执行原有 float32 归一化。

## PPU 运行步骤（已在指定实例验收）

1. 恢复指定实例 SSH 连接，核对当前文件摘要、GPU 状态、磁盘及依赖。
2. 保留平台 `torch 2.9.0+ppu2.0.0`、torchvision 0.24.0 和 Transformers 5.2.0。仅在缺少时补齐 `qwen-vl-utils==0.0.14`、兼容 accelerate 和 psutil；不能执行 Windows 的完整 requirements 覆盖 PPU 框架。
3. 下载或传输固定版本模型，逐文件核验官方 SHA-256／Git blob 摘要，最后生成完整模型清单。
4. 校验云端原文件摘要后部署本次 adapter、CLI、测试及必要源码变更，不能盲目覆盖其他任务的修改。
5. 通过已完成 SDK 配置的入口运行：

```bash
cd /workspace/work/road-infrastructure-finals-cloud
sh scripts/run_cqaip.sh qwen check --verify-hashes
sh scripts/run_cqaip.sh qwen smoke
```

该入口把所有缓存和验收产物留在 `/workspace`。PPU 初始权重显存预算为 80 GiB，运行时仍根据实际空闲显存预留至少 1.5 GiB；不会终止其他模型进程。实际峰值和兼容性必须由真实验收记录证明。

后续接入既有特征提取时，可在原本受控的 `config/cqaip.local.env` 配置以下无秘密字段；先检查已有内容，再合并，不能覆盖已有远程配置：

```bash
QWEN_PROVIDER_MODULE=qwen_local_provider
QWEN_LOCAL_MODEL_DIR=/workspace/work/road-infrastructure-finals-cloud/models/transformers/qwen3_vl_embedding
QWEN_LOCAL_GPU_GIB=80
QWEN_LOCAL_CPU_GIB=8
QWEN_LOCAL_BATCH_SIZE=1
PYTHONPATH=/workspace/work/road-infrastructure-finals-cloud/adapters
```

如需恢复远程官方接口，取消本地 `QWEN_PROVIDER_MODULE` 并恢复原接口设置；其他模型与分类器无须改动。是否满足赛事对接口、模型访问和正式结果的要求，仍需官方执行约定。

## 可选 Windows 入口

本机 RTX 3070 Laptop 为 8GB 显存，原始 8B 无法全部放入显存。适配支持 accelerate 的 GPU／CPU／磁盘分层加载，但本机真实前向未验证，速度和效果不能预先保证。

如后续仍需要 Windows 运行，在项目 `.venv` 完成安装；不要修改全局 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r road-infrastructure-finals-cloud-main/environment/requirements-qwen-local.txt
.\road-infrastructure-finals-cloud-main\scripts\run_qwen_local.ps1 check --verify-hashes
.\road-infrastructure-finals-cloud-main\scripts\run_qwen_local.ps1 smoke
```

`QWEN_LOCAL_GPU_GIB` 和 `QWEN_LOCAL_CPU_GIB` 的 Windows 默认值分别为 5 和 10，微批量为 1。用户需要覆盖时可用环境变量设置；已加载模型期间改变配置会被拒绝，先 `release_model()` 再重新加载。

已有普通本地图片可用 `embed` 保存真实三视图及聚合特征：

```powershell
.\road-infrastructure-finals-cloud-main\scripts\run_qwen_local.ps1 embed C:\path\image.jpg --output tmp\image_embedding.npz
```

这不授权把官方比赛图片下载到本机。已有输出文件默认拒绝覆盖。

## 验收结果与边界

- 13 项接口测试已通过：输入格式、网络 URL 拒绝、错误模型、空输入、资源参数、默认提示、三视图顺序、分批、释放、单设备报告以及旧 SDK／远程兼容。
- 真实 `smoke` 使用两张合成图，检查 `2×3×4096`、聚合维数、有限性、范数、不同图像差异、重复性、显存释放和零网络连接尝试。
- 真实验收已使用两张合成图，输出为 `2×3×4096`，聚合为 `2×4096`；范数范围约 0.99757–1.00311，不同图片余弦相似度约 0.03189，重复最大绝对误差为 0，网络连接尝试为 0。
- 经摘要核验的官方固定参考源码复用同一原始权重：预处理逐元素一致，向量最大绝对误差为 0。Transformers 5.2.0 缺少官方脚本导入但未使用的 `check_model_inputs`，验收入口只在导入期间临时补回恒等装饰器并立即撤销；官方源码和平台包未修改。
- 峰值 torch 已分配显存为 16,315,520,000 字节，释放后为 9,568,256 字节，模型引用已释放；随后 `preflight --require-qwen` 返回 `CLOUD_PREFLIGHT_PASS`。
- 即便上述全部通过，也不代表已对齐赛事远程接口或历史训练特征；没有同输入参考向量时，该一致性保持未知。
- 尚未执行完整 C4 特征链、比赛测试集推理、真实结果打包、正式上传或评分。
