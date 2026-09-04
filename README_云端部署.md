# 赛题四决赛云端训练与推理包

本包面向“决赛数据仅存在于官方云端、禁止下载”的运行条件。它不包含决赛数据，也不会尝试复制或导出云端图片。

## 已补齐的部署能力

- 官方 `/workspace/work` 部署目录已带入初赛expQ原始FP32全量SigLIP2权重；作品 `result/code` 不复制模型；
- Qwen仍使用决赛官方提供的Qwen3-VL Embedding，不提供Qwen-free分支；
- 官方部署目录已带入初赛实际使用的Google SigLIP2视觉塔、Facebook DINOv2-L、Facebook ConvNeXtV2-L和timm EVA02-L离线权重；
- 初赛expQ LR已导出为可移植NPZ，不依赖pickle或训练数据即可推理；
- 支持云端继续微调expQ、重新抽取特征、重新拟合LR和生成七字段JSON；
- 所有路径由manifest和配置文件驱动，不依赖初赛固定文件名或本机绝对路径；
- 支持CUDA、MPS和CPU自动选择；
- 提供离线模型哈希预检、结果覆盖检查和官方Qwen两种适配方式。

## 重要边界

初赛expV 85.49的五折概率只对应初赛300张测试图，不能直接套用到决赛新图片。官方工作区部署保留 expQ 全量模型、EVA 五折权重和初赛 expV 概率供研究；作品代码目录不携带这些模型资产。

桥梁分支沿用初赛保守“完好”策略。若决赛提供新的桥梁训练标签，应在云端建立独立桥梁模型，而不是根据测试图片人工改行。

## 1. 作品推理入口

提交代码的无后缀可执行入口为 `run`。模型由 `CQAIP_MODELS_DIR` 指向官方工作区，默认使用 `/workspace/work/road-infrastructure-finals-cloud/models`，不会复制到 `result/code`。模型缓存、PPU 编译缓存和临时文件由 `CQAIP_RUNTIME_ROOT` 固定到同一外部工作区，不会写入作品代码目录。

```bash
./run --check
./run
# 也可显式指定：
./run MANIFEST_JSONL OUTPUT_DIR [TEMPLATE_JSON]
```

无参数执行时扫描已核实的赛题 4 测试目录，完整推理后原子更新 `/workspace/result/result/result.json` 和 `/workspace/result/result/infer_time.json`；`--check` 只核对入口与外部模型布局，不启动完整推理。

## 2. 环境准备

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment/requirements.txt
python src/preflight.py --device auto
```

如官方容器已经预装PyTorch/Transformers，应优先使用官方镜像，只补齐缺失依赖。

## 3. 配置官方Qwen

### OpenAI兼容官方接口

```bash
export QWEN_BASE_URL='官方地址，例如 http://qwen-service/v1'
export QWEN_MODEL='Qwen3-VL-Embedding-8B'
export QWEN_API_KEY='官方要求时设置；不需要认证可不设置'
```

代码不会提供外部默认地址；未配置官方地址时会直接停止。

### 官方Python/本地SDK

实现 `adapters/qwen_provider_example.py` 中的 `embed_items`，然后：

```bash
export PYTHONPATH="$PWD/adapters:$PYTHONPATH"
export QWEN_PROVIDER_MODULE='你的官方适配模块名'
```

## 4. 建立云端数据清单

优先使用官方元数据：

```bash
python src/build_manifest.py \
  --dataset-root /cloud/dataset \
  --metadata-json /cloud/dataset/train_labels.json \
  --output /cloud/work/train_manifest.jsonl
```

无元数据时按目录扫描：名为“轨道”的目录判为轨道，其余图片父目录名作为桥名。正式运行前应人工检查目录规则，但不能人工检查测试图片并据此改预测。

## 5. 直接使用初赛expQ推理

```bash
./scripts/run_infer.sh \
  /cloud/work/test_manifest.jsonl \
  /cloud/work/inference \
  /cloud/dataset/sample_result.json
```

流程：官方Qwen三视图Embedding → 四个离线视觉骨干三视图特征 → expQ微调特征 → 初赛LR → 七字段JSON → 结构校验。

本地视觉分支会先严格复刻初赛预处理：最长边缩至不超过1024、Lanczos插值、JPEG质量92重新编码；Qwen分支仍直接处理官方原图并生成768像素三视图。

## 6. 在决赛训练集继续优化

```bash
./scripts/run_train.sh \
  /cloud/work/train_manifest.jsonl \
  /cloud/work/training
```

该命令会：

1. 从初赛expQ delta继续微调最后4块、norm和22类head；
2. 使用新delta重新抽取云端训练特征；
3. 在云端训练标签上重新拟合LR；
4. 输出 `expq_cloud_delta.pt` 与 `expq_cloud_lr.npz`。

使用新模型推理：

```bash
EXPQ_DELTA=/cloud/work/training/expq_cloud_delta.pt \
CLASSIFIER=/cloud/work/training/expq_cloud_lr.npz \
./scripts/run_infer.sh /cloud/work/test_manifest.jsonl /cloud/work/inference /cloud/dataset/sample_result.json
```

## 7. 输出和合规

- 作品顶层只保留 `code/`、`design/`、`result/`；`code/` 只装运行代码与必要配置，`design/` 只装方案设计书。
- 赛题 4 的 `result/` 同时包含七字段 `result.json` 和只含 `infer_time` 数值的 `infer_time.json`。官方示例未写单位；本实现按整数毫秒记录本次完整推理耗时。
- 附带的四字段示例含“有漂浮物”标签，与赛题 4 的字段和任务语义不一致，不用于替换赛题 4 输出合同。
- 只向官方Qwen服务发送决赛图片；禁止使用非官方URL。
- 特征、日志、权重和结果全部保存在官方云端工作目录。
- 禁止将图片、Embedding或逐图结果下载到本地。
- 阈值、类别和模型选择只能依据云端训练/验证数据。
- 测试阶段只允许批量推理和JSON结构检查，不允许人工看图挑行。
- 初赛85.49结果保存在 `reference/initial_best/`，仅作血缘和回归参考。

完整回归证据见 `docs/04_回归测试报告.md`。

## 8. 推荐决赛操作顺序

1. 运行 `preflight.py --require-qwen`；
2. 用2至4张官方提供的非测试示例做Qwen和模型冒烟；
3. 在云端训练集生成manifest并划分组别验证集；
4. 先复测初赛expQ，再决定是否继续微调；
5. 新模型通过验证后才运行决赛测试推理；
6. 保存模型、配置、manifest哈希和最终结果哈希。

现场仍需确认的官方接口、路径和资源限制见 `docs/06_决赛现场待确认项.md`。
