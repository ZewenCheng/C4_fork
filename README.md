# 城市路桥隧边坡结构病害智能巡检：决赛云端底座

本仓库保存赛题四初赛最终训练成果，以及用于决赛官方云端继续训练和推理的完整底座。仓库包含全部本地视觉模型权重、expQ 全量模型、EVA02 五折权重、可移植 LR 分类器、Qwen 官方接口适配层、训练/推理脚本和回归证据。

> 重要定位：这是“初赛最终成果 + 决赛优化底座”，不是已经适配决赛封闭数据的最终模型。初赛 85.49 只代表历史数据上的线上成绩。

## 仓库内容

- expQ SigLIP2 全量 FP32 权重；
- Google SigLIP2、Facebook DINOv2-L、Facebook ConvNeXtV2-L、timm EVA02-L 完整权重；
- 5 个 EVA02 折权重；
- 初赛 expQ LR 系数、截距、类别顺序和特征合同；
- 官方 Qwen3-VL Embedding 的 HTTP/本地 SDK 适配层；
- 云端 manifest、特征提取、继续微调、LR 重训、推理和七字段 JSON 校验程序；
- 初赛结果血缘、设计方案及离线回归记录。

Qwen 权重不在仓库中。决赛运行时必须使用赛事官方提供的 Qwen 服务，不得把决赛图片发送到公开模型网站或非官方接口。

## 克隆方式

模型由 Git LFS 保存，文件保持原始完整形态，没有分片。请先安装 Git LFS：

```bash
git lfs install
git clone https://github.com/13385696587/road-infrastructure-finals-cloud.git
cd road-infrastructure-finals-cloud
git lfs pull
```

完整检出约需 7GB 以上空间。若只看到一百多字节的权重指针文件，说明尚未执行 `git lfs pull`。

## 完整性检查

```bash
python src/validate_package.py
python src/preflight.py --device auto
```

模型文件的原始大小和 SHA-256 记录在 `models/MODEL_MANIFEST.json`。本仓库源部署包 SHA-256 为：

```text
c75925e79e469f5b6b63f2629fa4a16ec43c936c4ebfad1ae0a9513db81c1b11
```

GitHub 版本仅移除了 macOS 元数据、Python 字节码缓存等无运行价值文件，模型权重没有修改。

## 环境准备

推荐使用赛事官方 PyTorch/CUDA 镜像，只补装缺失依赖：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment/requirements.txt
python src/preflight.py --device auto
```

也可从仓库根目录构建容器：

```bash
docker build -f environment/Dockerfile -t track4-finals-base .
```

## 配置官方 Qwen

如果官方提供 OpenAI 兼容接口：

```bash
export QWEN_BASE_URL='官方接口地址'
export QWEN_MODEL='官方公布的模型名'
export QWEN_API_KEY='仅在官方要求认证时设置'
```

如果官方提供本地 Python SDK，请实现 `adapters/qwen_provider_example.py` 中的 `embed_items`，再设置：

```bash
export PYTHONPATH="$PWD/adapters:$PYTHONPATH"
export QWEN_PROVIDER_MODULE='适配模块名'
```

仓库不包含 API Key、访问令牌或外部默认 Qwen 地址。

## 建立数据清单

优先读取官方元数据：

```bash
python src/build_manifest.py \
  --dataset-root /cloud/dataset \
  --metadata-json /cloud/dataset/train_labels.json \
  --output /cloud/work/train_manifest.jsonl
```

无元数据时可以扫描目录：

```bash
python src/build_manifest.py \
  --dataset-root /cloud/dataset \
  --output /cloud/work/test_manifest.jsonl
```

正式运行前必须根据官方目录结构复核类别和桥名映射，但不得人工查看封闭测试图后逐行改预测。

## 运行初赛 expQ 基线

```bash
./scripts/run_infer.sh \
  /cloud/work/test_manifest.jsonl \
  /cloud/work/inference \
  /cloud/dataset/sample_result.json
```

流程为：官方 Qwen 三视图 Embedding → 四个本地视觉骨干 → expQ 全量模型 → LR → 七字段 `result.json` → 结构校验。

## 使用决赛训练集继续优化

```bash
./scripts/run_train.sh \
  /cloud/work/train_manifest.jsonl \
  /cloud/work/training
```

使用新模型推理：

```bash
EXPQ_DELTA=/cloud/work/training/expq_cloud_delta.pt \
CLASSIFIER=/cloud/work/training/expq_cloud_lr.npz \
./scripts/run_infer.sh \
  /cloud/work/test_manifest.jsonl \
  /cloud/work/inference \
  /cloud/dataset/sample_result.json
```

决赛优化时应另建分组验证集。只有新模型在验证集上稳定优于本仓库基线时，才应用于封闭测试集。

## 已知边界

- 当前轨道分类空间来自初赛出现过的 22 种组合；决赛若出现新组合，需要先改为动态多标签解码或扩展类别空间。
- 当前桥梁分支沿用初赛保守“完好”基线，不是完整的桥梁病害、描述和评级模型。
- 初赛 expV 的固定概率及 9 条后验变化不能套用到决赛新图片。
- 官方 Qwen 的真实协议、批量限制、认证方式、数据挂载路径和 GPU 资源必须在决赛环境中确认。
- 长批次运行前建议补充断点续跑、分编码器缓存和显存不足时自动减小 batch 的能力。

更多说明见 `README_云端部署.md` 和 `docs/`。

## 数据与安全

- 仓库不包含决赛原始图片；
- 不包含 GitHub Token、Qwen API Key 或个人绝对路径；
- 所有决赛图片、Embedding、日志和结果都应保留在官方云端；
- 不要提交 `.env`、私钥、访问令牌、云端数据或个人实验输出。

## 模型许可

模型文件保持其上游许可，不能由本仓库统一重新授权：

- Google SigLIP2：Apache-2.0；
- Facebook DINOv2-L：Apache-2.0；
- Facebook ConvNeXtV2-L：Apache-2.0；
- timm EVA02-L：MIT；
- expQ 基础 SigLIP：Apache-2.0；
- 官方 Qwen：以赛事官方服务条款为准，本仓库不分发其权重。

请同时阅读各模型目录中的模型卡和 `docs/05_模型来源与许可.md`。
