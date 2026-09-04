# CQAIP 平台部署与打包补充说明

本文对应 2026-09-03 的赛题 4 现成方案部署及 2026-09-04 的作品目录更新。原始设计书和 `reference/initial_best` 中的成绩、图像数量及预测属于初赛历史材料。本次运行使用 `config/deploy_config.json` 指定的 expQ 基线，不据此声称复现初赛 expV 的 85.49 分或获得决赛成绩。

## 运行位置与环境

- 部署目录：`/workspace/work/road-infrastructure-finals-cloud`。
- 数据目录：`/dataset/决赛数据集1/赛题4/线上测试集`，只在官方实例内读取。
- 已核实测试目录为平铺的桥梁 250 张、轨道 80 张，无重复文件名；未发现测试元数据或结果模板。字段补全后，无法由 2 km 内公开 GPS 近邻确定的桥名明确写为“未知桥梁”。
- 当前实例为 PPU-ZW810E，显存 96 GiB，Python 3.12.3，PyTorch 分发版本 `2.9.0+ppu2.0.0`，torchvision 0.24.0，timm 1.0.16，Transformers 5.2.0；`qwen-vl-utils==0.0.14` 隔离安装在项目 `.runtime/qwen_deps`。
- 使用 `scripts/run_cqaip.sh` 加载与平台 `envsetup.sh` CUDA 模式一致的完整 SDK 环境。仅设置动态库路径时，小矩阵检查可以通过，但实际 ConvNeXt 全连接计算会中止；补全 SDK 环境后，同形状 FP32 算子与 CPU 参考的最大绝对误差为约 `2.0×10⁻⁵`。不要用原通用 requirements 替换平台加速版 PyTorch。
- `ALIPPU_CONFIG_PATH` 要指向配置目录，该目录内的 `acompute.cfg` 使用 `[acConfigs]` 节设置 `RTC_CACHE_PATH`。`run` 会导出外部 `CQAIP_RUNTIME_ROOT`，入口将 ACompute、HGRTC、CUDA、Torch 等缓存和临时文件放在部署目录 `.runtime` 内，并关闭本进程的核心转储，不能写入 `result/code/.runtime`。

## 模型及当前限制

模型清单保持原始 `models/MODEL_MANIFEST.json`，总计 7,161,296,647 字节；自托管 Qwen 目录为 16,305,674,095 字节。两者保留在 `/workspace/work/road-infrastructure-finals-cloud/models`，不放入 `/workspace/result/code`。2026-09-04 曾为排查漏权重临时制作 23.47 GB 全量版本，随后用户明确修正作品合同：`code` 只提交代码，推理入口文件名为 `run`。当前作品目录已按新合同恢复，原模型源文件没有删除。

原方案要求 `Qwen3-VL-Embedding-8B` 输出 4096 维特征。平台 `/model` 中没有该模型，本次按用户授权在项目工作区自托管固定提交 `2c4565515e0f265c6511776e7193b22c0968ddc7`，通过 `QWEN_PROVIDER_MODULE=qwen_local_provider` 接入。21 个模型文件的完整摘要、真实 PPU 前向和官方固定源码对照均已通过；不要使用 `tests/mock_official_qwen_provider.py` 生成正式结果。自托管模型与历史训练 Embedding 的坐标是否逐值兼容、评测环境能否沿用工作区大权重，仍缺官方证明。

平台 Transformers 会默认选择 fast 图像预处理器，现成代码中的 `AutoImageProcessor` 调用保持原样。模型可运行性检查不能代替与初赛特征逐元素一致的回归验证，也不能代替决赛精度评估。

桥梁分支仍是原方案的固定“完好”策略，非实际桥梁诊断模型。没有官方模板时，桥名、病害位置、描述和等级由 `result_fields.py` 的可复现规则补全；其中未知桥名和无训练真值的等级不是实际诊断真值。本次部署不对其业务准确性作保证，也未重新训练。

## 真实推理入口

作品的标准入口是 `code/run`。无参数时它使用已核实的官方数据路径和工作区模型完成推理，并把结果原子写入 `/workspace/result/result/result.json` 与 `/workspace/result/result/infer_time.json`；`--check` 只检查入口与模型外置布局，不启动完整推理。也可显式传入 `MANIFEST_JSONL OUTPUT_DIR [TEMPLATE_JSON]`。

```bash
/workspace/result/code/run --check
/workspace/result/code/run
```

底层调试时可在部署目录执行以下命令。第一步只构造文件清单，后续推理必须先通过官方 Qwen 检查。

```bash
python -B src/build_cqaip_manifest.py --output .runtime/test_manifest.jsonl
sh scripts/run_cqaip.sh preflight --require-qwen
sh scripts/run_cqaip.sh infer .runtime/test_manifest.jsonl .runtime/inference
```

推理沿用原始融合流程：自托管官方原始 Qwen 三视图、四个视觉骨干、expQ 微调特征、可移植 LR、七字段 JSON 校验。成功后保存 `result.json`、`infer_time.json`、特征、概率及不含秘密的 `run_evidence.json`，均留在官方 `/workspace/work`。`infer_time.json` 只有一个数值字段；赛事材料没有显式给出单位，本实现根据整数示例按总推理毫秒数记录。输出目录已有内容时入口停止，避免覆盖之前的运行。

## 本次首版完整推理

2026-09-04 已在 `.runtime/inference-first-20260904` 完成 330 条正式清单推理，运行耗时 291.346 秒。结构校验覆盖桥梁 250 条、轨道 80 条；`Xbase` 为 `80×8832`、expQ 特征为 `80×1152`、概率为 `80×22`，全部为有限值，概率行和最大误差约 `4.44×10⁻¹⁶`。结果 SHA-256 为 `5db37fe9df54af5eac27d954b5b95369a3fd35c146568fbc34ad11b1ebf3a6b3`。

本次输出严格保留现有方案行为：250 条桥梁全部预测“完好”；330 条记录的桥名、病害位置、病害描述和等级均为空。这些字段通过当前七字段字符串和文件覆盖校验，但官方尚未说明空字段评分语义，因此“结构可提交”不等于这些字段可得分。轨道输出来自本次六分支真实特征及现成 22 类 LR；没有使用历史测试预测或占位 Embedding。

以上为空字段的内容只描述首版历史运行。用户随后要求补全四字段，并进一步要求刷新计时。缓存根修复后的最新完整运行位于 `.runtime/inference-fields-complete-20260904-v3`，耗时 306.742 秒；当前正式 `result.json` 与 `infer_time.json` 均直接来自该次运行，`infer_time` 为 `306742`。相对上一份正式结果有两条轨道分类变化，已按用户“以最新完整运行结果为准”的修正成对发布；当前 330 条四个目标字段均非空。

## 作品准备与打包

```bash
python -B src/package_cqaip.py prepare
# 真实推理成功后执行：
python -B src/package_cqaip.py build --run-evidence .runtime/inference/run_evidence.json
```

`prepare` 只在 `/workspace/work/c4_submission_staging` 准备运行代码、必要配置、`run`、单一方案设计书和空结果目录，不复制 `models/`，不生成占位 JSON，也不表示作品可提交。赛事说明要求 `design` 只提交方案设计书，因此本文只保留在工作源码，不进入作品 `design/`。

`build` 要求本次真实推理成功记录、文件哈希和七字段覆盖校验全部通过，才写入真实结果，生成 `/workspace/work/c4_submission.tar.gz`，检查包内目录并将相同内容放入 `/workspace/result`。目标目录已有文件时停止，不删除平台目录或覆盖用户文件。

最终结构为：

```text
code/                 运行代码、必要配置与可执行入口 run；不含模型
design/               原始方案设计书
result/result.json    本次真实推理产生的七字段结果
result/infer_time.json 本次完整推理总耗时（整数毫秒）
```

平台“打包上传作品”应以 `/workspace/result` 的这三个目录为内容根，不把已有 tar.gz 再套一层。作品名为 2–128 个非中文字符；上传额度和正式评测是不同事项。本脚本不会自动点击平台上传或发起正式评测。

历史首版 `package_cqaip.py build` 的结果和备份保留不变；当前正式结果已切换到字段补全后的最新完整运行。临时全权重版本已从作品目录撤回；`package_cqaip.py` 现只选择运行代码与必要配置，并确保 `run` 具有执行权限。模型及运行缓存仍在官方 `/workspace/work` 部署目录，可由 `CQAIP_MODELS_DIR` 和 `CQAIP_RUNTIME_ROOT` 显式指定。平台上传与正式评测不由本脚本自动触发。
