# 桥名与结构位置语义修复版提交包

版本`contract-v2-metadata2-20260909`修复parallel2把“桥梁/轨道”分类目录写作桥名、把图像方位写作工程结构位置的问题。330条结果只改变桥名和位置，其余五字段、模型及并行参数保持。用户明确选择“直接封装现有修复结果”，本包没有重新完整推理，尚无本版正式得分。

## 制品与验证

固定压缩包位于官方工作区：

`/workspace/work/c4-submission-versions/contract-v2-metadata2-20260909/c4-contract-v2-metadata2-20260909.tar.gz`

大小340772字节，SHA-256为`0060fa4404eae330c35df9a498d924f48a9a7efb13747eebc8dcfc7b59376297`。同目录`submit/`为不可覆盖的展开快照；平台作品目标为`/workspace/result`。压缩包只有`code/`、`design/`、`result/`，共32文件：29个代码/清单文件、1份四页中文方案PDF、330条七字段`result.json`与单字段`infer_time.json`。

本地及平台固定源码53项回归通过，平台330条新凭证全部恢复、330条旧凭证全部被新协议拒绝。独立SQL与结构位置语义核验、压缩成员及逐文件SHA核验通过；本次封装重新核验26个外置模型资产、25个恢复副本及Qwen元数据，执行了实际`run --check`与准备目录恢复检查。详见[包清单](package_manifest.json)、[独立验包](verification/package_validation.json)与[提交版本记录](submission_version.json)。

## 修复内容

- 桥名沿用v2九个公开数据聚合参考点、2公里近邻阈值和文件名左右幅规则。330条与v2一致：174条坐标近邻候选、76条明确未知、80条轨道不适用。近邻不代表真实桥梁身份已独立核实；推理不读取旧测试预测作为输入。
- 位置保留47条文件名结构部位或编号，兼容空格及括号照片序号；74条类型关联明确标注为推断，209条明确具体位置未知。图像方位可保留于描述和内部证据，不充当工程部位。
- 新增`metadata_semantics.py`，输出合同升级为`report-contract-v2-metadata`，元数据源码SHA进入每条恢复上下文。新入口拒绝混用旧工作目录。

本次确定性重装配由[重装配脚本](verification/reassemble_metadata.py)读取已核验输入、专家包和107条旧评级凭证，0次新模型调用。结果SHA-256为`ecc86813d9d5c9eeccb1a7942a98e07b717f7e77d9ac5102acd5009e4d7bc1d0`。逐图结果和凭证仅保留官方工作区，不进入公开GitHub。

## 计时来源

`result/infer_time.json`原样保留`{"infer_time":6568.275286197662}`，单位为秒，约109.47分钟。它来自parallel2历史完整模型入口的`time.time()`实测；本包按用户选择直接封装修复结果，不能称为修复代码新一次完整推理实测。

本轮修复及恢复核验154.12058997154236秒（约2.57分钟）独立记录；新完整推理时间为`null`。不把历史完整推理和后处理秒数相加冒充一次新入口墙钟计时。日后运行本版`run`会从Python脚本开头重新计时，执行全部阶段后生成新的结果和计时；生成不同结果时应保存为另一制品。

## 固定备份与恢复

固定标签为[c4-contract-v2-metadata2-20260909](https://github.com/ZewenCheng/C4_fork/tree/c4-contract-v2-metadata2-20260909)。GitHub保留本目录源码、配置、设计书、模型及结果摘要、验证脚本和聚合证据。官方快照目录的`publication_receipt.json`记录独立GitHub读回证明和目录切换凭证，`live_validation.json`记录切换后独立读回；两者在备份验收后生成，不修改本固定标签。

精确恢复使用官方`submit/`快照或上述tar.gz，先按包清单核验SHA。原始重装配输入及凭证位于`/workspace/work/c4-metadata-semantics2-20260909/assembled`；新代码准备目录为其同级`prepared-entry`。模型来自[模型清单](code/model_manifest.json)登记的`/workspace/work/c4-experiments/expert-autotrain-20260906`资产与备份，Qwen为平台只读`/model/Qwen3.6-27B`。执行入口是`code/run`，Qwen批量3、CPU4线程，保持区域编码复用。模型与缓存不置于提交代码目录。

切换前旧parallel2的31文件保留于`/workspace/work/c4-submission-versions/before-contract-v2-metadata2-20260909`，旧快照及固定标签保持。53.36属于历史parallel2用户反馈，60.85属于旧v2；均不是本版得分。尚未执行比赛上传或评测。

[version_record.json](version_record.json)及修复验证文件保留候选生成时“未正式打包”的历史状态；当前包状态由`submission_version.json`、`package_manifest.json`及官方发布/读回凭证共同确定。首轮漏识别带空格文件名的候选保留在本地相邻目录，不是本提交制品。
