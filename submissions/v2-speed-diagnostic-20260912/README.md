# v2提速版独立提交包

用户明确选择speed r1 v2（开发评级162/195）。本版保持原r3类型诊断与短引用描述/评级，不是167/195慢版或历史60.85比赛版。正式330图已从CV开始完整新推理，独立输入、来源、七字段、计时和恢复验收通过。当前正式v5未覆盖，本版未比赛上传或评测。

## 实测与边界

- 输入330张（250桥梁、80轨道），新LLM调用757次，输出23795词元。
- infer_time为4734.337030秒，按Python入口time.time差值计端到端；含资产校验、加载、预处理、CV、检索与生成。生成区间另列，不能与历史纯模型时间直接比。
- 协议问题：{}。原始响应及失败均留平台，不把解析失败空值算健康正确。
- 开发类型298→256、轨道171→106；速度版评级167→162。用户选择不等于整体最优，无本版比赛得分。
- v2旧轨道头训练重叠未知。桥梁CV及描述/评级候选来自排除345开发图的fold-0同架构teacher，而非正式v5全量头。训练描述2017条及两RAG均为桥梁，轨道无适用语料。
- Qwen3.6-27B保持历史Peft包装并禁用适配器。未新增训练；短编号只证明可映射，不能证明模型采用或语义正确。

## 运行与恢复

平台独立包：`/workspace/work/c4-submission-versions/v2-speed-diagnostic-20260912/submit`；归档：`/workspace/work/c4-submission-versions/v2-speed-diagnostic-20260912/c4-v2-speed-diagnostic-20260912.tar.gz`。

运行`bash code/run --input /dataset/决赛数据集1/赛题4/线上测试集 --work /workspace/work/新的独立目录 --output /workspace/work/新的输出目录`。已存在工作目录需显式`--resume`，完成恢复只核对资产、输入和结果摘要，不重新加载模型。新完整运行不得使用开发预测缓存。

模型与源码依赖按asset_manifest.json的路径和SHA校验；缺失立即停止，不自动换模型。小型语料/teacher/RAG副本在`/workspace/work/c4-submission-versions/v2-speed-diagnostic-20260912/external_asset_snapshot`，恢复时按清单还原至`/workspace/work/c4-v2-speed-release-20260912-r1/assets`，大模型保持清单列出的外部原路径。模型和数据不放code。原始运行快照：`/workspace/work/c4-submission-versions/v2-speed-diagnostic-20260912/run_snapshot`。

精确文件清单和压缩包摘要见package_manifest.json；代码、设计、来源和聚合证据通过固定标签`c4-v2-speed-diagnostic-20260912`备份。标签发布后不移动；修正另建版本。完整图片、训练标注、特征、逐图预测和凭证不进入公开仓库。
