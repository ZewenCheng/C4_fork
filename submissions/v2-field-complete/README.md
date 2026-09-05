# v2：字段补全版本，用户报告 60.85 分

用户确认：v1=55.15、v2=60.85，v2 只补齐结果字段，与补权重无关，两版权重均未丢失。Agent 评分可能约 ±1 抖动。分数与版本差异依据用户确认；精确评测回执暂未提供。

本次归档保留的是纯字段补全状态：结果 SHA-256 为 `a319a9fc1b36d0028c63832ebd0bb46500f5d00f8d6ceaa92fea860a56e378ef`，共 330 条，四个目标字段非空，病害类型与补全前完全一致；原计时保留 `291346`，单位尚待官方确认。这个摘要由官方工作区的补全后备份现场核验，与用户描述的“只补字段”一致；尚无评分回执将该摘要独立绑定到 60.85。

`code/` 是通过历史代码清单逐文件核验后恢复的原始运行代码快照，包含当时的 `run` 和缓存路径行为。主仓库的 `run` 已有后续缓存修复，两者有意分别保留；不要把主分支代码当成原始 v2 逐字节副本。设计书复用仓库 [原设计书](../../reference/initial_best/智能体设计方案_已核验.pdf)，与云端文件 SHA-256 一致。机器可读的定位、摘要和边界见 [version.json](version.json)。

GitHub 的 v2 标签固定本目录及其引用文件。模型来源和文件摘要保存在 `model-manifests/`；视觉模型沿用原仓库 LFS，Qwen 使用固定提交 `2c4565515e0f265c6511776e7193b22c0968ddc7`。本轮未改模型或重复上传权重。

精确原始结果和完整作品快照仅保存在官方工作区：

```text
/workspace/work/c4-submission-versions/v2-field-complete/
├─ package/code/
├─ package/design/
├─ package/result/result.json
├─ package/result/infer_time.json
├─ model-manifests/
└─ snapshot.json
```

恢复精确已保存作品时，在官方实例先核验 `snapshot.json` 中全部摘要，再复制 `package/` 到一个新的、空的 `/workspace/work` 子目录。模型目录仍为 `/workspace/work/road-infrastructure-finals-cloud/models`；验证模型文件与归档清单后再进行下一步。示例仅复制备份，不改当前发布目录：

```sh
set -eu
test ! -e /workspace/work/c4-recovered-v2
cp -a /workspace/work/c4-submission-versions/v2-field-complete/package /workspace/work/c4-recovered-v2
cd /workspace/work/c4-recovered-v2/code
sha256sum -c MANIFEST.sha256
chmod +x run
```

需要再次推理时，应先固定输入、环境与模型，并将输出写入新的隔离目录。原始 v2 保留当时的缓存路径逻辑，后续维护版修复了缓存误写代码目录的问题。历史运行存在少量分类变化，重新推理不能保证得到已保存结果的相同字节；要回溯原结果，应直接恢复已核验快照。

后续完整重跑单独保存在 `/workspace/work/c4-submission-versions/field-complete-rerun-20260904-v3/`：结果摘要为 `0e8dc6faf004dc73bd66d2b5b7f88ba5fa81fef21642f387e8088ae07e77e69c`，计时 `306742`，与纯补全结果有 2 条病害类型差异。该重跑产物没有单独提供的评分回执，不能静默替代本目录的纯补全结果或宣称同为 60.85 分。
