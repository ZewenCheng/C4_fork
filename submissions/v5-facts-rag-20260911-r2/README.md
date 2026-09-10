# C4 v5：伙伴分类、轻量事实链与RAG

用户指定本版作为当前v5大版本：沿用五折伙伴版分类，使用轻量事实评级和描述，连接两份PDF的RAG知识旁路。本轮不运行全量Qwen伙伴复核，不新增训练。用户采用授权与模型质量证据分开记录。

## r2入口兼容修复及时间来源

首版固定标签c4-v5-facts-rag-20260911保留。首版正式路径预检发现SDK envsetup.sh读取未定义LD_LIBRARY_PATH，发布已回滚，首版未成为正式作品。r2将SDK初始化置于nounset启用之前；随后仍启用严格变量检查，干净环境入口实际通过。

10个Python推理文件逐字节不变，模型和结果不变；r2复用本日真实330图新推理的已验收结果及7.473578秒纯模型时间，不声称修复后重新运行了模型。package_manifest.json分别记录实际运行源码清单与修订后的包装源码清单，entry_fix_validation.json绑定差异和预检证据。模型质量结论不因入口修复改变。

## 本轮结果

- 330张重新前向，输入与既有固定线上330图逐路径及内容摘要一致；七字段、模型、来源、RAG预算和计时独立验收通过。
- 纯模型推理：7.473578189秒；端到端：1157.580478秒。infer_time单位秒，含实际骨干、分类、RAG编码及评级模型区间并集；加载、预处理、传输、事实编码、检索整理与写入另计。
- 平台合同测试27项通过。评级分布：{"": 241, "2": 89}。
- 与旧伙伴版逐字段变化：{"questionCategory": 0, "bridgeName": 0, "defectLocation": 0, "filename": 0, "defectType": 0, "defectDescription": 330, "ratingScale(1-5)": 7}。变化不等于准确率或提分。
- 新比赛分数未知，本次未执行比赛上传或评测。

## 模型与知识来源

分类头来自3178份获授权初赛训练内容；事实评级模型使用632条可靠等级，110列输入。训练模型清单保留research_only及promotion_eligible=false的历史属性；本版在用户明确选择下封装，不伪造正向质量凭证。嵌套评级验证与全2基线持平，1级、5级可靠训练支持缺失。

RAG检索H21与CJJ/T233配套实施指南，实际页码证据留平台逐图记录。检索是知识旁路，不直接决定评级或图像程度。每来源独立8192 token预算，预留1024；轨道与完好按既定策略跳过。材料跨域训练尚未纳入本版，生锈未添加额外资料。

## 运行与恢复

固定平台版本根：`/workspace/work/c4-submission-versions/v5-facts-rag-20260911-r2`。
提交压缩包：`/workspace/work/c4-submission-versions/v5-facts-rag-20260911-r2/c4-v5-facts-rag-20260911-r2.tar.gz`。
压缩包SHA256：`9a8b0ac492d5df27e6ab8ae69d4b633ab61912edd57371a3f26230a464b2a83f`。

`submit/`为code/design/result三目录，`run_snapshot/`为本轮原始七字段、逐图事实、RAG引用、计时和来源凭证。公开Git只保存本目录代码、设计、验证工具与汇总；原始图像、特征、模型、逐图答案不得下载或上传公开仓库。

恢复前按asset_manifest.json检查外置模型、RAG索引和tokenizer的路径及摘要。运行依赖既有平台PPU SDK与`/workspace/work/c4-facts-rag-20260910-v1/deps`，不自动联网替换资产。

```bash
bash code/run --check
bash code/run --input /dataset/决赛数据集1/赛题4/线上测试集 --work /workspace/work/c4-v5-new-run --output /workspace/work/c4-v5-new-output
```

输入根须包含桥梁/轨道目录。work和结果路径应为新的独立目录；已存在结果或工作目录拒绝覆盖。运行将输出配对result.json/infer_time.json，首次运行完成前不可把中间进度当正式结果。

固定GitHub标签：`c4-v5-facts-rag-20260911-r2`。平台在独立GitHub读回通过后才切换官方作品；旧伙伴作品保留在`/workspace/work/c4-submission-versions/before-v5-facts-rag-20260911-r2`，是否实际完成切换以平台publication_receipt.json及独立读回为准。
