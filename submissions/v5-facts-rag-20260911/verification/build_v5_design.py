"""从v5本轮汇总生成三页中文方案设计书。"""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'output/pdf/v5/智能体设计方案_v5.pdf'
pdfmetrics.registerFont(TTFont('C4Sans','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('C4Bold','C:/Windows/Fonts/msyhbd.ttc'))
NAVY=colors.HexColor('#17334A');TEAL=colors.HexColor('#087F83');LIGHT=colors.HexColor('#EFF5F7')
WIDTH=A4[0]-96
styles={
 'body':ParagraphStyle('body',fontName='C4Sans',fontSize=9.5,leading=15,textColor=NAVY,wordWrap='CJK',spaceAfter=7),
 'small':ParagraphStyle('small',fontName='C4Sans',fontSize=8.5,leading=14,textColor=NAVY,wordWrap='CJK',spaceAfter=6),
 'title':ParagraphStyle('title',fontName='C4Bold',fontSize=21,leading=30,textColor=NAVY,wordWrap='CJK',spaceAfter=12),
 'h2':ParagraphStyle('h2',fontName='C4Bold',fontSize=13,leading=20,textColor=TEAL,wordWrap='CJK',spaceBefore=7,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='C4Sans',fontSize=8.6,leading=13.8,textColor=NAVY,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='C4Bold',fontSize=9,leading=15,textColor=colors.white,wordWrap='CJK')}
def p(text,style='body'):return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
def table(rows,widths):
    t=Table([[p(x,'head' if i==0 else 'cell') for x in row] for i,row in enumerate(rows)],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT,colors.white]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t
def footer(c,doc):
    c.saveState();c.setFillColor(NAVY);c.setFont('C4Sans',8)
    c.drawString(48,A4[1]-32,'C4 / V5 · 伙伴分类 + 事实链 + RAG')
    c.setStrokeColor(colors.HexColor('#D5E1E7'));c.line(48,42,A4[0]-48,42)
    c.drawString(48,28,'版本 v5-facts-rag-20260911 | 方案设计与验证边界')
    c.drawRightString(A4[0]-48,28,str(doc.page)+' / 3');c.restoreState()
def main():
    evidence=json.loads((ROOT/'tmp/v5-verification.json').read_text(encoding='utf-8'))
    summary=evidence['候选汇总'];assert summary['样本数']==330 and summary['660秒速度门']
    story=[Spacer(1,8),p('城市路桥隧边坡结构病害\n智能巡检与分级评定','title'),p('v5 方案设计书','h2'),
        p('本版沿用五折多视图伙伴版的分类主干，连接轻量事实评级、确定性事实描述与两份桥梁资料检索。每张输入图像重新执行视觉前向，七字段结果由统一合同装配。'),
        table([['环节','处理与输出'],['输入与身份','扫描桥梁、轨道图像；路径与图像摘要共同确定身份，桥名读取EXIF及既有公开坐标近邻，缺失保留未知。'],
               ['三视图分类','ConvNeXtV2 Large共享骨干处理全图、水平翻转及中心方裁切；固定多视图分类头输出病害候选。'],
               ['事实协议','将预测类型、结构域及元数据组织为有来源的事实，区分观察、估计、未知和冲突。'],
               ['双来源检索','桥梁病害检索H21标准原文和CJJ/T233配套实施指南；逐图保留请求、页码、来源与预算凭证。'],
               ['评级与描述','可靠训练等级拟合的轻量评级器预测病害等级；事实描述逐断言核验；完好按既有合同保留空等级。'],
               ['输出','生成七字段result.json和以秒记录的infer_time.json，完整性通过后才向输出目录写入。']], [90,WIDTH-90]),
        Spacer(1,10),p('1 版本范围','h2'),p('v5采用用户明确选择的轻量主链，不执行全量Qwen伙伴复核。本轮不新增训练、不改变分类参数、不利用最终测试标签或结果选择策略。材料跨域学习仅作为后续方案，尚未纳入本版。')]
    story += [PageBreak(),p('证据、模型与输出约束','title'),p('2 固定模型来源','h2'),
        p('多视图分类头来自3178份获授权初赛训练内容，使用实体/内容隔离的五折验证后在全训练内容拟合。其多视图相对固定全图头的历史验证为2295→2300命中；该结果不代表七字段比赛收益。'),
        p('事实评级器输入为固定110列编码，以632条可靠等级拟合。编码器源码、特征顺序、模型文件、训练清单及sklearn版本均核验。历史嵌套验证与全2基线持平，缺少1级和5级可靠训练支持；本次封装授权不会改写该质量结论。'),
        p('3 RAG的作用与限制','h2'),
        p('两份PDF已形成352页OCR、1184个512维语义向量。每个来源独立检索，单请求总预算8192 token，预留1024；候选类型最多两个，采用既有limit=1策略，同次相同知识请求可复用。'),
        p('检索输出是规范知识旁路。其页码与来源可以回溯，但不直接决定最终等级，也不生成图像中未观测到的宽度、长度、面积或程度。两来源的适用角色保持分离；轨道与完好按既有策略跳过桥梁检索。'),
        p('4 七字段合同','h2'),
        table([['字段','来源与约束'],['questionCategory / filename','输入结构域与原始文件名；不由语言模型改写。'],['bridgeName / defectLocation','沿用元数据协议；类型关联构件明确标推断，缺少证据保留未知。'],['defectType','固定多视图分类头首选；本版无伙伴改判。'],['defectDescription','仅依据事实协议逐断言生成；RAG知识不冒充图像事实。'],['ratingScale(1-5)','病害为模型实际等级字符串；完好为空字符串。']], [154,WIDTH-154])]
    story += [PageBreak(),p('全量实测与部署交付','title'),p('5 本轮330图实测','h2'),
        table([['指标','v5本轮结果'],['输入和输出覆盖','330 / 330；独立核验输入、模型、七字段、来源与计时'],
               ['纯模型推理',f"{summary['纯模型秒']:.6f} 秒；660秒目标通过"],['端到端',f"{summary['端到端秒']:.6f} 秒"],
               ['平台合同测试','27项通过；包括评级、来源校验、输入扫描和源码篡改拒绝'],
               ['评级分布', '；'.join(('完好空等级' if k=='' else k+'级')+' '+str(v)+'条' for k,v in summary['评级分布'].items())],
               ['正式比赛成绩','尚无本版比赛评测分数，不声称精度提升']], [125,WIDTH-125]),
        p('纯模型时间累计实际视觉骨干、分类头、RAG编码器及评级预测计算区间，并发区间按并集计算，设备计算结束同步。加载、预处理、传输、事实编码、检索整理和写入不计入纯模型时间；端到端另计。','small'),
        p('6 运行与恢复','h2'),p('提交包顶层为code、design、result。code/run启动平台SDK环境、验证源码后从输入图像重跑；以--input指定输入、--work指定新的工作目录、--output指定新的输出目录。已有输出或工作目录拒绝覆盖。--check执行入口与资产预检。'),
        p('模型、RAG索引、tokenizer与隔离依赖保留官方/workspace/work及/model的固定来源。公开版本记录提供精确摘要和恢复路径；不将模型、比赛图像、特征或逐图预测上传公开仓库。正式运行依赖这些外置资产可见。'),
        p('提交包与运行快照独立冻结，GitHub固定标签绑定代码、设计书、模型来源与聚合摘要。旧正式作品完整保留，切换后再次逐文件读回。封装与备份不等于比赛上传或评测。'),
        p('7 已知质量边界','h2'),p('本版验证证明全链可运行、产物完整和速度达标。实际模型可能全部预测2级；没有同划分旧评级对照及新比赛成绩时，不能判定等级变化或描述改写带来准确率收益。')]
    DEST.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(DEST),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=55,bottomMargin=55).build(story,onFirstPage=footer,onLaterPages=footer)
    print(str(DEST))
if __name__=='__main__':main()
