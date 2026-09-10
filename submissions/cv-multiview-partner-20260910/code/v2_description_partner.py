"""复用v2中文病害词项，保留当前事实与预测层级，不继承旧评级或全桥默认值。"""
from report_contract import validate_final


def describe(final,diagnosis):
    validate_final(final,diagnosis)
    output=dict(final)
    if diagnosis['is_intact']:
        text='当前分类预测为完好，尚不能排除未识别病害。'
    else:
        label=final['defectType'].replace('裂缝(混凝土裂缝)','混凝土裂缝')
        text='模型预测病害为'+label+'，预测评级为'+final['ratingScale(1-5)']+'级。'
    if final['defectLocation']:
        text+='结构位置：'+final['defectLocation']+'。'
    text+='尺寸与构件受损比例未知。'
    output['defectDescription']=text
    validate_final(output,diagnosis)
    return output
