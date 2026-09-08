"""把已视觉核对的H21面积指标蒸馏到表格专家，保留原表多项命中。

只覆盖表5.1.1-1/2/3的量化候选，不冒充所有规范或最终构件等级。
"""
import sys,json,time,itertools
from pathlib import Path
sys.path.insert(0,'/workspace/work/c4-experiments/expert-autotrain-20260906')
from download_assets import ROOT,dump,sha
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np
from catboost import CatBoostClassifier

SOURCE_SHA='a1c0184983f9ff1f156638338c322a59546483f17d681fe820ef85ac7ebb566c'

def quantitative_candidates(family,present,ratio,area):
    # 面积比以构件总面积为分母；单处面积单位m²。负值表示未提供。
    if present==0:
        return '1' if ratio<=0 and area<=0 else 'evidence_conflict'
    hits=[]
    if family=='honeycomb_pitting':
        if ratio<0:return 'insufficient'
        if ratio<=.5:hits.append(2)
        if ratio>.5:hits.append(3)
    elif family in ['spalling_corner_loss','cavity_hole']:
        # 原表明确使用“或”；两个量化维度不一致时保留命中集合，不能擅改成max。
        if 0<=ratio<=.05 or 0<=area<=.5:hits.append(2)
        if .05<ratio<.1 or .5<area<1.:hits.append(3)
        if ratio>=.1 or area>=1.:hits.append(4)
    else:raise ValueError('未核定的规范指标')
    return '+'.join(map(str,hits)) if hits else 'insufficient'

def main():
    source=ROOT/'standards/JTG_T_H21_2011/source.pdf'
    if sha(source)!=SOURCE_SHA:raise ValueError('规范原件摘要不符')
    out=ROOT/'checkpoints/normative_area_v1';out.mkdir(exist_ok=True)
    families=['honeycomb_pitting','spalling_corner_loss','cavity_hole']
    dump(out/'rulebook.json',{'standard':'JTG/T H21—2011','source_sha256':SOURCE_SHA,'source_url':'https://xxgk.mot.gov.cn/2020/jigou/glj/202006/P020240521541427295157.pdf',
        'scope':'5.1.1钢筋混凝土或预应力混凝土梁式桥上部承重及一般构件；其他桥型须有明确规范引用才适用',
        'tables':{'honeycomb_pitting':{'table':'5.1.1-1','pdf_page':20,'ratio_boundary':.5},'spalling_corner_loss':{'table':'5.1.1-2','pdf_page':20,'ratio_boundaries':[.05,.1],'area_m2_boundaries':[.5,1.]},'cavity_hole':{'table':'5.1.1-3','pdf_page':21,'ratio_boundaries':[.05,.1],'area_m2_boundaries':[.5,1.]}},
        'verification':'Agent已核对PDF第20、21页渲染；不是人工图像标注','inputs':['indicator_family','defect_present','cumulative_component_area_ratio','max_single_area_m2'],
        'output':'量化条件命中集合，仍需匹配定性描述和适用构件；不直接映射七字段ratingScale或整桥等级','missing':-1,'training_source':'公开规范程序生成情景，不使用比赛隐藏数据或旧评级作为规范真值'})
    rng=np.random.default_rng(20260906);train=[];labels=[]
    boundary_ratios=[-1,0,.05,.1,.5,1];boundary_areas=[-1,0,.5,1,2]
    for family in families:
        for _ in range(7000):
            present=int(rng.random()>.12)
            ratio=float(rng.choice(boundary_ratios)) if rng.random()<.25 else float(rng.random())
            area=float(rng.choice(boundary_areas)) if rng.random()<.25 else float(rng.random()*2)
            if present==0 and rng.random()<.8:ratio=area=0.
            train.append([family,present,ratio,area]);labels.append(quantitative_candidates(family,present,ratio,area))
    model=CatBoostClassifier(iterations=800,depth=8,learning_rate=.06,loss_function='MultiClass',thread_count=2,random_seed=20260906,allow_writing_files=False,verbose=100)
    dump(ROOT/'normative_area_status.json',{'status':'training_public_rule_scenarios','scenarios':len(train),'tables':3,'time':time.time()})
    model.fit(train,labels,cat_features=[0]);model.save_model(str(out/'model.cbm'))
    # 独立全边界组合审计；没有用结果挑选超参数，所有差异保留。
    rs=[-1,0,.049999,.05,.050001,.099999,.1,.100001,.499999,.5,.500001,1]
    ar=[-1,0,.499999,.5,.500001,.999999,1,1.000001,2]
    test=[list(x) for x in itertools.product(families,[0,1],rs,ar)];truth=[quantitative_candidates(*x) for x in test]
    pred=model.predict(test).reshape(-1);different=[{'facts':x,'rule':y,'model':str(p)} for x,y,p in zip(test,truth,pred) if y!=p]
    dump(out/'boundary_disagreements.json',different)
    dump(out/'complete.json',{'status':'three_area_tables_distilled_other_rules_pending','model_sha256':sha(out/'model.cbm'),'rulebook_sha256':sha(out/'rulebook.json'),'training_scenarios':len(train),'independent_boundary_scenarios':len(test),'exact_candidate_set_agreement':float(np.mean(pred==truth)),'disagreements':len(different),'runtime_contract':'明确阈值有权威程序可算时直接返回规则集合；模型输出供交叉核对，不覆盖原表边界','time':time.time()})
    dump(ROOT/'normative_area_status.json',json.loads((out/'complete.json').read_text()))

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'normative_area_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
