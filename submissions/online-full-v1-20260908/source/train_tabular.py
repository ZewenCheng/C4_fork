"""训练结构化历史评级专家；规范规则与视觉端到端评估另行接入，不混报。"""
from pathlib import Path
import sys
import os
import json
import re
import time
import subprocess
from download_assets import ROOT,dump,sha
deps=ROOT/'deps';deps.mkdir(exist_ok=True);sys.path.insert(0,str(deps))
try:from catboost import CatBoostClassifier
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','--target',str(deps),'--no-deps','--disable-pip-version-check','catboost==1.2.8','graphviz==0.21'],check=True)
    from catboost import CatBoostClassifier
import numpy as np

def facts(annotation):
    desc=str(annotation.get('defectDescription',''))
    # 不输入ratingScale、桥名、文件名，避免等级目标与结构身份直接泄漏。
    nums=[]
    for pattern,scale in [(r'(\d+(?:\.\d+)?)\s*mm',1),(r'(\d+(?:\.\d+)?)\s*cm',10),(r'(\d+(?:\.\d+)?)\s*m(?![m²2])',1000)]:
        nums.extend(float(x)*scale for x in re.findall(pattern,desc,re.I))
    return [str(annotation.get('questionCategory','未知')),str(annotation.get('defectLocation','未知')),str(annotation.get('defectType','未知')),
            max(nums) if nums else -1,float(bool(nums)),float('贯通' in desc),float('发展' in desc),float('修补' in desc or '处治' in desc)]

def main():
    status=ROOT/'tabular_status.json'
    while not (ROOT/'data/manifest.json').exists():
        dump(status,{'status':'waiting_manifest','time':time.time()});time.sleep(30)
    rows=json.loads((ROOT/'data/manifest.json').read_text());out=ROOT/'checkpoints/tabular_history';out.mkdir(parents=True,exist_ok=True)
    train_x=[];train_y=[];val_x=[];val_y=[]
    for row in rows:
        for a in row['annotations']:
            try:y=int(float(a['ratingScale(1-5)']))
            except (TypeError,ValueError):continue
            if not 1<=y<=5:continue
            x=facts(a)
            (train_x if row['split']=='train' else val_x).append(x)
            (train_y if row['split']=='train' else val_y).append(y)
    for depth in [6,8]:
        model=CatBoostClassifier(iterations=1200,depth=depth,learning_rate=.035,loss_function='MultiClass',random_seed=20260906,
                                thread_count=8,allow_writing_files=False,verbose=100)
        dump(status,{'status':'training_history_grade','depth':depth,'rows':len(train_y),'time':time.time()})
        model.fit(train_x,train_y,cat_features=[0,1,2])
        file=out/f'catboost_depth{depth}.cbm';model.save_model(str(file))
        pred=model.predict(val_x).reshape(-1).astype(int)
        dump(out/f'metrics_depth{depth}.json',{'status':'trained','train_rows':len(train_y),'holdout_rows':len(val_y),
             'history_grade_accuracy_given_annotation_facts':float(np.mean(pred==val_y)),'history_grade_mae_given_annotation_facts':float(np.mean(np.abs(pred-np.array(val_y)))),
             'sha256':sha(file),'限制':'输入为已有标注提取的事实；不是图片端到端分数，不是规范符合性评估。混合长度字段仅作文本数量特征，不作为工程实测裂缝宽度。'})
    dump(status,{'status':'history_grade_training_complete','normative_rule_training':'pending','end_to_end_evaluation':'pending','time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'tabular_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
