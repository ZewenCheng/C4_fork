"""DINOv3-7B局部特征的PatchCore式贪心coreset；区分弱正常参考与全体新颖度库。"""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import json,re,time
import numpy as np
from download_assets import ROOT,dump,sha

def weak_normal(row):
    for a in row['annotations']:
        if str(a.get('defectType','')).strip() not in {'完好','涂装完好'}:return False
        clauses=re.split(r'[，。；;\n]',str(a.get('defectDescription','')))
        for c in clauses:
            if re.search(r'裂缝|剥落|露筋|锈蚀|渗水|破损|变形|沉降',c) and not re.search(r'无|未见|未发现|不存在',c):return False
    return True
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def coreset(features,k,seed):
    rng=np.random.default_rng(seed)
    projection=rng.normal(size=(features.shape[1],128)).astype(np.float32)/np.sqrt(128)
    reduced=normalize(features.astype(np.float32)@projection)
    # 取接近中心点作为起始点，避免起点偶然落在离群特征。
    current=int(np.argmin(((reduced-reduced.mean(0))**2).sum(1)))
    distances=np.full(len(reduced),np.inf,np.float32);chosen=[]
    for _ in range(min(k,len(reduced))):
        chosen.append(current)
        d=((reduced-reduced[current])**2).sum(1)
        distances=np.minimum(distances,d);distances[chosen]=-1
        current=int(np.argmax(distances))
    return np.array(chosen),projection
def main():
    status=ROOT/'anomaly_banks_status.json'
    while True:
        p=ROOT/'teacher_features_status.json';s=json.loads(p.read_text()) if p.exists() else {}
        if s.get('status')=='complete':break
        dump(status,{'status':'waiting_complete_frozen_teacher_features','teacher_completed':s.get('completed'),'time':time.time()});time.sleep(60)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='train']
    out=ROOT/'checkpoints/anomaly_banks';out.mkdir(parents=True,exist_ok=True)
    pools={'training_novelty':[],'weak_normal_reference':[]};ids={k:[] for k in pools}
    for row in rows:
        with np.load(ROOT/'teacher_features'/(row['sample_id']+'.npz')) as data:features=normalize(data['local_features'][0].astype(np.float32)).astype(np.float16)
        pools['training_novelty'].append(features);ids['training_novelty'].extend([row['sample_id']]*len(features))
        if weak_normal(row):pools['weak_normal_reference'].append(features);ids['weak_normal_reference'].extend([row['sample_id']]*len(features))
    report={'source_sha256':sha(ROOT/'teacher_features/source.json'),'data_manifest_sha256':sha(ROOT/'data/manifest.json'),
       'feature_model':'DINOv3-7B BF16冻结教师','feature_dimension':4096,'grid':'原图4x4，不混入水平翻转重复坐标','banks':{},
       'interpretation':'training_novelty衡量训练分布新颖度；weak_normal_reference仅弱正常候选，并非确认正常集，距离不等于病害概率或规范等级。'}
    for name,chunks in pools.items():
        if not chunks:report['banks'][name]={'status':'unavailable_no_weak_normal_candidates'};continue
        dump(status,{'status':'building_coreset','bank':name,'time':time.time()})
        x=np.concatenate(chunks);image_ids=np.array(ids[name]);trimmed=0
        if name=='weak_normal_reference':
            # 只在弱正常候选内部按图均值距离剔除最外围10%；不声称该步骤证明无病害。
            image_features=normalize(np.stack([c.astype(np.float32).mean(0) for c in chunks]))
            center=normalize(image_features.mean(0,keepdims=True))[0];dist=1-image_features@center
            keep=dist<=np.quantile(dist,.9);mask=np.repeat(keep,16);trimmed=int((~keep).sum());x=x[mask];image_ids=image_ids[mask]
        selected,projection=coreset(x,512,20260906)
        filename=out/(name+'.npz');temp=filename.with_name(filename.name+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,features=x[selected],source_image_ids=image_ids[selected],projection=projection,seed=np.array(20260906))
        os.replace(temp,filename)
        report['banks'][name]={'status':'built','candidate_patches':len(x),'coreset_size':len(selected),'trimmed_weak_images':trimmed,'sha256':sha(filename)}
    dump(out/'configuration.json',report)
    dump(status,{'status':'banks_built_overall_inference_pending','banks':report['banks'],'time':time.time()})
if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'anomaly_banks_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
