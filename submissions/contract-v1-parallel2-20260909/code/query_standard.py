"""规范检索工具：返回来源和页码，数值规则未核定时显式保留该状态。"""
import json,sys
from functools import lru_cache
from pathlib import Path
import joblib
from scipy.sparse import load_npz
ROOT=Path('/workspace/work/c4-contract-entry-v3-20260908')

@lru_cache(maxsize=2)
def load_index(folder,signature):
    return joblib.load(folder/'vectorizer.joblib'),load_npz(folder/'matrix.npz'),json.loads((folder/'chunks.json').read_text())

def query_standard(query,standard=None,limit=5):
    folder=ROOT/'standards/retrieval_index'
    signature=tuple((p.stat().st_size,p.stat().st_mtime_ns) for p in [folder/'vectorizer.joblib',folder/'matrix.npz',folder/'chunks.json'])
    vectorizer,matrix,chunks=load_index(folder,signature)
    scores=(matrix@vectorizer.transform([query]).T).toarray().ravel()
    choices=[i for i in scores.argsort()[::-1] if scores[i]>0 and (standard is None or chunks[i]['standard']==standard)][:max(1,min(int(limit),20))]
    return {'tool':'query_standard','evidence_type':'public_standard_text','query':query,'results':[dict(chunks[i],retrieval_similarity=float(scores[i])) for i in choices],'rating_decision':None,'reason':'文本相关度不是病害置信度，也不是已核定规则判断'}

if __name__=='__main__':print(json.dumps(query_standard(' '.join(sys.argv[1:])),ensure_ascii=False))
