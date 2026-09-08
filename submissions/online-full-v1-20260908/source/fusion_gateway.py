"""将已完成的三专家融合头接入工具协议；没有权重时明确报告未就绪。"""
import json,uuid
import numpy as np
import torch
from torch.nn import functional as F
from combined_gateway import CombinedGateway,COMBINED_TOOLS,ROOT
from train_fusion import Fusion
from download_assets import sha

FUSION_TOOLS=COMBINED_TOOLS+[{'type':'function','function':{'name':'inspect_semantics','description':'调用完成训练的H+/WeMM/ConvNeXt融合头，返回图像级语义候选和专家权重，不返回评级或尺寸。','parameters':{'type':'object','properties':{'sample_id':{'type':'string'},'top_k':{'type':'integer','minimum':1,'maximum':10}},'required':['sample_id'],'additionalProperties':False}}}]

class FusionGateway(CombinedGateway):
    def __init__(self):
        super().__init__();self.fusion_evidence={}
    def dispatch(self,name,args):
        if name!='inspect_semantics':return super().dispatch(name,args)
        if not isinstance(args,dict) or set(args)-{'sample_id','top_k'}:raise ValueError('融合工具参数无效')
        sid=args.get('sample_id');k=args.get('top_k',5)
        if not isinstance(sid,str) or sid not in self.samples:raise ValueError('未登记样本')
        if type(k) is not int or not 1<=k<=10:raise ValueError('候选数量无效')
        outputs={};pending=[]
        for optimizer in ['adamw','sgd_momentum']:
            out=ROOT/'checkpoints/semantic_fusion'/optimizer
            if not (out/'complete.json').exists():pending.append(optimizer);continue
            config=json.loads((out/'configuration.json').read_text());complete=json.loads((out/'complete.json').read_text())
            if config['manifest_sha256']!=self.manifest_sha or config['vocabulary_sha256']!=sha(ROOT/'data/vocabulary.json'):raise ValueError('融合清单或词表版本不符')
            checkpoint=out/'last.pt';digest=sha(checkpoint)
            if digest!=complete['checkpoint_sha256']:raise ValueError('融合权重摘要不符')
            folders=[ROOT/'hplus_distilled_features'/optimizer,ROOT/'wemm_features'/optimizer,ROOT/'legacy_convnext_features']
            features=[];digests=[]
            for p in folders:
                if sha(p/'source.json')!=config['sources'][str(p.relative_to(ROOT))]:raise ValueError('融合特征来源发生变化')
                file=p/(sid+'.npz')
                with np.load(file,allow_pickle=False) as data:f=torch.from_numpy(data['features'].astype(np.float32))
                if f.ndim!=2 or not torch.isfinite(f).all():raise ValueError('融合特征无效')
                features.append(F.normalize(F.normalize(f,dim=-1).mean(0),dim=-1).unsqueeze(0));digests.append(sha(file))
            model=Fusion(config['widths'],config['classes'])
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
            with torch.inference_mode():
                logits,gates=model(features);values,indices=logits[0].softmax(-1).topk(min(k,config['classes']))
            outputs[optimizer]={'checkpoint_sha256':digest,'feature_sha256':digests,'expert_order':['hplus','wemm','convnext'],'expert_weights':gates[0].tolist(),'candidates':[{'label_id':int(i),'label':self.vocabulary['raw_classes'][int(i)],'uncalibrated_score':float(v)} for i,v in zip(indices,values)]}
        if not outputs:return {'protocol_version':1,'tool':name,'status':'not_ready','pending_optimizers':pending,'result':None}
        eid=uuid.uuid4().hex
        result={'evidence_id':eid,'sample_id':sid,'evidence_type':'image_level_semantic_fusion','outputs':outputs,'pending_optimizers':pending,'rating':None,'measurement':None,'limitations':['分数与门控权重未校准，不是尺寸、病害严重程度或规范等级']}
        self.fusion_evidence[eid]=result
        return {'protocol_version':1,'tool':name,'status':'partial' if pending else 'ok','result':result}
