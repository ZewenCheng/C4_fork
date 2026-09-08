"""已训练旧视觉专家的真实工具适配器；输入仅样本ID，不向模型提供标注。"""
import json,uuid
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from expert_gateway import ExpertGateway,TOOLS
from train_legacy_head import LegacyHead,ROOT
from download_assets import sha

VISUAL_TOOLS=TOOLS+[{'type':'function','function':{'name':'inspect_local','description':'调用已训练ConvNeXtV2双分类头，返回图像级候选及模型摘要。分类分数未校准，不是病害面积或评级。','parameters':{'type':'object','properties':{'sample_id':{'type':'string'},'top_k':{'type':'integer','minimum':1,'maximum':10}},'required':['sample_id'],'additionalProperties':False}}}]

class VisualGateway(ExpertGateway):
    def __init__(self):
        super().__init__();torch.set_num_threads(2)
        manifest=ROOT/'data/manifest.json';self.manifest_sha=sha(manifest)
        self.samples={r['sample_id'] for r in json.loads(manifest.read_text())}
        self.vocabulary=json.loads((ROOT/'data/vocabulary.json').read_text())
        base=ROOT/'checkpoints/legacy_convnext_head';config=json.loads((base/'config.json').read_text())
        source=json.loads((ROOT/'legacy_convnext_features/source.json').read_text())
        if source['manifest_sha256']!=self.manifest_sha:raise ValueError('特征库与输入清单版本不一致')
        if config['vocabulary_sha256']!=sha(ROOT/'data/vocabulary.json'):raise ValueError('分类头与词表版本不一致')
        self.heads={};self.digests={};self.source=source
        for optimizer in ['adamw','sgd_momentum']:
            checkpoint=base/optimizer/'last.pt';complete=json.loads((base/optimizer/'complete.json').read_text());digest=sha(checkpoint)
            if digest!=complete['checkpoint_sha256']:raise ValueError('分类头恢复文件摘要不符')
            model=LegacyHead(config['width'],config['classes']);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
            self.heads[optimizer]=model;self.digests[optimizer]=digest
        self.classification_evidence={}

    def dispatch(self,name,args):
        if name!='inspect_local':return super().dispatch(name,args)
        if not isinstance(args,dict) or set(args)-{'sample_id','top_k'}:raise ValueError('视觉参数结构无效')
        sample_id=args.get('sample_id');k=args.get('top_k',5)
        if not isinstance(sample_id,str) or sample_id not in self.samples:raise ValueError('样本不在已登记清单')
        if type(k) is not int or not 1<=k<=10:raise ValueError('候选数无效')
        file=ROOT/'legacy_convnext_features'/(sample_id+'.npz')
        if not file.exists():raise FileNotFoundError('该登记图像尚缺对应固定基底特征，不能生成假结果')
        with np.load(file,allow_pickle=False) as data:features=torch.from_numpy(data['features'].astype(np.float32))
        features=F.normalize(features,dim=-1);outputs={}
        with torch.inference_mode():
            for name_,model in self.heads.items():
                logits=torch.logsumexp(model(features),0)-np.log(3);scores=logits.softmax(-1);values,indices=scores.topk(k)
                outputs[name_]=[{'label_id':int(i),'label':self.vocabulary['raw_classes'][int(i)],'uncalibrated_score':float(v)} for i,v in zip(indices,values)]
        evidence_id=uuid.uuid4().hex
        result={'evidence_id':evidence_id,'sample_id':sample_id,'expert':'convnextv2_large_dual_head','evidence_type':'image_level_classification','vocabulary_version':self.vocabulary['version'],
                'base_sha256':self.source['base_sha256'],'checkpoint_sha256':self.digests,'feature_sha256':sha(file),'candidates':outputs,'measurement':None,'rating':None,
                'limitations':['原始组合标签写法，不是完整规范病害总表','softmax未校准，不能作为面积、严重程度或确认完好证明']}
        self.classification_evidence[evidence_id]=result
        return {'protocol_version':1,'tool':'inspect_local','status':'ok','result':result}
