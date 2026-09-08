"""将已计算的检测框候选纳入统一协议，保留来源与非量化边界。"""
import json,uuid,math
from fusion_gateway import FusionGateway,FUSION_TOOLS,ROOT
from download_assets import sha

REGION_TOOLS=FUSION_TOOLS+[{'type':'function','function':{'name':'inspect_regions','description':'读取训练后RT-DETR的图像区域候选，框为归一化坐标；不代表病害掩码或实测尺寸。','parameters':{'type':'object','properties':{'sample_id':{'type':'string'}},'required':['sample_id'],'additionalProperties':False}}}]

class RegionGateway(FusionGateway):
    def __init__(self):
        super().__init__();self.region_evidence={}
    def dispatch(self,name,args):
        if name!='inspect_regions':return super().dispatch(name,args)
        if not isinstance(args,dict) or set(args)!={'sample_id'}:raise ValueError('区域工具参数无效')
        sid=args['sample_id']
        if not isinstance(sid,str) or sid not in self.samples:raise ValueError('未登记样本')
        outputs={};pending=[]
        for optimizer in ['adamw','sgd_momentum']:
            folder=ROOT/'gateway_outputs/rtdetr_holdout'/optimizer
            file=folder/(sid+'.json');complete=folder/'complete.json'
            if not complete.exists() or not file.exists():pending.append(optimizer);continue
            packet=json.loads(file.read_text());receipt=json.loads(complete.read_text())
            if packet['sample_id']!=sid or packet['checkpoint_sha256']!=receipt['checkpoint_sha256']:raise ValueError('区域缓存身份或版本不符')
            source=ROOT/'checkpoints'/('rtdetr_'+optimizer)
            if sha(source/'last.pt')!=packet['checkpoint_sha256']:raise ValueError('检测权重版本不符')
            config=json.loads((source/'configuration.json').read_text())
            if packet['base_manifest_sha256']!=config['base_manifest_sha256']:raise ValueError('检测基底版本不符')
            for candidate in packet['candidates']:
                box=candidate['box_normalized_xyxy'];score=candidate['uncalibrated_score']
                if len(box)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in box):raise ValueError('候选坐标无效')
                if box[0]>box[2] or box[1]>box[3] or not math.isfinite(score) or not 0<=score<=1:raise ValueError('候选范围无效')
                if candidate['query'] not in config['query_vocabulary']:raise ValueError('候选词表不符')
            outputs[optimizer]=packet
        if not outputs:return {'protocol_version':1,'tool':name,'status':'not_ready','result':None,'pending_optimizers':pending}
        eid=uuid.uuid4().hex
        result={'evidence_id':eid,'sample_id':sid,'evidence_type':'ranked_detection_candidates','outputs':outputs,'pending_optimizers':pending,'measurement':None,'rating':None}
        self.region_evidence[eid]=result
        return {'protocol_version':1,'tool':name,'status':'partial' if pending else 'ok','result':result}
