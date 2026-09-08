"""将训练后SAM候选接入统一工具协议；网格面积不转换为物理面积。"""
import json,uuid,math
import numpy as np
from region_gateway import RegionGateway,REGION_TOOLS,ROOT
from download_assets import sha

MASK_TOOLS=REGION_TOOLS+[{'type':'function','function':{'name':'inspect_masks','description':'读取训练后SAM的掩码候选及来源；面积是256网格内像素数，不能直接用于规范物理阈值。','parameters':{'type':'object','properties':{'sample_id':{'type':'string'},'query':{'type':'string'}},'required':['sample_id'],'additionalProperties':False}}}]

class MaskGateway(RegionGateway):
    def __init__(self):
        super().__init__();self.mask_evidence={}
    def dispatch(self,name,args):
        if name!='inspect_masks':return super().dispatch(name,args)
        if not isinstance(args,dict) or set(args)-{'sample_id','query'}:raise ValueError('掩码参数无效')
        sid=args.get('sample_id');query=args.get('query')
        if not isinstance(sid,str) or sid not in self.samples:raise ValueError('未登记样本')
        if query is not None and not isinstance(query,str):raise ValueError('提示必须为字符串')
        outputs={};pending=[]
        for optimizer in ['adamw','sgd_momentum']:
            folder=ROOT/'gateway_outputs/sam_holdout'/optimizer
            file=folder/(sid+'.json')
            if not (folder/'complete.json').exists() or not file.exists():pending.append(optimizer);continue
            source=json.loads((folder/'source.json').read_text());receipt=json.loads((folder/'complete.json').read_text());packet=json.loads(file.read_text())
            if query is not None and query not in source['queries']:raise ValueError('提示不在当前专家词表')
            if packet['sample_id']!=sid or packet['checkpoint_sha256']!=receipt['checkpoint_sha256'] or packet['checkpoint_sha256']!=source['checkpoint_sha256']:raise ValueError('掩码版本不符')
            checkpoint=ROOT/'checkpoints'/('sam3_'+optimizer)/'last.pt'
            if sha(checkpoint)!=packet['checkpoint_sha256'] or sha(ROOT/'data/vocabulary.json')!=source['vocabulary_sha256']:raise ValueError('掩码权重或词表变更')
            if sha(ROOT/'models/sam3/source_manifest.json')!=source['base_manifest_sha256']:raise ValueError('掩码基底变更')
            maskfile=folder/(sid+'.npz')
            if packet['mask_file']!=maskfile.name or sha(maskfile)!=packet['mask_sha256']:raise ValueError('掩码文件校验失败')
            with np.load(maskfile,allow_pickle=False) as archive:masks=archive['masks']
            if masks.dtype!=np.uint8 or masks.shape!=(len(packet['candidates']),8192):raise ValueError('掩码压缩格式不符')
            selected=[]
            for candidate in packet['candidates']:
                index=candidate['mask_index'];score=candidate['uncalibrated_score'];box=candidate['box_normalized_xyxy']
                if type(index) is not int or not 0<=index<len(masks):raise ValueError('掩码索引无效')
                if candidate['query'] not in source['queries'] or not math.isfinite(score) or not 0<=score<=1:raise ValueError('掩码候选无效')
                if len(box)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in box) or box[0]>box[2] or box[1]>box[3]:raise ValueError('掩码框无效')
                if int(np.unpackbits(masks[index]).sum())!=candidate['mask_grid_area_pixels']:raise ValueError('掩码面积校验失败')
                if query is None or query==candidate['query']:selected.append(candidate)
            outputs[optimizer]={'checkpoint_sha256':packet['checkpoint_sha256'],'mask_sha256':packet['mask_sha256'],'mask_grid_hw':[256,256],'original_size_hw':packet['original_size_hw'],'candidates':selected,'limitations':packet['limitations']}
        if not outputs:return {'protocol_version':1,'tool':name,'status':'not_ready','result':None,'pending_optimizers':pending}
        eid=uuid.uuid4().hex
        result={'evidence_id':eid,'sample_id':sid,'evidence_type':'sam_mask_candidates','outputs':outputs,'pending_optimizers':pending,'measurement':None,'rating':None}
        self.mask_evidence[eid]=result
        return {'protocol_version':1,'tool':name,'status':'partial' if pending else 'ok','result':result}
