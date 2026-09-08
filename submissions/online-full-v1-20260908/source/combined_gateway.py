"""统一已就绪的视觉分类、异常距离和规范证据工具；其余专家持续接入。"""
import json,uuid
from visual_gateway import VisualGateway,VISUAL_TOOLS,ROOT

COMBINED_TOOLS=VISUAL_TOOLS+[{'type':'function','function':{'name':'inspect_anomaly','description':'读取已由冻结7B和参考库计算的局部距离证据。仅支持已完成图像；距离不是病害概率、面积或等级。','parameters':{'type':'object','properties':{'sample_id':{'type':'string'}},'required':['sample_id'],'additionalProperties':False}}}]

class CombinedGateway(VisualGateway):
    def __init__(self):
        super().__init__();self.anomaly_evidence={}
        self.bank_config=json.loads((ROOT/'checkpoints/anomaly_banks/configuration.json').read_text())
    def dispatch(self,name,args):
        if name!='inspect_anomaly':return super().dispatch(name,args)
        if not isinstance(args,dict) or set(args)!={'sample_id'}:raise ValueError('异常工具只接受登记样本ID')
        sample_id=args['sample_id']
        if not isinstance(sample_id,str) or sample_id not in self.samples:raise ValueError('样本ID未登记')
        path=ROOT/'gateway_outputs/anomaly_holdout'/(sample_id+'.json')
        if not path.exists():raise FileNotFoundError('该图尚未完成冻结教师异常推理')
        packet=json.loads(path.read_text());result=packet['result']
        if result['sample_id']!=sample_id:raise ValueError('缓存样本身份不符')
        for bank,scores in result['scores'].items():
            if scores['bank_sha256']!=self.bank_config['banks'][bank]['sha256']:raise ValueError('异常参考库版本不符')
        evidence_id=uuid.uuid4().hex;result['evidence_id']=evidence_id
        self.anomaly_evidence[evidence_id]=result
        return packet
