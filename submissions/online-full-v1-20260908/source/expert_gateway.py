"""异构专家协议：受控证据引用、规范检索与面积规则工具。

训练模型视觉入口和Qwen调用循环仍需接入；不声称本模块已经接管生产推理。
"""
import math,time,uuid,hashlib,json
from pathlib import Path

FAMILIES=['honeycomb_pitting','spalling_corner_loss','cavity_hole']
TOOLS=[
 {'type':'function','function':{'name':'query_standard','description':'检索公开规范原文，返回PDF页码；相关度不是评级。','parameters':{'type':'object','properties':{'query':{'type':'string'},'standard':{'type':'string','enum':['JTG_T_H21_2011','CJJ_99_2017']},'limit':{'type':'integer','minimum':1,'maximum':10}},'required':['query'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'assess_area_indicator','description':'读取已有专业工具证据，返回已核定H21面积表量化命中集合。不能直接给整桥等级。','parameters':{'type':'object','properties':{'evidence_id':{'type':'string'},'indicator':{'type':'string','enum':FAMILIES}},'required':['evidence_id','indicator'],'additionalProperties':False}}}
]

class ExpertGateway:
    def __init__(self):
        self._evidence={};self._rulebook=None

    def register_area_evidence(self,*,sample_id,expert,model_digest,scope,indicator,defect_present,ratio=None,area_m2=None,measurement_level='unscaled',uncertainty=None):
        """仅由可信专业工具适配器调用，不暴露为Qwen工具。"""
        if not all(isinstance(x,str) and x for x in [sample_id,expert,model_digest]):raise ValueError('证据缺少来源')
        if indicator not in FAMILIES:raise ValueError('指标未在当前规则范围')
        if defect_present is not None and type(defect_present) is not bool:raise ValueError('病害存在状态必须为布尔或未知')
        for value,maximum in [(ratio,1),(area_m2,None)]:
            if value is not None and (type(value) not in [int,float] or not math.isfinite(value) or value<0 or (maximum is not None and value>maximum)):raise ValueError('面积输入必须为有效量值')
        if measurement_level not in ['calibrated','estimated','unscaled']:raise ValueError('量化证据层级无效')
        key=uuid.uuid4().hex
        self._evidence[key]={'sample_id':sample_id,'expert':expert,'model_digest':model_digest,'scope':scope,'indicator':indicator,'defect_present':defect_present,
            'cumulative_component_area_ratio':ratio,'max_single_area_m2':area_m2,'measurement_level':measurement_level,'uncertainty':uncertainty,'created_at':time.time()}
        return key

    def dispatch(self,name,args):
        if not isinstance(args,dict):raise ValueError('工具参数必须为对象')
        if name=='query_standard':
            if set(args)-{'query','standard','limit'}:raise ValueError('未知工具参数')
            if not isinstance(args.get('query'),str) or not 1<=len(args['query'])<=1000:raise ValueError('规范查询长度无效')
            if args.get('standard') not in [None,'JTG_T_H21_2011','CJJ_99_2017']:raise ValueError('未知规范')
            if type(args.get('limit',5)) is not int or not 1<=args.get('limit',5)<=10:raise ValueError('结果条数无效')
            from query_standard import query_standard
            result=query_standard(**args)
        elif name=='assess_area_indicator':
            if set(args)!={'evidence_id','indicator'}:raise ValueError('必须只提供证据ID与指标')
            if not isinstance(args['evidence_id'],str) or args['evidence_id'] not in self._evidence:raise ValueError('证据ID不存在，不接受模型编造的数值')
            record=self._evidence[args['evidence_id']]
            if args['indicator']!=record['indicator']:raise ValueError('指标与证据不一致')
            result=self._assess(record,args['evidence_id'])
        else:raise ValueError('未知工具，不能执行任意命令或路径')
        return {'protocol_version':1,'tool':name,'status':'ok','result':result}

    def _assess(self,record,key):
        from train_normative_area import quantitative_candidates,ROOT
        if self._rulebook is None:self._rulebook=json.loads((ROOT/'checkpoints/normative_area_v1/rulebook.json').read_text())
        result={'evidence_id':key,'sample_id':record['sample_id'],'source_expert':record['expert'],'source_model_digest':record['model_digest'],
            'assessment_level':'indicator_quantitative_candidates','final_rating':None,'standard':'JTG/T H21—2011','table':self._rulebook['tables'][record['indicator']],
            'pdf_sha256':self._rulebook['source_sha256'],'measurement_level':record['measurement_level'],'uncertainty':record['uncertainty']}
        if record['scope']!='concrete_beam_superstructure':
            return dict(result,decision='outside_verified_scope',candidates=[])
        if record['defect_present'] is None:return dict(result,decision='presence_unknown',candidates=[])
        ratio=record['cumulative_component_area_ratio'];area=record['max_single_area_m2']
        # 无物理尺度的像素面积不得代入平方米阈值；合法构件面积比例仍可单独使用。
        if record['measurement_level']=='unscaled':area=None
        candidate=quantitative_candidates(record['indicator'],int(record['defect_present']),-1 if ratio is None else ratio,-1 if area is None else area)
        candidates=[int(x) for x in candidate.split('+')] if candidate[0].isdigit() else []
        result.update(decision=candidate,candidates=candidates,requires_qualitative_and_component_context=True)
        if record['measurement_level']=='estimated':result['conditional_on_scale_estimate']=True
        return result
