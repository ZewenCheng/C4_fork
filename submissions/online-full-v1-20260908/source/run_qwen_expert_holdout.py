"""冻结Qwen基于专家工具生成完整保留集七字段草稿；不是正式提交产物。"""
from prepare_regions import ROOT
import json,time,os,collections
from pathlib import Path
import torch
from grounding_gateway import GroundingGateway,GROUNDING_TOOLS
from qwen_tool_router import QwenToolRouter
from download_assets import dump,sha

FIELDS=['questionCategory','bridgeName','defectLocation','filename','defectType','defectDescription','ratingScale(1-5)']
class ReportGateway(GroundingGateway):
    def dispatch(self,name,args):
        if 'sample_id' in args and args['sample_id']!=self.active_sample:raise ValueError('只能查询本次报告登记图像')
        result=super().dispatch(name,args)
        # 完整专家证据已存在平台文件；上下文只给各分支排序前五候选及明确截断计数。
        def compact(value):
            if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
            if isinstance(value,list):
                if value and isinstance(value[0],dict) and 'uncalibrated_score' in value[0]:
                    return {'total_candidates':len(value),'shown_candidates':sorted(value,key=lambda x:x['uncalibrated_score'],reverse=True)[:5],'truncated':len(value)>5}
                return [compact(v) for v in value]
            return value
        return compact(result)

def main():
    torch.set_num_threads(4)
    out=ROOT/'gateway_outputs/qwen_expert_reports_v3';out.mkdir(parents=True,exist_ok=True)
    status=ROOT/'qwen_report_status.json'
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='holdout']
    config={'manifest_sha256':sha(ROOT/'data/manifest.json'),'router_sha256':sha(ROOT/'qwen_tool_router.py'),'runner_sha256':sha(Path(__file__)),'model':'/model/Qwen3.6-27B','input':'登记ID、文件名及训练后图像专家工具；不输入保留图标注或标签','scope':'专家证据驱动七字段草稿，规范覆盖未完成，不作竞赛提交','fields':FIELDS}
    cfg=out/'configuration.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    dump(cfg,config)
    gateway=ReportGateway();router=QwenToolRouter(gateway,GROUNDING_TOOLS)
    while torch.cuda.mem_get_info()[0]<64*1024**3:
        dump(status,{'status':'waiting_64gib_free_ppu','time':time.time()});time.sleep(60)
    dump(status,{'status':'loading_frozen_qwen','time':time.time()});router.load_frozen()
    counts=collections.Counter();begin=time.monotonic();processed=0
    for i,row in enumerate(rows):
        target=out/(row['sample_id']+'.json')
        if target.exists():counts[json.loads(target.read_text())['status']]+=1;continue
        gateway.active_sample=row['sample_id']
        messages=[{'role':'system','content':'你是巡检报告编排器。必须先调用inspect_semantics、inspect_regions、inspect_masks、inspect_grounding，必要时inspect_local、inspect_anomaly和query_standard。仅使用当前sample_id。工具候选和分数不是确认事实；不把分数换成等级，不把256网格面积当作原图或平方米。规范检索文本不是图像实测证据。不得根据文件名猜桥名、位置或等级。最终只输出一个七字段JSON对象，字段为'+json.dumps(FIELDS,ensure_ascii=False)+'。所有值为字符串。filename保持给定文件名；无法证实的字段用空字符串。defectDescription用中文说明候选、专家分歧和证据不足。ratingScale(1-5)只有可靠适用规范证据充分才填1至5，否则空字符串。不得声称已测出无标定的物理尺寸。'}, {'role':'user','content':json.dumps({'sample_id':row['sample_id'],'filename':Path(row['path']).name,'task':'调用专家工具并生成巡检草稿'},ensure_ascii=False)}]
        result=router.run(messages)
        final=result.get('final');events=result.get('events',[])
        required={'inspect_semantics','inspect_regions','inspect_masks','inspect_grounding'}
        called={e['name'] for e in events if e.get('status')=='ok'}
        valid=isinstance(final,dict) and set(final)==set(FIELDS) and all(isinstance(v,str) for v in final.values()) and final['filename']==Path(row['path']).name and final['ratingScale(1-5)'] in ['', '1','2','3','4','5']
        result['structural_validation']={'seven_fields_valid':valid,'required_visual_tools_called':sorted(required&called),'missing_visual_tools':sorted(required-called),'normative_coverage':'partial','not_competition_submission':True}
        dump(target,result);counts[result['status']]+=1;processed+=1
        dump(status,{'status':'generating_whole_holdout_reports','completed':i+1,'total':len(rows),'images_per_second':processed/max(time.monotonic()-begin,1),'outcomes':dict(counts),'time':time.time()})
    dump(status,{'status':'whole_holdout_drafts_complete_effectiveness_review_pending','images':len(rows),'outcomes':dict(counts),'time':time.time()})

if __name__=='__main__':
    try:main()
    except Exception as e:dump(ROOT/'qwen_report_status.json',{'status':'failed','error_type':type(e).__name__,'time':time.time()});raise
