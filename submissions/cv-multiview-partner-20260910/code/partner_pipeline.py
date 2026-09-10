"""预计算基线后的伙伴复核入口：共享模型批量、独立案件事务和秒级阶段计时。"""
import time
begin_time=time.time()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from report_contract import read_json,atomic_json,digest_json,digest_file
from partner_review import run_review
from partner_batcher import BatchReviewer
from run_contract_reports import FrozenGrader,model_signature
from v2_rail_partner import V2RailPartner
from partner_saved_regions import SavedRegions
from partner_extent import MaskExtentPartner
from partner_crop_tool import CropPartner
from partner_semantic_tool import SemanticPartner
from partner_dependency_refresh import DependencyRefresh,StandardRetriever


def main():
    p=argparse.ArgumentParser()
    for arg in ['manifest','packets','baseline','baseline-receipt','labels','output','legacy-root']:p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=3);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.workers<=8:raise ValueError('CPU案件并发数非法')
    rows=read_json(args.manifest);labels=read_json(args.labels)['raw_classes']
    packets={r['sample_id']:read_json(args.packets/(r['sample_id']+'.json')) for r in rows}
    baseline={r['filename']:r for r in read_json(args.baseline)}
    base_receipt=read_json(args.baseline_receipt)
    if base_receipt.get('status')!='complete' or base_receipt.get('result_sha256')!=digest_file(args.baseline):
        raise ValueError('伙伴基线未通过原始完成凭证校验')
    if len(baseline)!=len(rows) or set(baseline)!={Path(r['path']).name for r in rows}:raise ValueError('基线覆盖不一致')
    adapter=V2RailPartner(args.legacy_root,args.output/'v2_features',{s:s for s in labels})
    rail=[{**r,'questionCategory':'轨道'} for r in rows if packets[r['sample_id']]['category']=='轨道']
    # 首期完整调用用于避免漏路由，按需策略仅在贡献验证后启用。
    old=adapter.inspect_many(rail) if rail else {}
    regions=SavedRegions(rows,args.packets.parent,args.packets.parent.parent/'checkpoints')
    extent=MaskExtentPartner(rows,args.packets.parent/'sam_holdout')
    crops=CropPartner(rows,args.packets.parent.parent,args.output/'crop_calls')
    semantics=SemanticPartner(rows,args.packets.parent,args.packets.parent.parent)
    standards=StandardRetriever(args.packets.parent.parent/'standards/retrieval_index')
    grader=FrozenGrader('sdpa');producer={'model':model_signature('/model/Qwen3.6-27B'),'mode':'partner_full_review_gate_closed',
        'v2_version':adapter.version,'batch':args.batch_size,'workers':args.workers,'driver_sha256':digest_file(__file__),
        'batcher_sha256':digest_file(Path(__file__).parent/'partner_batcher.py'),'region_weights':regions.weights,
        'baseline_receipt_sha256':digest_file(args.baseline_receipt),'crop_version':crops.version,
        'semantic_weights':semantics.weights,'semantic_sources':semantics.sources,
        'semantic_tool_sha256':digest_file(Path(__file__).parent/'partner_semantic_tool.py'),'standard_index':standards.assets}
    with BatchReviewer(grader,args.batch_size) as reviewer:
        def process(row):
            sid=row['sample_id'];tools={'inspect_semantics':semantics,'inspect_regions':regions,'inspect_extent':extent,'inspect_crops':crops}
            if sid in old:tools['inspect_v2_rail']=lambda a:old[a['sample_id']]
            refresh=DependencyRefresh(row,packets[sid],regions=regions,extent=extent,standards=standards)
            return run_review(row,packets[sid],baseline[Path(row['path']).name],args.output/'cases'/sid,labels,tools,reviewer,
                producer=producer,refresh=refresh,grader=reviewer)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(process,rows))
    output=[r['result'] for r in results]
    atomic_json(args.output/'result.json',output)
    elapsed=time.time()-begin_time
    atomic_json(args.output/'infer_time.json',{'infer_time':elapsed})
    atomic_json(args.output/'run_receipt.json',{'status':'complete','mode':'precomputed_baseline_then_partner_review',
        'cases':len(rows),'result_sha256':digest_json(output),'infer_time':elapsed,'model_batch_metrics':grader.batch_metrics,
        'limitation':'计时从本脚本开始，包含本次旧链提取与伙伴复核；不含上游预计算基线，不冒充全架构冷启动。'})


if __name__=='__main__':main()
