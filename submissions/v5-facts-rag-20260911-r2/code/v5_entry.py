"""v5：伙伴多视图分类、事实评级/描述和两来源RAG的独立提交入口。"""
import time
BEGIN = time.time()
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import uuid

VERSION = 'v5-facts-rag-20260911'
GRADE_ROOT = Path('/workspace/work/c4-final-fact-grade-20260910-v1')
GRADE_SHA = 'a41100cfca5dd9650f777589d512ae906e94b241ca4c597c59cab153464b0f60'
BASE_ROOT = Path('/workspace/work/road-infrastructure-finals-cloud/models/transformers/convnextv2_large')
HEAD_ROOT = Path('/workspace/work/c4-cv-multiview-final-20260910')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)

def scan_inputs(root):
    root = Path(root).resolve(strict=True)
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError('图像链接超出输入目录')
        relative = path.relative_to(root)
        categories = set(relative.parts) & {'桥梁', '轨道'}
        if len(categories) != 1:
            raise ValueError('输入图像缺少唯一桥梁/轨道目录')
        image_sha = sha(path)
        sid = hashlib.sha256((relative.as_posix() + '\0' + image_sha).encode()).hexdigest()
        rows.append({'sample_id': sid, 'path': str(path), 'image_sha256': image_sha,
                     'category': next(iter(categories))})
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('输入为空或身份重复')
    return rows

def verify_code(code):
    manifest = json.loads((code / 'source_manifest.json').read_text(encoding='utf-8'))
    actual = {str(p.relative_to(code)).replace('\\', '/'): sha(p)
              for p in code.rglob('*') if p.is_file() and p.name != 'source_manifest.json'
              and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual != manifest:
        raise ValueError('提交源码集合或摘要变化')
    return sha(code / 'source_manifest.json')

def assemble(diagnosis, grade, description):
    from report_contract import FIELDS, validate_final
    final = {k: diagnosis[k] for k in ('questionCategory', 'bridgeName', 'filename', 'defectType')}
    final.update(defectLocation=diagnosis['location_metadata']['value'],
                 defectDescription=description['text'])
    final['ratingScale(1-5)'] = grade
    validate_final(final, diagnosis)
    return {k: final[k] for k in FIELDS}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('/dataset/决赛数据集1/赛题4/线上测试集'))
    p.add_argument('--work', type=Path)
    p.add_argument('--output', type=Path, default=Path('/workspace/result/result'))
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    code = Path(__file__).resolve().parent
    source_sha = verify_code(code)
    from research_rag_adapter import REQUIRED_SYS_PATH
    import sys
    for path in reversed(REQUIRED_SYS_PATH):
        if path not in sys.path:
            sys.path.insert(0, path)
    from inference_clock import InferenceClock
    from fact_grade_predictor import FactGradePredictor
    from research_rag_adapter import load_research_rag
    from fact_protocol import facts_from_prediction
    from grounded_description import describe_facts, validate_description
    from report_contract import locked_diagnosis
    out = a.output.resolve()
    if not out.is_relative_to('/workspace') or out == Path('/workspace'):
        raise ValueError('输出必须位于平台工作区子目录')
    work = (a.work or Path('/workspace/work') / ('c4-v5-run-' + uuid.uuid4().hex)).resolve()
    if not work.is_relative_to('/workspace/work') or work == Path('/workspace/work'):
        raise ValueError('工作目录必须位于平台work子目录')
    rows = scan_inputs(a.input)
    if a.check:
        clock = InferenceClock({'version': VERSION, 'mode': 'preflight'})
        rag = load_research_rag(clock)
        grader = FactGradePredictor(GRADE_ROOT/'grade_model.joblib', GRADE_ROOT/'manifest.json', expected_manifest_sha256=GRADE_SHA)
        for asset in [BASE_ROOT/'model.safetensors', BASE_ROOT/'config.json', BASE_ROOT/'preprocessor_config.json', HEAD_ROOT/'multiview_model.npz', HEAD_ROOT/'manifest.json']:
            if not asset.is_file(): raise ValueError('视觉资产缺失')
        print(json.dumps({'状态': 'v5入口预检通过', '输入数': len(rows), '源码摘要': source_sha,
                          '评级模型': grader.model_sha256, 'RAG来源': rag.fingerprint}, ensure_ascii=False))
        return
    if any((out/name).exists() for name in ('result.json', 'infer_time.json')):
        raise FileExistsError('输出已存在；请使用新的独立目录，禁止覆盖历史结果')
    work.mkdir(parents=True, exist_ok=False)
    config = {'version': VERSION, 'mode': 'user_selected_release', 'release_authorized': True,
              'quality_gain_verified': False, 'images': len(rows), 'input_root': str(a.input.resolve()),
              'source_manifest_sha256': source_sha, 'grade_manifest_sha256': GRADE_SHA,
              'timing_scope': 'model_inference_only',
              'policy': {'vision_batch': 4, 'qwen_generation': False, 'legacy_prediction_cache': False,
                         'rag_role': 'knowledge_sidecar_not_grade'}}
    dump(work/'input_manifest.json', rows)
    config['manifest_sha256'] = sha(work/'input_manifest.json')
    dump(work/'configuration.json', config)
    clock = InferenceClock(config, journal=work/'model_intervals.json')
    try:
        rag = load_research_rag(clock)
        grader = FactGradePredictor(GRADE_ROOT/'grade_model.joblib', GRADE_ROOT/'manifest.json', expected_manifest_sha256=GRADE_SHA)
        from fast_vision_runtime import FastVisionRuntime
        vision = FastVisionRuntime(BASE_ROOT, HEAD_ROOT, clock, batch_size=4)
        dump(work/'assets.json', {'vision': vision.source, 'grade_manifest_sha256': grader.manifest_sha256,
             'grade_model_sha256': grader.model_sha256, 'rag': rag.assets, 'rag_fingerprint': rag.fingerprint})
        finals = []
        for row, pred in vision.predict_many(rows):
            sid = row['sample_id']
            packet = {'sample_id': sid, 'category': row['category'], 'predicted_type': pred['candidates'][0]['label'],
                      'prediction_source': 'inspect_cv_multiview/multiview/top1'}
            diagnosis = locked_diagnosis(row, packet)
            facts = facts_from_prediction(pred, category=row['category'], metadata={'structure_location': diagnosis['location_metadata']}, sample_id=sid)
            knowledge = rag.lookup(facts, pred, diagnosis['is_intact'])
            if diagnosis['is_intact']:
                grade = {'predicted_grade': '', 'model_version': None, 'mode': 'intact_policy'}
            else:
                prepared = grader.prepare(facts)
                with clock.model('fact_grade_head', batch_size=1, model_sha256=grader.model_sha256):
                    raw = grader.predict_values(prepared)
                grade = grader.format_prediction(raw, facts)
            description = describe_facts(facts, sources=facts['sources'])
            validate_description(description, facts, facts['sources'])
            final = assemble(diagnosis, grade['predicted_grade'], description)
            finals.append(final)
            dump(work/'records'/(sid+'.json'), {'version': VERSION, 'sample_id': sid,
                 'input_sha256': row['image_sha256'], 'cv': pred, 'facts': facts, 'knowledge': knowledge,
                 'grade': grade, 'description': description, 'final': final})
            if len(finals) % 10 == 0:
                dump(work/'progress.json', {'完成': len(finals), '总数': len(rows), '纯模型秒': clock.snapshot()['infer_time']})
        timing = clock.require_complete()
        if verify_code(code) != source_sha: raise ValueError('运行期间源码发生变化')
        dump(work/'result.json', finals)
        dump(work/'infer_time.json', {'infer_time': timing['infer_time']})
        summary = {'版本': VERSION, '状态': '完整推理完成', '样本数': len(rows), '纯模型秒': timing['infer_time'],
                   '端到端秒': time.time()-BEGIN, '660秒速度门': timing['infer_time'] <= 660,
                   '评级分布': dict(collections.Counter(f['ratingScale(1-5)'] for f in finals)),
                   '结果SHA256': sha(work/'result.json'), '计时SHA256': sha(work/'infer_time.json'),
                   '质量提升已验证': False, '官方评分': None,
                   '说明': '用户指定v5封装；训练模型的研究来源与未优于全2的历史验证保持。RAG为知识旁路，不直接生成等级或图像事实。'}
        dump(work/'summary.json', summary)
        dump(work/'complete.json', {'summary_sha256': sha(work/'summary.json'), 'result_sha256': sha(work/'result.json'),
                                    'timing_sha256': sha(work/'infer_time.json')})
        # 双JSON仅在全链成功后输出；失败时不得带有完整输出凭证。
        out.mkdir(parents=True, exist_ok=True)
        dump(out/'result.json', finals)
        dump(out/'infer_time.json', {'infer_time': timing['infer_time']})
        dump(work/'output_receipt.json', {'output': str(out), 'result_sha256': sha(out/'result.json'),
                                         'timing_sha256': sha(out/'infer_time.json')})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        dump(work/'failure.json', {'状态': '未完成', '类型': type(exc).__name__, '原因': str(exc), '端到端秒': time.time()-BEGIN})
        raise

if __name__ == '__main__':
    main()
