"""独立读回330条候选，与固定v2核对桥名；仅导出聚合核验。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

ROOT = Path('/workspace/work/c4-metadata-semantics2-20260909')
OUT = ROOT / 'assembled'
V2 = Path('/workspace/work/c4-submission-versions/v2-field-complete/package/result/result.json')
BASE = Path('/workspace/work/c4-submission-versions/contract-v1-parallel2-20260909/submit/result/result.json')
INPUT = Path('/workspace/work/c4-parallel2-entry-v1-20260909/candidate_online_20260908/input_manifest.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    if (OUT / 'verification_summary.json').exists():
        raise ValueError('独立核验已固定，不覆盖')
    assert sha(V2) == 'a319a9fc1b36d0028c63832ebd0bb46500f5d00f8d6ceaa92fea860a56e378ef'
    assert sha(BASE) == 'a09293f6fcb9efe043a32ed7feb8924fbcbfc41d03f1dc5be38fecfae3634473'
    summary = read(OUT / 'repair_summary.json')
    assert sha(OUT / 'result.json') == summary['result_sha256']
    values = {'v2': read(V2), 'baseline': read(BASE), 'repaired': read(OUT / 'result.json')}
    db = sqlite3.connect(':memory:')
    for name, rows in values.items():
        assert len(rows) == 330
        db.execute('CREATE TABLE ' + name + '(domain TEXT, file TEXT, bridge TEXT, location TEXT, defect TEXT, description TEXT, grade TEXT, PRIMARY KEY(domain,file))')
        db.executemany('INSERT INTO ' + name + ' VALUES(?,?,?,?,?,?,?)', [(r['questionCategory'], r['filename'], r['bridgeName'], r['defectLocation'], r['defectType'], r['defectDescription'], r['ratingScale(1-5)']) for r in rows])
    agreed = db.execute('SELECT COUNT(*),SUM(a.bridge=b.bridge) FROM v2 a JOIN repaired b ON a.domain=b.domain AND a.file=b.file').fetchone()
    changed = db.execute('SELECT COUNT(*),SUM(a.bridge<>b.bridge),SUM(a.location<>b.location),SUM(a.defect<>b.defect),SUM(a.description<>b.description),SUM(a.grade<>b.grade) FROM baseline a JOIN repaired b ON a.domain=b.domain AND a.file=b.file').fetchone()
    assert agreed == (330, 330)
    assert changed == (330, 330, 330, 0, 0, 0)
    rows = read(INPUT)
    assert [(r['questionCategory'], r['filename']) for r in values['baseline']] == [(r['questionCategory'], r['filename']) for r in values['repaired']]
    sources, certainties, names = Counter(), Counter(), Counter()
    filename_identifiers_preserved = 0
    for row, result in zip(rows, values['repaired']):
        assert result['filename'] == Path(row['path']).name
        assert result['bridgeName'] not in {'桥梁', '轨道'}
        assert result['defectLocation'] and '图像' not in result['defectLocation'] and result['defectLocation'] not in {'全桥', '桥梁结构'}
        record = read(OUT / 'reports/records' / (row['sample_id'] + '.json'))
        assert record['final'] == result
        location = record['locked_diagnosis']['location_metadata']
        sources[location['source']] += 1
        certainties[location['certainty']] += 1
        names[result['bridgeName']] += 1
        if location['source'] in {'文件名结构部位', '文件名编号'}:
            stem = re.sub(r'^(?:左右幅|左幅|右幅)[_-]?', '', Path(row['path']).stem).replace('_', '-')
            stem = ' '.join(stem.split())
            assert result['defectLocation'].split('（文件名编号')[0] == stem
            filename_identifiers_preserved += 1
        else:
            assert '未知' in result['defectLocation']
            if location['certainty'] == 'type_inference':
                assert '按病害类型推断' in result['defectLocation']
    assert dict(sources) == summary['位置来源计数'] and dict(certainties) == summary['位置确定性计数']
    assert filename_identifiers_preserved == 47, '既有文件名结构部位尚未完整保留'
    # 原目录、原快照均保持53.36版，候选结果不冒充已发布制品。
    assert sha(Path('/workspace/result/result/result.json')) == sha(BASE)
    timing = read(OUT / 'repair_timing.json')
    assert timing['repair_time_seconds'] > 0 and timing['new_full_inference_time_seconds'] is None
    result = {'状态': '独立SQL及结构位置语义核验通过', '图数': 330, '桥名与v2一致数': agreed[1],
              '仅桥名和位置改变': True, '文件名部位或编号保留数': filename_identifiers_preserved,
              '位置来源计数': dict(sources), '位置确定性计数': dict(certainties), '桥名分布': dict(names),
              '逐字段变化数': dict(zip(['bridgeName', 'defectLocation', 'defectType', 'defectDescription', 'ratingScale(1-5)'], changed[1:])),
              '旧正式结果未修改': True, 'result_sha256': sha(OUT / 'result.json'),
              'repair_summary_sha256': sha(OUT / 'repair_summary.json'), 'repair_timing_sha256': sha(OUT / 'repair_timing.json'),
              '计时': timing, '核验时间': time.time()}
    path = OUT / 'verification_summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'核验': result, 'verification_summary_sha256': sha(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
