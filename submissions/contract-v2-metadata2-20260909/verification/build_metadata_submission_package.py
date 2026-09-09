"""固定已核验的元数据修复结果及历史秒计时，不运行模型，不执行比赛提交。"""
import collections
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

VERSION = 'contract-v2-metadata2-20260909'
RUN = Path('/workspace/work/c4-metadata-semantics2-20260909')
ENTRY = RUN / 'prepared-entry'
BASE = Path('/workspace/work/c4-submission-versions/contract-v1-parallel2-20260909')
PREP = Path('/workspace/work/c4-release-prep-' + VERSION)
DEST = Path('/workspace/work/c4-submission-versions') / VERSION
BUNDLE = Path('/workspace/work/c4-' + VERSION + '-code-design.zip')
RESULT_SHA = 'ecc86813d9d5c9eeccb1a7942a98e07b717f7e77d9ac5102acd5009e4d7bc1d0'
TIME_SHA = 'de49c91f6dc294f670d23ab4c94bc9b053d060cbf4982a84b68c070658c7702a'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(value, why):
    if not value:
        raise ValueError(why)


def main():
    expected = json.load(sys.stdin)
    require(not DEST.exists(), '固定版本已存在，禁止覆盖')
    require(sha(BUNDLE) == expected['bundle_sha256'], '传输包摘要不符')
    evidence = {
        'repair_summary.json': RUN / 'assembled/repair_summary.json',
        'verification_summary.json': RUN / 'assembled/verification_summary.json',
        'code_validation.json': RUN / 'code_validation.json',
    }
    for name, path in evidence.items():
        require(sha(path) == expected['evidence_sha256'][name], '修复证据变化：' + name)
    repair = read(evidence['repair_summary.json'])
    verified = read(evidence['verification_summary.json'])
    validation = read(evidence['code_validation.json'])
    require(repair['新模型调用次数'] == 0 and repair['新凭证恢复数'] == 330, '修复与恢复证据不符')
    require(verified['桥名与v2一致数'] == 330 and verified['文件名部位或编号保留数'] == 47, '语义核验不符')
    require(validation['测试数'] == 53 and validation['退出码'] == 0, '源码回归不符')
    candidate = PREP / 'submit-candidate'
    require(not candidate.exists(), '封装暂存目录已存在，请先检查')
    candidate.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE) as archive:
        items = archive.infolist()
        require(len(items) == 30 and len({i.filename for i in items}) == 30, '源码设计书成员数或唯一性不符')
        for item in items:
            rel = PurePosixPath(item.filename)
            require(len(rel.parts) == 2 and rel.parts[0] in {'code', 'design'} and '..' not in rel.parts and not rel.is_absolute(), '传输路径越界')
            require(not item.is_dir() and item.file_size < 10_000_000, '传输成员类型或大小不符')
            path = candidate / str(rel)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(archive.read(item))
    code = candidate / 'code'
    (code / 'run').chmod(0o755)
    require({p.name for p in code.iterdir()} == {p.name for p in (RUN / 'code').iterdir()}, '源码集合不同于修复版')
    for path in code.iterdir():
        require(sha(path) == sha(RUN / 'code' / path.name), '封装源码变化：' + path.name)
    models = read(code / 'model_manifest.json')
    backups_checked = 0
    for item in models['files']:
        path = Path(item['path'])
        require(path.resolve().is_relative_to(Path('/workspace/work')) and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], '模型文件变化')
        if item.get('backup'):
            backup = Path(item['backup'])
            require(backup.resolve().is_relative_to(Path('/workspace/work')) and backup.stat().st_size == item['bytes'] and sha(backup) == item['sha256'], '模型恢复副本变化')
            backups_checked += 1
    for name, digest in models['qwen_metadata']['metadata_sha256'].items():
        require(sha(Path(models['qwen_metadata']['model']) / name) == digest, 'Qwen元数据变化')
    source = read(code / 'source_manifest.json')
    require(len(source['files']) == 24 and sha(code / 'source_manifest.json') == repair['source_manifest_sha256'], '推理依赖闭包不符')
    for name, digest in source['files'].items():
        require(sha(code / name) == digest, '依赖源码摘要不符')
    check = subprocess.run([str(code / 'run'), '--check'], cwd=code, capture_output=True, text=True, timeout=60)
    require(check.returncode == 0, '封装入口自检失败')
    resume = subprocess.run([str(code / 'run'), '--check', '--work', str(ENTRY), '--resume'], cwd=code, capture_output=True, text=True, timeout=60)
    require(resume.returncode == 0, '封装入口恢复自检失败')
    require(sha(RUN / 'assembled/result.json') == RESULT_SHA == repair['result_sha256'] == verified['result_sha256'], '修复结果变化')
    require(sha(BASE / 'submit/result/infer_time.json') == TIME_SHA, '历史计时文件变化')
    result = candidate / 'result'
    result.mkdir()
    shutil.copyfile(RUN / 'assembled/result.json', result / 'result.json')
    shutil.copyfile(BASE / 'submit/result/infer_time.json', result / 'infer_time.json')
    rows = read(result / 'result.json')
    timing = read(result / 'infer_time.json')
    require(timing == {'infer_time': 6568.275286197662}, '历史秒计时不符')
    fields = {'questionCategory', 'bridgeName', 'defectLocation', 'filename', 'defectType', 'defectDescription', 'ratingScale(1-5)'}
    require(len(rows) == 330 and len({(r['questionCategory'], r['filename']) for r in rows}) == 330, '输入身份重复或缺失')
    require(all(set(r) == fields and all(isinstance(v, str) for v in r.values()) for r in rows), '七字段结构不符')
    require({p.name for p in candidate.iterdir()} == {'code', 'design', 'result'}, '提交顶层不符')
    require([p.name for p in (candidate / 'design').iterdir()] == ['智能体设计方案_已核验.pdf'], '必须只有一份设计书')
    require(not any(p.is_symlink() for p in candidate.rglob('*')), '提交目录含链接')
    files = [{'path': p.relative_to(candidate).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size,
              'mode': '0755' if p == code / 'run' else '0644'} for p in sorted(candidate.rglob('*')) if p.is_file()]
    require(len(files) == 32, '提交文件数不符')
    DEST.mkdir(parents=True)
    shutil.copytree(candidate, DEST / 'submit')
    archive = DEST / ('c4-' + VERSION + '.tar.gz')
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for name in ['code', 'design', 'result']:
            item = tarfile.TarInfo(name)
            item.type = tarfile.DIRTYPE
            item.mode = 0o755
            tar.addfile(item)
        for row in files:
            item = tarfile.TarInfo(row['path'])
            item.size = row['bytes']
            item.mode = int(row['mode'], 8)
            with (candidate / row['path']).open('rb') as stream:
                tar.addfile(item, stream)
    manifest = {
        'version': VERSION, 'files': files, 'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
        'source_result': str(RUN / 'assembled/result.json'), 'source_result_sha256': RESULT_SHA,
        'submitted_infer_time_sha256': TIME_SHA, 'infer_time': timing['infer_time'], 'infer_time_unit': 'seconds',
        'timing_transformation': '按用户明确选择直接封装现有修复结果。infer_time原样沿用parallel2历史完整推理time.time实测秒差；本版未完整重跑。修复及恢复核验154.120590秒单独记录，不相加冒充一次新入口计时。',
        'timing': repair['计时'], 'timing_choice': '直接封装现有修复结果',
        'historical_inference_version': 'contract-v1-parallel2-20260909',
        'historical_inference_entry': '/workspace/work/c4-parallel2-entry-v1-20260909',
        'transfer_bundle_sha256': sha(BUNDLE), 'source_entry': str(ENTRY),
        'source_manifest_sha256': sha(code / 'source_manifest.json'), 'model_manifest_sha256': sha(code / 'model_manifest.json'),
        'evidence_sha256': {name: sha(path) for name, path in evidence.items()},
        'raw_records': str(RUN / 'assembled/reports/records'), 'input_manifest_sha256': repair['输入SHA256'],
        'verified_images': 330, 'field_summary': {f: sum(r[f] == '' for r in rows) for f in sorted(fields)},
        'grade_summary': dict(collections.Counter(r['ratingScale(1-5)'] for r in rows)),
        'execution': validation['执行参数'], 'semantic_summary': {'桥名与v2一致数': 330, '文件名结构部位数': 47, '类型推断数': 74, '未知位置数': 209, '其余五字段不变': True},
        'entry_validation': {'check': json.loads(check.stdout.strip().splitlines()[-1]), 'prepared_work_resume_check': True,
                             'full_cold_rerun': False, 'new_model_calls': 0, 'regressions': 53,
                             'model_manifest_files_sha256_checked': len(models['files']), 'model_backup_files_sha256_checked': backups_checked,
                             'qwen_metadata_sha256_checked': True},
        'github_tag': 'c4-' + VERSION, 'github_backup_gate': '发布前独立读取固定标签及版本清单；凭证见publication_receipt.json',
        'official_score': None, 'score_source': '本版未正式评测，53.36属于历史基线反馈', 'competition_submitted': False, 'created_at': time.time(),
    }
    (DEST / 'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'状态': '修复版提交包已固定，待独立验包和GitHub核验', '版本': VERSION, '压缩包': str(archive),
                      '压缩包SHA256': sha(archive), '清单SHA256': sha(DEST / 'package_manifest.json'), '文件数': 32, 'infer_time': timing}, ensure_ascii=False))


if __name__ == '__main__':
    main()
