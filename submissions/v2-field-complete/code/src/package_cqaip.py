#!/usr/bin/env python3
"""按赛题规范准备作品；只有真实推理及摘要核验通过后才生成正式结果包。"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from common import PACKAGE_ROOT, atomic_json, resolve_package_path, sha256_file


STAGING = Path('/workspace/work/c4_submission_staging')
ARCHIVE = Path('/workspace/work/c4_submission.tar.gz')
PUBLISH = Path('/workspace/result')
DESIGN = 'reference/initial_best/智能体设计方案_已核验.pdf'


def sources() -> dict[str, Path]:
    selected = {}
    directories = ('src', 'scripts', 'adapters')
    for directory in directories:
        for path in (PACKAGE_ROOT / directory).rglob('*'):
            if path.is_file() and not path.is_symlink() and '.local.' not in path.name:
                if path.suffix in {'.py', '.sh'}:
                    selected[path.relative_to(PACKAGE_ROOT).as_posix()] = path
    for name in (
        'config/deploy_config.json',
        'run',
    ):
        selected[name] = PACKAGE_ROOT / name
    return dict(sorted(selected.items()))


def prepare() -> None:
    temporary = STAGING.with_name(STAGING.name + '.tmp')
    if STAGING.exists() or temporary.exists():
        raise FileExistsError('已有作品准备目录，需检查内容后再决定如何更新。')
    selected = sources()
    for name in ('code', 'design', 'result'):
        (temporary / name).mkdir(parents=True)
    evidence = {'source_hashes': {}, '说明': '仅准备代码与设计；没有真实结果，不能提交。'}
    for relative, source in selected.items():
        destination = temporary / 'code' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if relative == 'run':
            destination.chmod(0o755)
        digest = sha256_file(source)
        if sha256_file(destination) != digest:
            raise RuntimeError('准备文件的副本摘要不符。')
        evidence['source_hashes'][relative] = digest
    for relative in (DESIGN,):
        source = PACKAGE_ROOT / relative
        shutil.copyfile(source, temporary / 'design' / source.name)
        evidence['source_hashes']['design/' + source.name] = sha256_file(source)
    lines = [f'{sha256_file(path)}  {path.relative_to(temporary / "code").as_posix()}' for path in sorted((temporary / 'code').rglob('*')) if path.is_file()]
    (temporary / 'code/MANIFEST.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_json(temporary / '.cqaip-staging.json', evidence)
    os.replace(temporary, STAGING)
    print(json.dumps({'状态': '代码和设计已准备，真实结果待生成', '准备目录': str(STAGING), '代码文件数量': len(selected) + 1, '设计文件数量': 1, '正式作品包已生成': False}, ensure_ascii=False))


def build(evidence_path: Path) -> None:
    evidence_path = evidence_path.resolve()
    if not evidence_path.is_relative_to(PACKAGE_ROOT / '.runtime'):
        raise ValueError('运行证据必须来自本部署的 .runtime 目录。')
    if not evidence_path.is_file():
        raise FileNotFoundError('缺少本次真实推理完成记录，不能打包正式作品。')
    run = json.loads(evidence_path.read_text(encoding='utf-8'))
    if run.get('run_kind') != 'official_c4_inference' or run.get('completed') is not True:
        raise RuntimeError('该记录不表示本次官方测试推理已经成功。')
    for field in ('manifest', 'features', 'result'):
        path = Path(run[field]).resolve()
        if not path.is_relative_to(Path('/workspace/work')) or not path.is_file() or sha256_file(path) != run[field + '_sha256']:
            raise RuntimeError('本次推理文件不存在或摘要已变化。')
    for relative, field in (('config/deploy_config.json', 'config_sha256'), ('models/MODEL_MANIFEST.json', 'model_manifest_sha256')):
        if sha256_file(resolve_package_path(relative)) != run[field]:
            raise RuntimeError('配置或模型清单已在推理后变化，需要重新验收。')
    elapsed_seconds = run.get('elapsed_seconds')
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds <= 0:
        raise RuntimeError('运行记录缺少合法的总推理时间。')
    infer_time_payload = {'infer_time': round(float(elapsed_seconds) * 1000)}
    if 'infer_time' in run or 'infer_time_sha256' in run:
        infer_time_source = Path(run.get('infer_time', '')).resolve()
        if (
            not infer_time_source.is_relative_to(Path('/workspace/work'))
            or not infer_time_source.is_file()
            or sha256_file(infer_time_source) != run.get('infer_time_sha256')
            or json.loads(infer_time_source.read_text(encoding='utf-8')) != infer_time_payload
        ):
            raise RuntimeError('本次推理计时文件不存在、摘要变化或格式不符。')
    check = subprocess.run([sys.executable, '-B', str(PACKAGE_ROOT / 'src/validate_result.py'), '--manifest', run['manifest'], '--result', run['result']], capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError('真实结果未通过七字段和文件覆盖校验。')
    if not (STAGING / '.cqaip-staging.json').is_file():
        raise RuntimeError('请先执行 prepare 准备作品代码和设计。')
    stage = json.loads((STAGING / '.cqaip-staging.json').read_text(encoding='utf-8'))
    for relative, source in sources().items():
        if sha256_file(source) != stage['source_hashes'].get(relative) or sha256_file(STAGING / 'code' / relative) != stage['source_hashes'].get(relative):
            raise RuntimeError('准备目录与当前源码不一致，需要重新准备。')
    for relative in (DESIGN,):
        path = STAGING / 'design' / Path(relative).name
        if sha256_file(path) != stage['source_hashes']['design/' + path.name] or sha256_file(PACKAGE_ROOT / relative) != sha256_file(path):
            raise RuntimeError('设计书或补充说明已变化。')
    if not PUBLISH.is_dir() or any(PUBLISH.iterdir()):
        raise FileExistsError('/workspace/result 必须存在且为空；脚本不会删除原内容。')
    result_target = STAGING / 'result/result.json'
    infer_time_target = STAGING / 'result/infer_time.json'
    temporary = ARCHIVE.with_name(ARCHIVE.name + '.tmp')
    if ARCHIVE.exists() or temporary.exists() or result_target.exists() or infer_time_target.exists():
        raise FileExistsError('已有作品或结果，不能覆盖。')
    shutil.copyfile(run['result'], result_target)
    if sha256_file(result_target) != run['result_sha256']:
        raise RuntimeError('结果副本摘要不一致。')
    atomic_json(infer_time_target, infer_time_payload)
    with tarfile.open(temporary, 'w:gz') as archive:
        for name in ('code', 'design', 'result'):
            for path in [STAGING / name, *sorted((STAGING / name).rglob('*'))]:
                if path.is_symlink():
                    raise RuntimeError('作品准备目录中出现符号链接。')
                archive.add(path, arcname=path.relative_to(STAGING).as_posix(), recursive=False)
    if temporary.stat().st_size > 5_000_000_000:
        raise RuntimeError('作品超过已知作品区总容量。')
    with tarfile.open(temporary, 'r:gz') as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        required_results = {'result/result.json', 'result/infer_time.json'}
        if {name.split('/')[0] for name in names} != {'code', 'design', 'result'} or not required_results.issubset(names):
            raise RuntimeError('压缩包目录不符合赛事规范。')
        if any(not (item.isdir() or item.isfile()) for item in members):
            raise RuntimeError('作品包包含非普通文件。')
        import hashlib
        if hashlib.sha256(archive.extractfile('result/result.json').read()).hexdigest() != run['result_sha256']:
            raise RuntimeError('压缩包中结果摘要不一致。')
        archived_time = json.loads(archive.extractfile('result/infer_time.json').read().decode('utf-8'))
        if archived_time != infer_time_payload:
            raise RuntimeError('压缩包中总推理时间格式不一致。')
    os.replace(temporary, ARCHIVE)
    for name in ('code', 'design', 'result'):
        shutil.copytree(STAGING / name, PUBLISH / name)
    receipt = {'状态': '真实结果作品包校验通过', '压缩包': str(ARCHIVE), '字节数': ARCHIVE.stat().st_size, 'sha256': sha256_file(ARCHIVE), '平台上传内容根': str(PUBLISH), '总推理毫秒数': infer_time_payload['infer_time'], '已触发平台上传或评测': False, '模型权重位置': str(resolve_package_path('models'))}
    atomic_json(PACKAGE_ROOT / '.runtime/package_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'build'))
    parser.add_argument('--run-evidence', type=Path)
    args = parser.parse_args()
    if not PACKAGE_ROOT.resolve().is_relative_to(Path('/workspace/work')):
        raise ValueError('本入口只能在官方 /workspace/work 内运行。')
    if args.action == 'prepare':
        prepare()
    else:
        if not args.run_evidence:
            parser.error('build 必须提供 --run-evidence，不能使用占位结果。')
        build(args.run_evidence)


if __name__ == '__main__':
    main()
