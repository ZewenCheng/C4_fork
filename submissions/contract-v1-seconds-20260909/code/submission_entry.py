"""提交版入口：核验发布源码，在官方work创建独立执行目录并成对发布双JSON。"""
import time
begin_time = time.time()

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

WORKSPACE_BOUNDARY = Path('/workspace')
WORK_BOUNDARY = WORKSPACE_BOUNDARY / 'work'
DATA_BOUNDARY = Path('/dataset')
TEMPLATE_ROOT = '/workspace/work/c4-contract-entry-v3-20260908'
ASSET_ROOT = Path('/workspace/work/c4-experiments/expert-autotrain-20260906')
DEFAULT_INPUT = '/dataset/决赛数据集1/赛题4/线上测试集'
ASSETS = ['models', 'checkpoints', 'data', 'standards', 'hplus_distilled_features',
          'wemm_features', 'legacy_convnext_features', 'deps']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sources_for_work(code, work, inp):
    manifest = read(code / 'source_manifest.json')
    sources = {}
    for name, expected in manifest['files'].items():
        if Path(name).name != name or not name.endswith('.py') or digest(code / name) != expected:
            raise ValueError('发布源码清单或摘要不符')
        text = (code / name).read_text(encoding='utf-8').replace(TEMPLATE_ROOT, work.as_posix())
        if name == 'start_online_candidate.py':
            text = text.replace(repr(DEFAULT_INPUT), repr(inp.as_posix()))
        compile(text, name, 'exec')
        sources[name] = text
    if not {'start_online_candidate.py', 'online_full_pipeline.py', 'report_contract.py', 'run_contract_reports.py'} <= sources.keys():
        raise ValueError('发布源码缺少执行入口')
    return sources


def prepare(code, work, inp, resume=False, check_only=False):
    work, inp = work.resolve(), inp.resolve()
    if work == WORK_BOUNDARY.resolve() or not work.is_relative_to(WORK_BOUNDARY.resolve()):
        raise ValueError('运行目录必须位于官方work的独立子目录')
    if not inp.is_dir() or not inp.is_relative_to(DATA_BOUNDARY.resolve()):
        raise ValueError('输入必须位于官方dataset目录')
    sources = sources_for_work(code, work, inp)
    asset_paths = {name: str((ASSET_ROOT / name).resolve()) for name in ASSETS}
    if not all(Path(path).is_dir() for path in asset_paths.values()):
        raise ValueError('缺少已部署的外置资产目录')
    signature = read(code / 'model_manifest.json')
    for item in signature['files']:
        path = Path(item['path'])
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError('外置资产不存在或大小变化')
    for name, expected in signature['qwen_metadata']['metadata_sha256'].items():
        if digest(Path(signature['qwen_metadata']['model']) / name) != expected:
            raise ValueError('冻结Qwen元数据版本不符')
    receipt = {'protocol': 'c4-submission-contract-v1', 'input': str(inp), 'work': str(work),
               'entry_sha256': digest(code / 'submission_entry.py'),
               'source_manifest_sha256': digest(code / 'source_manifest.json'),
               'model_manifest_sha256': digest(code / 'model_manifest.json'),
               'source_sha256': {name: hashlib.sha256(text.encode('utf-8')).hexdigest() for name, text in sources.items()},
               'assets': asset_paths}
    if work.exists():
        if not resume or not work.is_dir() or read(work / 'submission_entry_receipt.json') != receipt:
            raise ValueError('已有工作目录不属于同一输入及发布版本，禁止混用')
        for name, expected in receipt['source_sha256'].items():
            if digest(work / name) != expected:
                raise ValueError('恢复工作源码发生变化')
        if any(str((work / name).resolve()) != path for name, path in asset_paths.items()):
            raise ValueError('恢复资产映射发生变化')
    elif resume:
        raise ValueError('恢复目录不存在')
    if check_only:
        return work, receipt
    if not work.exists():
        work.mkdir(parents=True)
        for name, path in asset_paths.items():
            (work / name).symlink_to(path, target_is_directory=True)
        for name, text in sources.items():
            (work / name).write_text(text, encoding='utf-8')
        write(work / 'submission_entry_receipt.json', receipt)
    return work, receipt


def publish(work, output, elapsed=None, *, started_at=None):
    if (elapsed is None) == (started_at is None):
        raise ValueError('必须指定一次实测秒数或本次脚本开始时间')
    value = elapsed if elapsed is not None else started_at
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('运行计时无效')
    online = work / 'candidate_online_20260908'
    state = read(online / 'contract_reports/status.json')
    result = online / 'result.json'
    values = read(result)
    if state['status'] != 'complete' or state['images'] != len(values) or state['result_sha256'] != digest(result):
        raise ValueError('运行结果与完成凭证不符，不发布')
    output = output.resolve()
    if not output.is_relative_to(WORKSPACE_BOUNDARY.resolve()) or output == WORKSPACE_BOUNDARY.resolve():
        raise ValueError('输出必须位于官方workspace子目录')
    immutable = WORK_BOUNDARY / 'c4-submission-versions'
    if output.is_relative_to(immutable.resolve()) or output.is_relative_to(Path('/workspace/work/c4-preliminary-contract-v1-20260908').resolve()):
        raise ValueError('禁止发布到已固定版本目录')
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / 'result.json', output / 'infer_time.json']
    if any(target.exists() for target in targets):
        archive = work / 'previous_published_results' / uuid.uuid4().hex
        archive.mkdir(parents=True)
        for target in targets:
            if target.exists():
                shutil.copy2(target, archive / target.name)
                if digest(target) != digest(archive / target.name):
                    raise ValueError('旧结果备份不一致')
    # 两个文件先准备完整，再依次替换；最后用发布凭证确认成对摘要。
    pending_result = output / 'result.json.pending'
    shutil.copyfile(result, pending_result)
    if digest(pending_result) != state['result_sha256']:
        raise ValueError('新结果复制摘要不一致')
    # 结果校验、旧输出保留及新结果准备均计入；序列化计时文件必须在采样之后。
    end_time = time.time()
    infer_time = end_time - started_at if started_at is not None else elapsed
    if not math.isfinite(infer_time) or infer_time <= 0:
        raise ValueError('运行计时无效')
    write(output / 'infer_time.json.pending', {'infer_time': infer_time})
    os.replace(pending_result, targets[0])
    os.replace(output / 'infer_time.json.pending', targets[1])
    published = {'status': 'complete', 'images': len(values), 'result_sha256': digest(targets[0]),
                 'infer_time_sha256': digest(targets[1]), 'infer_time': infer_time, 'infer_time_unit': 'seconds',
                 'output': str(output), 'scope': '本次脚本开始至结果准备结束的秒数，包含准备和模型加载；恢复只计本次调用，不累计停机间隔'}
    write(work / 'published_result_receipt.json', published)
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/workspace/result/result'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true', help='只核验源码及外置资产可用性，不执行模型')
    args = parser.parse_args()
    if args.resume and args.work is None:
        parser.error('恢复必须明确--work')
    code = Path(__file__).resolve().parent
    work = args.work or WORK_BOUNDARY / ('c4-contract-run-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    work, receipt = prepare(code, work, Path(args.input), args.resume, args.check)
    if args.check:
        print(json.dumps({'状态': '提交入口自检通过', '源码文件数': len(receipt['source_sha256']),
                          '模型执行': False, '说明': '资产大小及Qwen元数据核验；完整资产SHA在封装时独立核验'}, ensure_ascii=False))
        return
    sys.path.insert(0, str(work))
    from report_contract import run_lock
    with run_lock(WORK_BOUNDARY / 'c4-submission-run.lock'), run_lock(work / 'entry.lock'):
        state_path = work / 'entry_execution.json'
        old = read(state_path) if state_path.exists() else {}
        state = {'status': 'running', 'first_started_at': old.get('first_started_at', time.time()),
                 'attempt_started_at': begin_time, 'attempt': old.get('attempt', 0) + 1, 'pid': os.getpid()}
        write(state_path, state)
        try:
            for name in ['start_online_candidate.py', 'online_full_pipeline.py']:
                subprocess.run([sys.executable, '-B', '-u', str(work / name)], cwd=work,
                               env=dict(os.environ, PYTHONPATH=str(work), TABPFN_DISABLE_TELEMETRY='1'), check=True)
            published = publish(work, args.output, started_at=begin_time)
            state.update(status='complete', result_sha256=published['result_sha256'], elapsed_wall_seconds=published['infer_time'])
        except BaseException:
            state.update(status='failed', stopped_at=time.time())
            write(state_path, state)
            raise
        write(state_path, state)
    print(json.dumps({'状态': '本次独立推理与双JSON发布完成', **published}, ensure_ascii=False))


if __name__ == '__main__':
    main()
