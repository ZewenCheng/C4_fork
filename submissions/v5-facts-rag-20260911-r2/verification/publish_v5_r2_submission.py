"""GitHub独立核验后切换作品三目录，完整保留旧版，失败回滚；不提交比赛。"""
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
def verify_packaged_timing(manifest, result_dir):
    require(manifest['timing_mode']=='model_inference_only', '计时口径不符')
    require(read(result_dir/'infer_time.json')=={'infer_time':manifest['inference_seconds']}, '打包计时与本轮不一致')
    require(0<manifest['inference_seconds']<=660, '纯模型速度未通过')

VERSION = 'v5-facts-rag-20260911-r2'
WORK = Path('/workspace/work')
FIXED = WORK / 'c4-submission-versions' / VERSION
PUBLISH = Path('/workspace/result')
BACKUP = WORK / 'c4-submission-versions' / ('before-' + VERSION)
STAGE = WORK / ('c4-release-prep-' + VERSION) / 'publish-stage'
OLD_FILES = json.loads((FIXED/'old_official_inventory.json').read_text())['files']
OLD_RESULT = next(r['sha256'] for r in OLD_FILES if r['path']=='result/result.json')
OLD_TIME = next(r['sha256'] for r in OLD_FILES if r['path']=='result/infer_time.json')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inventory(root):
    require(not root.is_symlink(), '目录是链接')
    result = []
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), '目录含链接或越界')
        if path.is_file():
            result.append({'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
    return result


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    github = json.load(sys.stdin)
    require(github.get('remote_verified') is True and github.get('repository') == 'ZewenCheng/C4_fork'
            and github.get('tag') == 'c4-' + VERSION and re.fullmatch('[0-9a-f]{40}', github.get('commit', '')),
            '缺少本版GitHub独立核验凭证')
    require(set(github).issubset({'remote_verified', 'repository', 'tag', 'commit', 'manifest_sha256',
                                'verification_receipt_sha256', 'verified_files', 'verified_at'}), '凭证含未审阅字段')
    require(sha(FIXED / 'package_manifest.json') == github['manifest_sha256'], '固定清单变化')
    manifest = read(FIXED / 'package_manifest.json')
    require(manifest['version'] == VERSION and manifest['github_tag'] == github['tag'], '版本不符')
    require(not (FIXED / 'publication_receipt.json').exists(), '本版已存在发布凭证，禁止再次切换')
    require(PUBLISH.resolve() == PUBLISH and {p.name for p in PUBLISH.iterdir()} == {'code', 'design', 'result'}, '旧作品结构变化')
    require(sha(PUBLISH / 'result/result.json') == OLD_RESULT and sha(PUBLISH / 'result/infer_time.json') == OLD_TIME,
            '现有作品不是预期五折伙伴版，拒绝覆盖')
    for path in [FIXED, BACKUP, STAGE]:
        require(path.resolve().is_relative_to(WORK) and not path.is_symlink(), '工作路径越界')
    require(not BACKUP.exists() and not STAGE.exists(), '保留或暂存目录已存在，请先检查')
    expected = []
    for row in manifest['files']:
        rel = PurePosixPath(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in {'code', 'design', 'result'}, '清单路径越界')
        expected.append({key: row[key] for key in ['path', 'sha256', 'bytes']})
    expected.sort(key=lambda x: x['path'])
    require(len({r['path'] for r in expected}) == len(expected), '固定清单重复')
    require(inventory(FIXED / 'submit') == expected, '固定作品与清单不同')
    verify_packaged_timing(manifest, FIXED / 'submit/result')
    archive = Path(manifest['archive'])
    require(archive.resolve().parent == FIXED and sha(archive) == manifest['archive_sha256'], '压缩包变化')
    validation = read(FIXED / 'package_validation.json')
    require(validation['清单SHA256'] == github['manifest_sha256'] and validation['压缩包SHA256'] == manifest['archive_sha256'],
            '缺少对应独立验包记录')
    old = inventory(PUBLISH)
    require(old == OLD_FILES, '官方作品与封包前旧版本快照不一致')
    STAGE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXED / 'submit', STAGE)
    require(inventory(STAGE) == expected, '暂存作品复制不完整')
    lock_path = STAGE.parent / 'publication.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(inventory(PUBLISH) == old, '切换前旧作品已变化')
        BACKUP.mkdir()
        moved_old, moved_new = [], []
        try:
            for name in ['code', 'design', 'result']:
                source, target = PUBLISH / name, BACKUP / name
                require(source.resolve().parent == PUBLISH and target.resolve().parent == BACKUP, '旧目录移动越界')
                source.rename(target)
                moved_old.append(name)
            for name in ['code', 'design', 'result']:
                source, target = STAGE / name, PUBLISH / name
                require(source.resolve().parent == STAGE and target.resolve().parent == PUBLISH, '新目录移动越界')
                source.rename(target)
                moved_new.append(name)
            require(inventory(BACKUP) == old and inventory(PUBLISH) == expected, '新旧作品逐文件读回失败')
            verify_packaged_timing(manifest, PUBLISH / 'result')
            check = subprocess.run([str(PUBLISH / 'code/run'), '--check'], cwd=PUBLISH / 'code',
                                   capture_output=True, text=True, timeout=1200,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            require(check.returncode == 0, '最终路径入口自检失败')
            require(inventory(BACKUP) == old and inventory(PUBLISH) == expected, '自检改变文件内容')
            save(BACKUP / 'preservation_manifest.json', {'来源': str(PUBLISH), 'files': old, '原因': '完整保留五折多视图伙伴版作品'})
            receipt = {'状态': '官方作品切换及新旧逐文件读回通过', 'version': VERSION, '目录': str(PUBLISH),
                       '旧作品保留': str(BACKUP), '旧文件数': len(old), '新文件数': len(expected),
                       '旧结果SHA256': OLD_RESULT, '旧计时SHA256': OLD_TIME,
                       '新结果SHA256': sha(PUBLISH / 'result/result.json'), '新计时SHA256': sha(PUBLISH / 'result/infer_time.json'),
                       'infer_time': read(PUBLISH / 'result/infer_time.json'), 'GitHub': github,
                       'package_manifest_sha256': github['manifest_sha256'], 'entry_check_exit_code': check.returncode,
                       'timing_mode': manifest.get('timing_mode', 'native'),
                       'timing_scope': manifest['timing_transformation'],
                       '切换方式': '同文件系统逐目录rename，持锁且失败回滚；不声称三目录同时原子可见',
                       '比赛提交或评测': False, 'published_at': time.time()}
            save(FIXED / 'publication_receipt.json', receipt)
        except BaseException:
            for name in reversed(moved_new):
                (PUBLISH / name).rename(STAGE / name)
            for name in reversed(moved_old):
                (BACKUP / name).rename(PUBLISH / name)
            require(inventory(PUBLISH) == old, '回滚后旧作品读回失败')
            raise
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
