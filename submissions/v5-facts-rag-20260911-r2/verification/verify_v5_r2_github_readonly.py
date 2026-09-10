"""独立只读核验新伙伴固定标签、提交、树及全部代码/关键文档字节。"""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'v5-facts-rag-20260911-r2'
REPOSITORY = 'ZewenCheng/C4_fork'
PREFIX = 'submissions/' + VERSION + '/'
API = 'https://api.github.com/repos/' + REPOSITORY


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def credential():
    result = subprocess.run(['git', 'credential-manager', '--no-ui', 'get'],
                            input=b'protocol=https\nhost=github.com\nusername=ZewenCheng\n\n',
                            check=True, capture_output=True)
    fields = {}
    for line in result.stdout.splitlines():
        if b'=' in line:
            key, value = line.split(b'=', 1)
            fields[key.decode('ascii')] = value.decode('utf-8')
    token = fields.get('password') or fields.get('credential')
    require(fields.get('username', '').casefold() == 'zewencheng' and bool(token), '未取得目标账号的可用凭证')
    return token


def get(token, endpoint):
    require(endpoint.startswith('/git/'), '只允许Git对象只读接口')
    request = Request(API + endpoint, method='GET', headers={
        'Accept': 'application/vnd.github+json', 'Authorization': 'Bearer ' + token,
        'User-Agent': 'C4-independent-readonly-verification', 'X-GitHub-Api-Version': '2022-11-28',
        'Accept-Encoding': 'identity', 'Connection': 'close'})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            if exc.code >= 500 and attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError('GitHub只读请求失败，HTTP ' + str(exc.code)) from None
        except (URLError, TimeoutError, OSError):
            if attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError('GitHub只读网络请求失败') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/github_v5_r2_independent_verification.json')
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text(encoding='utf-8'))
    require(expected['version'] == VERSION and re.fullmatch('[0-9a-f]{40}', expected['commit']), '目标版本或提交无效')
    files = expected['files_sha256']
    require(isinstance(files, dict) and files, '缺少逐文件摘要')
    for path, digest in files.items():
        rel = PurePosixPath(path)
        require(path.startswith(PREFIX) and path == rel.as_posix() and '..' not in rel.parts
                and re.fullmatch('[0-9a-f]{64}', digest), '摘要白名单路径不符')
    require(PREFIX + 'README.md' in files and PREFIX + 'package_manifest.json' in files
            and any(p.endswith('.pdf') for p in files) and any(p.startswith(PREFIX + 'code/') for p in files),
            '必须包含README、清单、设计PDF和全部代码')
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'tmp') and not args.output.is_symlink(), '核验记录只能写项目tmp')
    require(not output.exists(), '核验记录已存在，请用新的本地输出名')
    token = credential()
    tag_name = 'c4-' + VERSION
    tag = get(token, '/git/ref/tags/' + tag_name)
    obj = tag['object']
    if obj['type'] == 'tag':
        annotated = get(token, '/git/tags/' + obj['sha'])
        obj = annotated['object']
    require(obj['type'] == 'commit' and obj['sha'] == expected['commit'], '固定标签未指向预期提交')
    commit = get(token, '/git/commits/' + expected['commit'])
    require(commit['sha'] == expected['commit'], '提交对象不符')
    if expected.get('base_commit'):
        require([p['sha'] for p in commit['parents']] == [expected['base_commit']], '提交父版本不符')
    tree = get(token, '/git/trees/' + commit['tree']['sha'] + '?recursive=1')
    require(not tree.get('truncated') and tree['sha'] == commit['tree']['sha'], '远端树截断或摘要不同')
    rows = {row['path']: row for row in tree['tree'] if row['type'] == 'blob'}
    code_paths = {p for p in rows if p.startswith(PREFIX + 'code/')}
    require(code_paths == {p for p in files if p.startswith(PREFIX + 'code/')}, '全部代码清单不完整或存在额外代码')
    checked = {}
    for path, digest in files.items():
        row = rows.get(path)
        require(row is not None and row['mode'] in {'100644', '100755'}, '文件缺失或不是普通文件')
        require(row.get('size', 0) <= 10_000_000, '核验文件异常大')
        blob = get(token, '/git/blobs/' + row['sha'])
        require(blob.get('encoding') == 'base64', 'GitHub未返回可核验原始字节')
        raw = base64.b64decode(blob['content'])
        object_sha = hashlib.sha1(('blob ' + str(len(raw)) + '\0').encode('ascii') + raw).hexdigest()
        require(object_sha == row['sha'] == blob['sha'] and hashlib.sha256(raw).hexdigest() == digest,
                '远端文件字节摘要不符：' + path)
        checked[path] = {'sha256': digest, 'bytes': len(raw)}
    # 最后再次读取固定标签，防止核验期间其被移动。
    final_tag = get(token, '/git/ref/tags/' + tag_name)
    require(final_tag['object'] == tag['object'], '核验期间标签变化')
    receipt = {'remote_verified': True, 'repository': REPOSITORY, 'tag': tag_name,
               'commit': expected['commit'], 'manifest_sha256': files[PREFIX + 'package_manifest.json'],
               'verified_files': len(checked), 'verified_at': time.time()}
    report = {'状态': '独立只读标签及逐文件字节核验通过', 'publication_credential': receipt,
              'files': checked, 'remote_writes': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), '核验记录已存在，请用新的本地输出名')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'状态': report['状态'], '文件数': len(checked), '输出': str(output),
                      '凭证': receipt}, ensure_ascii=False))


if __name__ == '__main__':
    main()
