#!/usr/bin/env python3
"""执行原推理脚本并记录成功运行的摘要，供正式打包校验来源。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from common import PACKAGE_ROOT, atomic_json, load_config, load_manifest, resolve_image_path, resolve_package_path, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('template', nargs='?', type=Path)
    args = parser.parse_args()
    manifest, output = args.manifest.resolve(), args.output.resolve()
    if not output.is_relative_to(Path('/workspace/work')) or output == Path('/workspace/work'):
        raise ValueError('推理产物必须保存在 /workspace/work 的独立子目录。')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('输出目录已有内容，请为本次推理选择新的目录。')
    rows = load_manifest(manifest)
    if not rows or any(not resolve_image_path(manifest, row).resolve().is_relative_to(Path('/dataset/决赛数据集1/赛题4/线上测试集')) for row in rows):
        raise ValueError('正式推理清单必须只引用官方赛题 4 测试目录。')
    if os.environ.get('EXPQ_DELTA') or os.environ.get('CLASSIFIER'):
        raise ValueError('本部署入口使用现成 expQ 基线，不接受未经本任务验证的训练产物。')
    config = load_config()
    module = os.environ.get(config['qwen']['provider_module_env'], '')
    if 'mock' in module.lower() or module.startswith('tests'):
        raise ValueError('正式推理不能使用历史模拟接口。')
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    command = ['sh', str(PACKAGE_ROOT / 'scripts/run_infer.sh'), str(manifest), str(output)]
    if args.template:
        command.append(str(args.template.resolve()))
    subprocess.run(command, cwd=PACKAGE_ROOT, check=True)
    elapsed_seconds = round(time.monotonic() - clock, 3)
    infer_time = output / 'infer_time.json'
    # 官方示例使用无单位的整数；按常见计时约定记录总推理毫秒数。
    atomic_json(infer_time, {'infer_time': round(elapsed_seconds * 1000)})
    evidence = {
        'run_kind': 'official_c4_inference', 'completed': True,
        'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': elapsed_seconds,
        'infer_time': str(infer_time), 'infer_time_sha256': sha256_file(infer_time),
        'infer_time_milliseconds': round(elapsed_seconds * 1000),
        'manifest': str(manifest), 'manifest_sha256': sha256_file(manifest),
        'result': str(output / 'result.json'), 'result_sha256': sha256_file(output / 'result.json'),
        'features': str(output / 'test_features.npz'), 'features_sha256': sha256_file(output / 'test_features.npz'),
        'config_sha256': sha256_file(PACKAGE_ROOT / 'config/deploy_config.json'),
        'model_manifest_sha256': sha256_file(resolve_package_path('models/MODEL_MANIFEST.json')),
        'rows': len(rows), 'qwen_model': os.environ.get(config['qwen']['model_env'], config['qwen']['default_model']),
        '说明': '由既有真实推理流程成功退出后生成；不含图片、逐图结果、接口地址或认证信息。',
    }
    atomic_json(output / 'run_evidence.json', evidence)
    print(json.dumps({'状态': '真实推理及结果校验完成', '证据': str(output / 'run_evidence.json'), '记录数': len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
