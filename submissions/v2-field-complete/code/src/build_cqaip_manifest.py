#!/usr/bin/env python3
"""为已核实的 CQAIP 赛题 4 平铺测试目录生成清单，未知桥名保留为空。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from common import IMAGE_SUFFIXES, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('/dataset/决赛数据集1/赛题4/线上测试集'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.dataset_root.resolve(), args.output.resolve()
    if source != Path('/dataset/决赛数据集1/赛题4/线上测试集'):
        raise ValueError('本入口只适用于已核实的赛题 4 测试目录。')
    if not output.is_relative_to(Path('/workspace/work')):
        raise ValueError('清单必须保存在 /workspace/work 内。')
    if output.suffix != '.jsonl':
        raise ValueError('清单文件必须使用 .jsonl 扩展名。')
    rows = []
    for category in ('桥梁', '轨道'):
        directory = source / category
        if not directory.is_dir():
            raise FileNotFoundError('缺少已核实的分类目录。')
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.is_symlink() or path.parent != directory:
                    raise ValueError('目录结构与已核实的平铺结构不同，需要重新检查元数据。')
                rows.append({'filename': path.name, 'questionCategory': category, 'bridgeName': '', 'image_path': str(path)})
    if not rows or len({row['filename'] for row in rows}) != len(rows):
        raise ValueError('清单为空或存在重复文件名。')
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError('已有清单内容不同，不能覆盖。')
    else:
        temporary = output.with_name(output.stem + '.tmp.jsonl')
        with temporary.open('xb') as handle:
            handle.write(payload)
        load_manifest(temporary)
        os.replace(temporary, output)
    load_manifest(output)
    print(json.dumps({'状态': '清单结构检查通过', '总数': len(rows), '分类数量': dict(Counter(row['questionCategory'] for row in rows)), 'sha256': hashlib.sha256(payload).hexdigest(), '云端路径': str(output), '未知字段': '桥名缺少官方元数据，保留空字符串；未读取测试图片内容。'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
