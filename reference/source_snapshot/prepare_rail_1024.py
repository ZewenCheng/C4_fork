#!/usr/bin/env python3
"""初赛轨道本地视觉分支的原始1024像素预处理配方。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def convert(job):
    source, output = job
    image = Image.open(source).convert("RGB")
    width, height = image.size
    scale = 1024 / max(width, height)
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "JPEG", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [
        (path, args.output_dir / f"{path.stem}.jpg")
        for path in sorted(args.input_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    with ThreadPoolExecutor(args.workers) as executor:
        list(executor.map(convert, jobs))
    print(f"converted={len(jobs)}")


if __name__ == "__main__":
    main()
