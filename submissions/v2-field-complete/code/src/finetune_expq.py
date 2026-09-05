#!/usr/bin/env python3
"""在决赛云端训练集上继续微调expQ最后4块、norm和22类head。"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import PACKAGE_ROOT, choose_device, load_config, load_expq_model, load_manifest, prepare_local_image, resolve_image_path


class CloudRailDataset(Dataset):
    def __init__(self, manifest_path: Path, rows: list[dict], labels: np.ndarray, transform, config: dict):
        self.manifest_path = manifest_path
        self.rows = rows
        self.labels = labels
        self.transform = transform
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = prepare_local_image(resolve_image_path(self.manifest_path, self.rows[index]), self.config)
        return self.transform(image), int(self.labels[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config/deploy_config.json")
    parser.add_argument("--class-names", type=Path, default=PACKAGE_ROOT / "models/classifier/class_names.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = load_config(args.config)
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    class_index = {name: index for index, name in enumerate(class_names)}
    rows = [row for row in load_manifest(args.manifest) if row["questionCategory"] == "轨道"]
    if not rows or any(not isinstance(row.get("defectType"), str) for row in rows):
        raise RuntimeError("Training manifest needs rail defectType labels")
    unknown = sorted({row["defectType"] for row in rows} - set(class_names))
    if unknown:
        raise RuntimeError(f"Labels outside locked 22-class space: {unknown}")
    labels = np.asarray([class_index[row["defectType"]] for row in rows], dtype=np.int64)

    model = load_expq_model(config, device)
    import timm

    data_config = timm.data.resolve_data_config({}, model=model)
    transform = T.Compose([
        T.Resize((378, 378)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.1),
        T.ToTensor(),
        T.Normalize(data_config["mean"], data_config["std"]),
    ])
    for parameter in model.parameters():
        parameter.requires_grad = False
    last_blocks = list(range(len(model.blocks) - 4, len(model.blocks)))
    for block in model.blocks[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.norm.parameters():
        parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": [p for block in model.blocks[-4:] for p in block.parameters()], "lr": 3e-5},
        {"params": list(model.norm.parameters()) + list(model.head.parameters()), "lr": 1e-3},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        CloudRailDataset(args.manifest, rows, labels, transform, config),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    history = []
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        started = time.time()
        for images, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(targets)
            count += len(targets)
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": total / count, "seconds": time.time() - started})
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    prefixes = tuple([f"blocks.{index}." for index in last_blocks] + ["norm.", "head."])
    state = {
        name: tensor.detach().cpu().to(torch.float16)
        for name, tensor in model.state_dict().items()
        if name.startswith(prefixes)
    }
    payload = {
        "model_name": config["models"]["expq"]["timm_name"],
        "img_size": 378,
        "num_classes": len(class_names),
        "last_blocks": last_blocks,
        "tensor_count": len(state),
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "cloud_training_rows": len(rows),
        "seed": args.seed,
        "epochs": args.epochs,
        "history": history,
        "state_dict": state,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(json.dumps({"status": "EXPQ_CLOUD_FINETUNE_PASS", "output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
