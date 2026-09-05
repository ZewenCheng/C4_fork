#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MANIFEST=${1:?usage: run_train.sh TRAIN_MANIFEST_JSONL OUTPUT_DIR}
OUTPUT_DIR=${2:?usage: run_train.sh TRAIN_MANIFEST_JSONL OUTPUT_DIR}

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$ROOT/src/preflight.py" --require-qwen
"$PYTHON_BIN" "$ROOT/src/finetune_expq.py" --manifest "$MANIFEST" --output "$OUTPUT_DIR/expq_cloud_delta.pt"
"$PYTHON_BIN" "$ROOT/src/extract_features.py" --manifest "$MANIFEST" --expq-delta "$OUTPUT_DIR/expq_cloud_delta.pt" --output "$OUTPUT_DIR/train_features.npz"
"$PYTHON_BIN" "$ROOT/src/train_classifier.py" --manifest "$MANIFEST" --features "$OUTPUT_DIR/train_features.npz" --output "$OUTPUT_DIR/expq_cloud_lr.npz"
