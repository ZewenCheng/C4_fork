#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MANIFEST=${1:?usage: run_infer.sh MANIFEST_JSONL OUTPUT_DIR [TEMPLATE_JSON]}
OUTPUT_DIR=${2:?usage: run_infer.sh MANIFEST_JSONL OUTPUT_DIR [TEMPLATE_JSON]}
TEMPLATE=${3:-}

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$ROOT/src/preflight.py" --require-qwen
if [ -n "${EXPQ_DELTA:-}" ]; then
  "$PYTHON_BIN" "$ROOT/src/extract_features.py" --manifest "$MANIFEST" --expq-delta "$EXPQ_DELTA" --output "$OUTPUT_DIR/test_features.npz"
else
  "$PYTHON_BIN" "$ROOT/src/extract_features.py" --manifest "$MANIFEST" --output "$OUTPUT_DIR/test_features.npz"
fi
MODEL_ROOT=${CQAIP_MODELS_DIR:-$ROOT/models}
CLASSIFIER=${CLASSIFIER:-$MODEL_ROOT/classifier/expq_lr.npz}
if [ -n "$TEMPLATE" ]; then
  "$PYTHON_BIN" "$ROOT/src/infer_cloud.py" --manifest "$MANIFEST" --features "$OUTPUT_DIR/test_features.npz" --classifier "$CLASSIFIER" --template-json "$TEMPLATE" --output "$OUTPUT_DIR/result.json"
else
  "$PYTHON_BIN" "$ROOT/src/infer_cloud.py" --manifest "$MANIFEST" --features "$OUTPUT_DIR/test_features.npz" --classifier "$CLASSIFIER" --output "$OUTPUT_DIR/result.json"
fi
"$PYTHON_BIN" "$ROOT/src/validate_result.py" --manifest "$MANIFEST" --result "$OUTPUT_DIR/result.json"
