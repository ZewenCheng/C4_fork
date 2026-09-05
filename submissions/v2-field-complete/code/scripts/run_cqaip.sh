#!/bin/sh
# 在赛事 PPU 镜像中运行现成方案，所有缓存和产物留在 /workspace。
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
case "$ROOT" in
    /workspace/*) ;;
    *) echo "本入口仅用于官方云端 /workspace 下的部署。" >&2; exit 2 ;;
esac
SDK_ROOT=/usr/local/PPU_SDK
SDK_LIB="$SDK_ROOT/targets/x86_64-linux/lib"
if [ ! -f "$SDK_LIB/libhggcrt1.so" ]; then
    echo "没有找到平台 PPU 动态库，请核对官方镜像。" >&2
    exit 2
fi
# 与平台 SDK 的 envsetup.sh CUDA 模式保持一致；完整编译环境是大模型算子所必需的。
export PPU_SDK="$SDK_ROOT"
export CUDA_PATH="$SDK_ROOT/CUDA_SDK"
export CUDA_HOME="$CUDA_PATH"
export CUDA_TOOLKIT_ROOT="$CUDA_PATH"
export SAILSHMEM_DIR="$SDK_ROOT/sailSHMEM"
export PATH="$CUDA_PATH/bin:$SDK_ROOT/asight/bin:$SDK_ROOT/bin:$SDK_ROOT/ppu-smi/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$SDK_ROOT/lib:$SAILSHMEM_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_PATH/lib64:$SDK_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$ROOT/.runtime/huggingface"
export TORCH_HOME="$ROOT/.runtime/torch"
export XDG_CACHE_HOME="$ROOT/.runtime/cache"
export TMPDIR="$ROOT/.runtime/tmp"
export TRITON_CACHE_DIR="$ROOT/.runtime/triton"
export CUDA_CACHE_PATH="$ROOT/.runtime/cuda"
export MPLCONFIGDIR="$ROOT/.runtime/matplotlib"
export ALIPPU_CONFIG_PATH="$ROOT/.runtime/ppu"
export HGRTC_CACHE_PATH="$ROOT/.runtime/ppu/hgrtc"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
mkdir -p "$ALIPPU_CONFIG_PATH" "$HGRTC_CACHE_PATH"
for ARCH in PPU0010 PPU0015 PPU0017; do
    mkdir -p "$ALIPPU_CONFIG_PATH/rtc/$ARCH"
done
if [ ! -f "$ALIPPU_CONFIG_PATH/acompute.cfg" ]; then
    (set -C; printf '[acConfigs]\nRTC_CACHE_ENABLE=1\nRTC_CACHE_PATH=%s\n' "$ALIPPU_CONFIG_PATH/rtc" > "$ALIPPU_CONFIG_PATH/acompute.cfg")
fi
if ! grep -Fxq "RTC_CACHE_PATH=$ALIPPU_CONFIG_PATH/rtc" "$ALIPPU_CONFIG_PATH/acompute.cfg"; then
    echo "已有 PPU 缓存配置与本工作区不同，请检查后再运行。" >&2
    exit 2
fi
if [ -f "$ROOT/config/cqaip.local.env" ]; then
    set -a
    . "$ROOT/config/cqaip.local.env"
    set +a
fi
# 仅增加工作区内的图像工具依赖和 provider；平台 PyTorch 不在此目录安装。
export PYTHONPATH="$ROOT/.runtime/qwen_deps:$ROOT/adapters${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
ACTION=${1:-preflight}
if [ "$#" -gt 0 ]; then shift; fi
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHON_BIN
case "$ACTION" in
    preflight) exec "$PYTHON_BIN" -B "$ROOT/src/preflight.py" "$@" ;;
    infer) exec "$PYTHON_BIN" -B "$ROOT/src/record_cqaip_run.py" "$@" ;;
    qwen)
        export QWEN_PROVIDER_MODULE=qwen_local_provider
        export QWEN_LOCAL_MODEL_DIR="${QWEN_LOCAL_MODEL_DIR:-$ROOT/models/transformers/qwen3_vl_embedding}"
        export QWEN_LOCAL_OFFLOAD_DIR="${QWEN_LOCAL_OFFLOAD_DIR:-$ROOT/.runtime/qwen_offload}"
        export QWEN_LOCAL_GPU_GIB="${QWEN_LOCAL_GPU_GIB:-80}"
        export QWEN_LOCAL_CPU_GIB="${QWEN_LOCAL_CPU_GIB:-8}"
        exec "$PYTHON_BIN" -B "$ROOT/src/qwen_local_cli.py" "$@"
        ;;
    *) echo "用法：preflight [--require-qwen]、infer 清单 输出目录 [官方模板]、qwen check|smoke|embed" >&2; exit 2 ;;
esac
