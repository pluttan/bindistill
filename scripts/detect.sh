#!/bin/sh
# What this machine can run, decided before anything is installed.
#
# Prints shell assignments so the Makefile can eval them. Two things are
# decided here: which torch build matches the driver, and which preset fits the
# card. Both have bitten us — a wheel built for a newer CUDA falls back to the
# processor without an error, and a preset sized for a bigger card dies hours in.
#
# Signed: pluttan

set -eu

GPU_NAME=""
GPU_MB=0
DRIVER_CUDA=""
TORCH_CUDA=""
PRESET="smoke"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)
fi

[ -z "${GPU_MB:-}" ] && GPU_MB=0

# The wheel must not ask for more CUDA than the driver provides.
case "$DRIVER_CUDA" in
    "")            TORCH_CUDA="" ;;
    12.0|12.1|12.2|12.3) TORCH_CUDA="cu121" ;;
    12.4|12.5|12.6|12.7) TORCH_CUDA="cu124" ;;
    *)             TORCH_CUDA="cu128" ;;
esac

# Preset by card memory. The boundaries are where the presets were measured.
if [ "$GPU_MB" -ge 30000 ]; then
    PRESET="full"
elif [ "$GPU_MB" -ge 10000 ]; then
    PRESET="small"
elif [ "$GPU_MB" -gt 0 ]; then
    PRESET="tiny"
fi

if [ "${1:-}" = "--report" ]; then
    if [ "$GPU_MB" -gt 0 ]; then
        printf 'card          %s, %s MB\n' "${GPU_NAME:-unknown}" "$GPU_MB"
        printf 'driver CUDA   %s\n' "${DRIVER_CUDA:-unknown}"
        printf 'torch build   %s\n' "${TORCH_CUDA:-default}"
    else
        printf 'card          none found (nvidia-smi absent or no device)\n'
        printf 'torch build   default (processor)\n'
    fi
    printf 'preset        %s\n' "$PRESET"
    exit 0
fi

printf 'GPU_MB=%s\n' "$GPU_MB"
printf 'TORCH_CUDA=%s\n' "$TORCH_CUDA"
printf 'PRESET=%s\n' "$PRESET"
