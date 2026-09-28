#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-huihui-ai/GLM-5.3-Flash-abliterated-GGUF}"
MODEL_QUANT="${MODEL_QUANT:-UD-IQ1_S}"
MODEL_ROOT="${MODEL_ROOT:-/runpod-volume/models}"
CONTEXT_PER_SLOT="${CONTEXT_PER_SLOT:-204800}"
PARALLEL="${PARALLEL:-4}"
GPU_LAYERS="${GPU_LAYERS:-999}"
KV_CACHE_K="${KV_CACHE_K:-q8_0}"
KV_CACHE_V="${KV_CACHE_V:-q8_0}"
FLASH_ATTN="${FLASH_ATTN:-on}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
PORT="${PORT:-8000}"

TOTAL_CONTEXT=$(( CONTEXT_PER_SLOT * PARALLEL ))
SAFE_REPO="${MODEL_REPO//\//--}"
LOCAL_DIR="${MODEL_ROOT}/${SAFE_REPO}"

mkdir -p "${LOCAL_DIR}"

echo "============================================================"
echo "GLM-5.3 Flash RunPod Serverless"
echo "Repository:       ${MODEL_REPO}"
echo "Quant:            ${MODEL_QUANT}"
echo "Context / slot:   ${CONTEXT_PER_SLOT}"
echo "Parallel slots:   ${PARALLEL}"
echo "Total context:    ${TOTAL_CONTEXT}"
echo "KV cache K/V:     ${KV_CACHE_K} / ${KV_CACHE_V}"
echo "GPU layers:       ${GPU_LAYERS}"
echo "Flash attention:  ${FLASH_ATTN}"
echo "Model storage:    ${LOCAL_DIR}"
echo "HTTP port:        ${PORT}"
echo "============================================================"

echo "[1/2] Ensuring GGUF is present on the Network Volume..."
hf download "${MODEL_REPO}" \
  --include "${MODEL_QUANT}/*.gguf" \
  --local-dir "${LOCAL_DIR}"

MODEL_FILE="$(find "${LOCAL_DIR}/${MODEL_QUANT}" -maxdepth 1 -type f -name '*.gguf' | sort | head -n 1 || true)"

if [[ -z "${MODEL_FILE}" ]]; then
  echo "ERROR: No GGUF files found under ${LOCAL_DIR}/${MODEL_QUANT}"
  echo "Check MODEL_REPO and MODEL_QUANT."
  exit 1
fi

# For split GGUFs prefer shard 00001; llama.cpp discovers the remaining shards.
FIRST_SHARD="$(find "${LOCAL_DIR}/${MODEL_QUANT}" -maxdepth 1 -type f -name '*-00001-of-*.gguf' | sort | head -n 1 || true)"
if [[ -n "${FIRST_SHARD}" ]]; then
  MODEL_FILE="${FIRST_SHARD}"
fi

echo "[2/2] Starting llama-server with:"
echo "      ${MODEL_FILE}"
echo

EXTRA=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA=( ${EXTRA_ARGS} )
fi

exec llama-server \
  --model "${MODEL_FILE}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --ctx-size "${TOTAL_CONTEXT}" \
  --parallel "${PARALLEL}" \
  --n-gpu-layers "${GPU_LAYERS}" \
  --cache-type-k "${KV_CACHE_K}" \
  --cache-type-v "${KV_CACHE_V}" \
  --flash-attn "${FLASH_ATTN}" \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --cont-batching \
  "${EXTRA[@]}"
