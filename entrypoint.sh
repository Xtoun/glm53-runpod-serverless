#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${MODEL_ROOT:-/runpod-volume/models}" "${STATE_DIR:-/runpod-volume/glm53-panel}"

echo "============================================================"
echo "GLM-5.3 RunPod panel starting"
echo "Panel port: ${PANEL_PORT:-8000}"
echo "Model root: ${MODEL_ROOT:-/runpod-volume/models}"
echo "State dir:  ${STATE_DIR:-/runpod-volume/glm53-panel}"
echo "============================================================"

exec python3 /app/main.py
