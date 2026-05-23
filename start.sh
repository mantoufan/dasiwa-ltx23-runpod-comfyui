#!/usr/bin/env bash
set -Eeuo pipefail

: "${COMFYUI_DIR:=/workspace/ComfyUI}"
: "${COMFYUI_HOST:=0.0.0.0}"
: "${COMFYUI_PORT:=8188}"
: "${CLI_ARGS:=--listen ${COMFYUI_HOST} --port ${COMFYUI_PORT} --preview-method auto}"
: "${DOWNLOAD_MODELS:=true}"

if [ -x /start.sh ]; then
  /start.sh &
fi

if [ ! -d "${COMFYUI_DIR}/.git" ]; then
  mkdir -p "$(dirname "${COMFYUI_DIR}")"
  rsync -a /opt/ComfyUI/ "${COMFYUI_DIR}/"
fi

mkdir -p "${COMFYUI_DIR}/user/default/workflows"
cp -n /opt/workflows/*.json "${COMFYUI_DIR}/user/default/workflows/" 2>/dev/null || true

if [ "${DOWNLOAD_MODELS}" = "true" ] || [ "${DOWNLOAD_MODELS}" = "1" ] || [ "${DOWNLOAD_MODELS}" = "yes" ]; then
  /download_models.sh
fi

cd "${COMFYUI_DIR}"
python main.py ${CLI_ARGS}
