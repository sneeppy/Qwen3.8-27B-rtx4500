#!/bin/bash
# Container entrypoint. First argument selects what to run:
#   single   single-user/start_qwen.sh  (speculative decoding, low latency)
#   batch    batch/start_qwen.sh        (throughput)
#   prepare  docker/prepare.sh          (download + requantize the model into /app/models)
#   verify   verify.sh [args]
#   <anything else> is exec'd as a command (e.g. bash)
#
# Compose knobs are mapped here so start_qwen.sh still sees vLLM variables:
# API_KEY, numeric CTX, MODE.
#
# Before serving, docker/prepare.sh runs (idempotent), then verify.sh --no-server,
# which aborts on FAIL. PREPARE=0 / VERIFY=0 skip those steps.
set -e
cd /app

# API_KEY is the compose name; vLLM launchers read VLLM_API_KEY.
if [ -n "${API_KEY:-}" ] && [ -z "${VLLM_API_KEY:-}" ]; then
  export VLLM_API_KEY="$API_KEY"
fi

# Numeric CTX (token window) -> MAX_LEN + a named profile.
# Named values fast/long/huge pass through unchanged.
if [ -n "${CTX:-}" ] && [[ "$CTX" =~ ^[0-9]+$ ]]; then
  export MAX_LEN="${MAX_LEN:-$CTX}"
  if [ "$MAX_LEN" -le 65536 ]; then
    export CTX=fast
  elif [ "$MAX_LEN" -le 131072 ]; then
    export CTX=long
  else
    export CTX=huge
  fi
  echo "[entrypoint] numeric context ${MAX_LEN} -> CTX=${CTX} MAX_LEN=${MAX_LEN}"
fi

cmd=${1:-${MODE:-single}}; shift || true
case "$cmd" in
  single|batch)
    if [ "${PREPARE:-1}" != "0" ]; then
      bash docker/prepare.sh
    fi
    if [ "${VERIFY:-1}" != "0" ]; then
      bash verify.sh --no-server || { echo "entrypoint: verify.sh FAILED — fix the above or set VERIFY=0"; exit 1; }
    fi
    echo "=============================================================================="
    echo " Qwen3.8-27B Serving"
    echo "=============================================================================="
    echo " Mode:            ${cmd}"
    echo " SPEC:            ${SPEC:-mtp}"
    echo " CTX profile:     ${CTX:-fast}"
    echo " max-model-len:   ${MAX_LEN:-"(launcher default)"}"
    echo " Prefix cache:    ${PREFIX_CACHE:-0}"
    echo " API key:         $([ -n "${VLLM_API_KEY:-}" ] && echo set || echo unset)"
    echo " Endpoint:        http://0.0.0.0:${PORT:-8080}"
    echo "=============================================================================="
    if [ "$cmd" = single ]; then exec bash single-user/start_qwen.sh "$@"; else exec bash batch/start_qwen.sh "$@"; fi ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  verify)  exec bash verify.sh "$@" ;;
  *)       exec "$cmd" "$@" ;;
esac
