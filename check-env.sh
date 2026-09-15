#!/usr/bin/env bash
# ==============================================================================
# check-env.sh — Host checks for Docker serving of Qwen3.8-27B on Ada (sm_89)
# ==============================================================================
# Validates NVIDIA driver, GPU, Docker, and NVIDIA Container Toolkit.
# vLLM lives in the image; host nvcc/GCC are not required.
#
# Usage:
#   ./check-env.sh
# ==============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    # shellcheck disable=SC1090
    set -a
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/.env"
    set +a
fi

PASS="[  OK  ]"
WARN="[ WARN ]"
FAIL="[ FAIL ]"
INFO="[ INFO ]"

ERRORS=0
WARNINGS=0

echo "=============================================================================="
echo " Environment Doctor: Qwen3.8-27B Docker Stack"
echo " Target Hardware: NVIDIA Ada sm_89 (RTX 4500 Ada Generation, 24 GB)"
echo "=============================================================================="
echo ""

echo "--- 1. NVIDIA Driver & GPU Hardware ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
    DRIVER_MAJOR="$(echo "${DRIVER_VER}" | cut -d. -f1)"
    echo "${INFO} Detected NVIDIA Driver version: ${DRIVER_VER}"

    if [[ "${DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
        if (( DRIVER_MAJOR < 575 )); then
            echo "${FAIL} NVIDIA driver ${DRIVER_VER} is too old for the CUDA 13 image."
            echo "       Need a branch that speaks CUDA 13 (typically 575/580+)."
            ERRORS=$((ERRORS + 1))
        else
            echo "${PASS} NVIDIA driver version ${DRIVER_VER} can run the CUDA 13 image."
        fi
    else
        echo "${WARN} Could not parse driver version '${DRIVER_VER}'."
        WARNINGS=$((WARNINGS + 1))
    fi

    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
    GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo "unknown")"
    echo "${INFO} GPU: ${GPU_NAME} (${GPU_MEM})"

    if echo "${GPU_NAME}" | grep -qiE 'RTX 4500.*Ada|4500 Ada'; then
        echo "${PASS} GPU is RTX 4500 Ada Generation."
    elif echo "${GPU_NAME}" | grep -qiE 'Ada|L40'; then
        echo "${WARN} GPU (${GPU_NAME}) is Ada-class but not the 4500 Ada this stack names."
        echo "       24 GB profiles may still fit; re-measure tok/s."
        WARNINGS=$((WARNINGS + 1))
    else
        echo "${WARN} GPU (${GPU_NAME}) is not verified as RTX 4500 Ada."
        echo "       This repo is the vLLM stack for a 24 GB Ada card."
        WARNINGS=$((WARNINGS + 1))
    fi

    MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || echo 0)"
    if [[ "${MEM_MIB}" =~ ^[0-9]+$ ]]; then
        if (( MEM_MIB < 20000 )); then
            echo "${FAIL} VRAM ${MEM_MIB} MiB is below 20 GB; the 24 GB launcher pins will not fit."
            ERRORS=$((ERRORS + 1))
        elif (( MEM_MIB < 23000 )); then
            echo "${WARN} VRAM ${MEM_MIB} MiB is under 24 GB. Lower GPU_UTIL / MAX_LEN before first boot."
            WARNINGS=$((WARNINGS + 1))
        else
            echo "${PASS} VRAM ${MEM_MIB} MiB is enough for the 24 GB profiles."
        fi
    fi
else
    echo "${FAIL} 'nvidia-smi' not found on PATH. NVIDIA GPU driver may not be installed."
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "--- 2. Docker & NVIDIA Container Toolkit ---"
if command -v docker >/dev/null 2>&1; then
    echo "${PASS} docker: $(docker --version 2>/dev/null | head -n1)"
    if docker info >/dev/null 2>&1; then
        echo "${PASS} Docker daemon is reachable."
    else
        echo "${WARN} Docker CLI found, but the daemon is not reachable from this user."
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "${FAIL} 'docker' not found. Install Docker Engine on the Ada host."
    ERRORS=$((ERRORS + 1))
fi

if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        echo "${PASS} docker compose: $(docker compose version --short 2>/dev/null || docker compose version | head -n1)"
    elif command -v docker-compose >/dev/null 2>&1; then
        echo "${PASS} docker-compose: $(docker-compose --version 2>/dev/null | head -n1)"
    else
        echo "${FAIL} Docker Compose not found (need 'docker compose' or docker-compose)."
        ERRORS=$((ERRORS + 1))
    fi

    if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia\|nvidia'; then
        echo "${PASS} NVIDIA container runtime appears registered with Docker."
    else
        echo "${WARN} Could not confirm NVIDIA Container Toolkit from 'docker info'."
        echo "       If GPU access fails, install the toolkit and restart Docker."
        echo "       Probe: docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi"
        WARNINGS=$((WARNINGS + 1))
    fi
fi
echo ""

echo "--- 3. Serving artifacts ---"
if [[ -f "${SCRIPT_DIR}/docker-compose.yml" && -f "${SCRIPT_DIR}/Dockerfile" ]]; then
    echo "${PASS} Docker serving files present (Dockerfile, docker-compose.yml)."
else
    echo "${FAIL} Dockerfile or docker-compose.yml missing from ${SCRIPT_DIR}."
    ERRORS=$((ERRORS + 1))
fi

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    if [[ -z "${CADDY_NETWORK:-}" ]]; then
        echo "${FAIL} CADDY_NETWORK is empty. Set it in .env to Caddy's Docker network."
        ERRORS=$((ERRORS + 1))
    else
        echo "${PASS} CADDY_NETWORK=${CADDY_NETWORK}"
        if command -v docker >/dev/null 2>&1 && docker network inspect "${CADDY_NETWORK}" >/dev/null 2>&1; then
            echo "${PASS} Docker network ${CADDY_NETWORK} exists."
        else
            echo "${WARN} Docker network '${CADDY_NETWORK}' not found yet. Create it or fix the name before compose up."
            WARNINGS=$((WARNINGS + 1))
        fi
    fi

    if [[ -z "${WEBUI_SECRET_KEY:-}" ]]; then
        echo "${FAIL} WEBUI_SECRET_KEY is empty. Generate one: openssl rand -hex 32"
        ERRORS=$((ERRORS + 1))
    elif (( ${#WEBUI_SECRET_KEY} < 32 )); then
        echo "${WARN} WEBUI_SECRET_KEY is shorter than 32 characters; use a random 32-byte key."
        WARNINGS=$((WARNINGS + 1))
    else
        echo "${PASS} WEBUI_SECRET_KEY is configured."
    fi

    if [[ -z "${API_KEY:-}" ]]; then
        echo "${WARN} API_KEY is empty. vLLM has no authentication (loopback-only API is still protected from the network)."
        WARNINGS=$((WARNINGS + 1))
    else
        echo "${PASS} API_KEY is configured for vLLM and Open WebUI."
    fi
else
    echo "${INFO} No .env yet. Copy .env.example and set CADDY_NETWORK, WEBUI_SECRET_KEY and API_KEY."
fi

MODEL_DIR_RAW="${MODEL_DIR:-${MODELS_DIR:-./models}}"
if [[ "${MODEL_DIR_RAW}" = /* ]]; then
    MODEL_DIR_HOST="${MODEL_DIR_RAW}"
else
    MODEL_DIR_HOST="${SCRIPT_DIR}/${MODEL_DIR_RAW}"
fi

echo "${INFO} MODEL_DIR=${MODEL_DIR_RAW} -> ${MODEL_DIR_HOST}"

if [[ -d "${MODEL_DIR_HOST}/Qwen3.8-27B-W4A16-AutoRound" || -d "${MODEL_DIR_HOST}/Qwen3.8-27B-W4A16-AutoRound-fast" ]]; then
    echo "${PASS} Prepared model directory found under MODEL_DIR."
else
    echo "${INFO} Weights not in MODEL_DIR; the container will download and requantize on first start (~20 GB)."
fi
echo ""

echo "=============================================================================="
echo " Summary: ${ERRORS} error(s), ${WARNINGS} warning(s)"
echo "=============================================================================="

if (( ERRORS > 0 )); then
    echo "Result: Host does not meet critical stack requirements."
    echo "        Resolve the errors above before: docker compose up -d --build"
    exit 1
elif (( WARNINGS > 0 )); then
    echo "Result: Host is usable, but review warnings above."
    echo "        Next: docker compose up -d --build"
    exit 0
else
    echo "Result: Host is ready. Next: docker compose up -d --build"
    exit 0
fi
