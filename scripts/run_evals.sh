#!/usr/bin/env bash
# Launch evaluation for all models in parallel.
# Each model is expected to be served on its own vLLM port.
#
# SFT models (name ends in -SFT) are evaluated with the training prompts only:
#   arithmetic_mod_train  and  polynomial_mod_train
# Zero-shot models run all applicable prompts.
#
# Usage:
#   bash scripts/run_evals.sh
#   WORKERS=64 OUT_DIR=zero-shot-evals bash scripts/run_evals.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="${CONDA_ENV:-venv}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/zero-shot-evals}"
WORKERS="${WORKERS:-32}"
EXTRA_ARGS="${*}"

mkdir -p "${REPO_DIR}/logs"

# format: served_model_name:port
# Zero-shot models
# MODELS=(
#     "Qwen3.5-0.8B:8000"
#     "Qwen3.5-2B:8001"
#     "Qwen3.5-4B:8002"
#     "Qwen3.5-9B:8003"
# )

# SFT models
MODELS=(
    "Qwen3.5-0.8B-SFT:8000"
    "Qwen3.5-2B-SFT:8001"
    "Qwen3.5-4B-SFT:8002"
    "Qwen3.5-9B-SFT:8003"
)

echo "Starting evals..."

for entry in "${MODELS[@]}"; do
    IFS=: read -r model port <<< "$entry"
    log="${REPO_DIR}/logs/eval_${model}.log"

    # SFT models use only the two training-style prompts
    if [[ "$model" == *-SFT ]]; then
        PROMPTS_ARG="--prompts arithmetic_mod_train polynomial_mod_train"
    else
        PROMPTS_ARG=""
    fi

    echo "  ${model}  port=${port}  log=${log}"

    conda run -n "${CONDA_ENV}" python "${REPO_DIR}/eval.py" \
        --model      "${model}" \
        --port       "${port}" \
        --out_dir    "${OUT_DIR}" \
        --prompt_dir "${REPO_DIR}/prompts" \
        --workers    "${WORKERS}" \
        ${PROMPTS_ARG} \
        ${EXTRA_ARGS} \
        &> "${log}" &

    echo "  PID $!"
done

echo ""
echo "All evals running. Follow logs:"
echo "  tail -f ${REPO_DIR}/logs/eval_*.log"

wait
echo "All evals complete."
