#!/usr/bin/env bash
# Launch evaluation for all models in parallel.
# Each model is expected to be served on its own vLLM port.
#
# Usage:
#   bash scripts/run_evals.sh
#   bash scripts/run_evals.sh --decode greedy
#   WORKERS=64 bash scripts/run_evals.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="${CONDA_ENV:-vllm-eval}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/zero-shot-evals}"
WORKERS="${WORKERS:-32}"
EXTRA_ARGS="${*}"

mkdir -p "${REPO_DIR}/logs"

run_eval() {
    local model=$1
    local port=$2
    local log="${REPO_DIR}/logs/eval_${model//\//_}.log"

    echo "  ${model}  port=${port}  log=${log}"

    conda run -n "${CONDA_ENV}" python "${REPO_DIR}/eval.py" \
        --model      "${model}" \
        --port       "${port}" \
        --out_dir    "${OUT_DIR}" \
        --prompt_dir "${REPO_DIR}/prompts" \
        --workers    "${WORKERS}" \
        ${EXTRA_ARGS} \
        &> "${log}" &

    echo "  PID $!"
}

echo "Starting evals..."
run_eval "Qwen3.5-0.8B"  8000
run_eval "Qwen3.5-2B"  8001
run_eval "Qwen3.5-4B"    8002
run_eval "Qwen3.5-9B"    8003

echo ""
echo "All evals running. Follow logs:"
echo "  tail -f ${REPO_DIR}/logs/eval_*.log"

wait
echo "All evals complete."
