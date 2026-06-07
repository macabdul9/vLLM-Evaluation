# Configuration for vLLM Server
MODEL="Qwen/Qwen3.5-0.8B"
PORT=8000
GPU_ID=0

# To avoid Out-of-Memory (OOM) errors, we limit the memory usage of the vLLM server.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}

# Export CUDA_VISIBLE_DEVICES to target the specific GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --served-model-name "$MODEL"