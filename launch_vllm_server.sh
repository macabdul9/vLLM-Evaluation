#!/bin/bash

GPU_MEMORY_UTILIZATION=0.85

# zero-shot models
# declare -A MODELS=(
#     [0]="Qwen/Qwen3.5-0.8B:Qwen3.5-0.8B:8000"
#     [1]="Qwen/Qwen3.5-2B:Qwen3.5-2B:8001"
#     [2]="Qwen/Qwen3.5-4B:Qwen3.5-4B:8002"
#     [3]="Qwen/Qwen3.5-9B:Qwen3.5-9B:8003"
# )

# sft models  format: model_path:served_name:port:tokenizer
declare -A MODELS=(
    [0]="/data/user_data/abdulw/llmmath/Qwen3.5-0.8B-LLMMath-SFT:Qwen3.5-0.8B-SFT:8000:Qwen/Qwen3.5-0.8B"
    [1]="/data/user_data/abdulw/llmmath/Qwen3.5-2B-LLMMath-SFT:Qwen3.5-2B-SFT:8001:Qwen/Qwen3.5-2B"
    [2]="/data/user_data/abdulw/llmmath/Qwen3.5-4B-LLMMath-SFT:Qwen3.5-4B-SFT:8002:Qwen/Qwen3.5-4B"
    [3]="/data/user_data/abdulw/llmmath/Qwen3.5-9B-LLMMath-SFT/checkpoint-2000:Qwen3.5-9B-SFT:8003:Qwen/Qwen3.5-9B"
)

mkdir -p logs

for gpu in "${!MODELS[@]}"; do
    IFS=: read -r model_id served_name port tokenizer <<< "${MODELS[$gpu]}"

    echo "Launching ${served_name} on GPU ${gpu}, port ${port}"

    CUDA_VISIBLE_DEVICES=$gpu nohup python3 -m vllm.entrypoints.openai.api_server \
        --model              "$model_id" \
        --served-model-name  "$served_name" \
        --port               "$port" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --trust-remote-code \
        ${tokenizer:+--tokenizer "$tokenizer"} \
        &> "logs/logs_${served_name,,}.log" &

    echo "  PID $!"
done

echo ""
echo "All servers launched. Follow logs:"
echo "  tail -f logs/logs_*.log"
