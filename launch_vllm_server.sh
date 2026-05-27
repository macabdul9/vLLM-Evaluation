#!/bin/bash

GPU_MEMORY_UTILIZATION=0.85

# Model 1 -> GPU 0
export CUDA_VISIBLE_DEVICES=0 
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-0.8B \
    --served-model-name Qwen3.5-0.8B \
    --port 8000 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    &> logs/logs_qwen_0.8b.log &

# Model 2 -> GPU 1
export CUDA_VISIBLE_DEVICES=1
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-2B \
    --served-model-name Qwen3.5-2B \
    --port 8001 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    &> logs/logs_qwen_2b.log &

# Model 3 -> GPU 2
export CUDA_VISIBLE_DEVICES=2 
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-4B \
    --served-model-name Qwen3.5-4B \
    --port 8002 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    &> logs/logs_qwen_4b.log &

# Model 4 -> GPU 3
export CUDA_VISIBLE_DEVICES=3
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --served-model-name Qwen3.5-9B \
    --port 8003 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --trust-remote-code \
    &> logs/logs_qwen_9b.log &