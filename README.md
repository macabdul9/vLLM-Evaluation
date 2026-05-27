# vLLM Evaluation — LLMMath Zero-Shot Benchmark

A framework for running zero-shot evaluations of open-weight models served via vLLM. Currently evaluates Qwen3.5 models on a custom math benchmark ([LLMsHub/LLMMath-Eval](https://huggingface.co/datasets/LLMsHub/LLMMath-Eval)) covering arithmetic and polynomial tasks, with support for both thinking and non-thinking modes, greedy decoding, and best-of-N majority voting.

---

## Environment Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the environment

```bash
conda create -n vllm-eval python=3.11 -y
conda activate vllm-eval
```

### 3. Install PyTorch (CUDA 12.x)

```bash
uv pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install vLLM

```bash
uv pip install vllm==0.17.1
```

> vLLM will try to pull its own `torch` — install torch first (step 3) to avoid version conflicts.

### 5. Install evaluation dependencies

```bash
uv pip install \
    openai==2.24.0 \
    datasets==4.8.2 \
    sympy==1.14.0 \
    math-verify==0.9.0 \
    transformers==4.57.6
```

### Full package versions (pinned)

| Package | Version |
|---|---|
| `vllm` | 0.17.1 |
| `torch` | 2.10.0 |
| `transformers` | 4.57.6 |
| `openai` | 2.24.0 |
| `datasets` | 4.8.2 |
| `sympy` | 1.14.0 |
| `math-verify` | 0.9.0 |

---

## Models

Four Qwen3.5 checkpoints are evaluated, one per GPU:

| Model | HuggingFace ID | Port | GPU |
|---|---|---|---|
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` | 8000 | 0 |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | 8001 | 1 |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | 8002 | 2 |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | 8003 | 3 |

Model weights are downloaded automatically from HuggingFace on first launch. Set `HF_HOME` to control the cache location:

```bash
export HF_HOME=/path/to/hf_cache
```

---

## Project Structure

```
vLLMEvaluation/
├── launch_vllm_server.sh   # Start all 4 vLLM servers (one per GPU)
├── eval.py                 # Core evaluation script
├── debug_eval.py           # End-to-end test on a single model and sample
├── test_vllm.py            # Minimal connectivity test for a running server
├── test_vllm.sh            # Shell-based server smoke test
├── scripts/
│   └── run_evals.sh        # Launch eval.py for all 4 models in parallel
├── prompts/
│   ├── arithmetic_mod.txt      # Arithmetic with modular reduction
│   ├── arithmetic_standard.txt # Standard arithmetic
│   ├── polynomial_mod.txt      # Polynomial simplification with mod
│   └── polynomial_standard.txt # Standard polynomial simplification
├── logs/
│   ├── logs_qwen_*.log     # vLLM server logs
│   └── eval_*.log          # Per-model evaluation logs
├── zero-shot-evals/        # Output directory (created at runtime)
│   └── <model>/
│       └── <decode>_<prompt>/
│           └── <split>/
│               ├── results.jsonl
│               └── metrics.json
└── models.txt              # List of HuggingFace model IDs
```

---

## Quickstart

### Step 1 — Launch the vLLM servers

```bash
mkdir -p logs
bash launch_vllm_server.sh
```

This starts 4 background vLLM processes (one per GPU). Wait ~60–90 seconds for all servers to finish loading before running evaluations. Check readiness with:

```bash
tail -f logs/logs_qwen_0.8b.log
# Ready when you see: "Application startup complete."
```

Or verify all four ports are responding:

```bash
for port in 8000 8001 8002 8003; do
    curl -s http://localhost:$port/v1/models | python3 -m json.tool | grep '"id"'
done
```

### Step 2 — Run a quick sanity check

Run one sample through the full pipeline before committing to a multi-hour eval run:

```bash
conda run -n vllm-eval python debug_eval.py \
    --port 8000 --model Qwen3.5-0.8B --samples 1
```

### Step 3 — Run the full evaluation

```bash
nohup bash scripts/run_evals.sh &> logs/run_evals.log &
```

This runs all 4 models in parallel. Full evaluation takes several hours depending on dataset size and model speed.

---

## Monitoring Progress

```bash
# Follow per-model evaluation logs (written by eval.py)
tail -f logs/eval_Qwen3.5-0.8B.log
tail -f logs/eval_Qwen3.5-2B.log
tail -f logs/eval_Qwen3.5-4B.log
tail -f logs/eval_Qwen3.5-9B.log

# Follow vLLM server request logs (POST lines appear as requests arrive)
tail -f logs/logs_qwen_0.8b.log

# Check GPU utilisation
watch -n2 nvidia-smi

# Count completed result files
find zero-shot-evals -name "metrics.json" | wc -l
```

---

## Advanced Usage

### Greedy decoding only (skip best-of-N)

```bash
bash scripts/run_evals.sh --decode greedy
```

### Single model, single split

```bash
conda run -n vllm-eval python eval.py \
    --model Qwen3.5-0.8B \
    --port  8000 \
    --splits iid_arithmetic \
    --decode greedy
```

### Enable Qwen Thinking Mode

```bash
conda run -n vllm-eval python eval.py \
    --model Qwen3.5-0.8B \
    --port  8000 \
    --enable_thinking
```

### Change number of workers (parallel threads per split)

```bash
WORKERS=64 bash scripts/run_evals.sh
```

### Specify output directory

```bash
OUT_DIR=/scratch/results bash scripts/run_evals.sh
```

---

## eval.py — CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Served model name (must match `--served-model-name` in vLLM) |
| `--port` | *(required)* | vLLM server port |
| `--host` | `localhost` | vLLM server host |
| `--dataset` | `LLMsHub/LLMMath-Eval` | HuggingFace dataset repo ID |
| `--prompt_dir` | `prompts/` | Directory containing `.txt` prompt templates |
| `--out_dir` | `zero-shot-evals/` | Root output directory |
| `--splits` | all splits | Subset of dataset splits to evaluate |
| `--prompts` | all prompts | Subset of prompt templates to use |
| `--decode` | `greedy best_of_n` | Decoding strategies to run |
| `--best_of_n` | `8` | Number of samples for best-of-N voting |
| `--max_tokens` | `4096` | Maximum tokens per completion |
| `--workers` | `32` | Parallel threads per split |
| `--enable_thinking` | `False` | Enable Qwen thinking mode |

---

## Prompt Templates

Prompts live in `prompts/` as `.txt` files. The filename stem becomes the prompt name used in output paths. Templates support three placeholders:

| Placeholder | Value |
|---|---|
| `{expression}` | The input expression from the dataset |
| `{mod_base}` | Modular base (default: 10000) |
| `{result}` | Literal `{result}` passthrough for formatting |

The evaluation pipeline automatically matches prompts to splits by type: `arithmetic_*` prompts run on arithmetic splits and `polynomial_*` prompts run on polynomial splits.

---

## Output Format

Results are written to `zero-shot-evals/<model>/<decode>_<prompt>/<split>/`:

**`metrics.json`** — per-split accuracy summary:
```json
{
  "model": "Qwen3.5-0.8B",
  "split": "iid_arithmetic",
  "prompt": "arithmetic_mod",
  "decode": "greedy",
  "n": 1,
  "total": 10000,
  "correct": 7339,
  "accuracy": 0.7339
}
```

**`results.jsonl`** — one JSON object per sample:
```json
{
  "input": "23 - 100",
  "expected": "9923",
  "predicted": "9923",
  "correct": true,
  "raw_output": "Step 1: -77\n-77 mod 10000 = 9923\nAnswer: \\boxed{9923}",
  "split": "iid_arithmetic",
  "prompt": "arithmetic_mod",
  "decode": "greedy"
}
```

A `summary.json` aggregating all splits is written to `zero-shot-evals/<model>/summary.json` when evaluation completes.

---

## Important: Model Name Convention

The `--served-model-name` in `launch_vllm_server.sh` uses short names (e.g., `Qwen3.5-0.8B`), **not** the full HuggingFace path (`Qwen/Qwen3.5-0.8B`). The `--model` argument passed to `eval.py` must match the served name exactly, or the API will return a 404.

| Script | Name used |
|---|---|
| `launch_vllm_server.sh` `--served-model-name` | `Qwen3.5-0.8B` |
| `eval.py` `--model` / `scripts/run_evals.sh` | `Qwen3.5-0.8B` |

---

## Stopping the Servers

```bash
pkill -f "vllm.entrypoints.openai.api_server"
```
