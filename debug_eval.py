"""
debug_eval.py
Smoke-test the evaluation pipeline on a tiny sample before running at scale.

Usage:
    python debug_eval.py
    python debug_eval.py --port 8001 --model Qwen3.5-2B --samples 3
"""

import argparse
import sys

import datasets
from openai import OpenAI

from eval import load_prompts, build_prompt, extract_answer, is_correct, greedy_request, best_of_n_request


def check_server(client, model):
    try:
        names = [m.id for m in client.models.list().data]
        status = "OK" if model in names else f"WARNING: not found in {names}"
        print(f"server: {status}")
    except Exception as e:
        print(f"server: FAILED — {e}")
        sys.exit(1)


def run_sample(client, model, row, prompt_name, template, decode_mode, n, max_tokens, enable_thinking):
    expr = row["input"]
    expected = row["final_answer"]
    prompt = build_prompt(template, expr)

    if decode_mode == "greedy":
        raw = greedy_request(client, model, prompt, max_tokens, enable_thinking)
        pred = extract_answer(raw)
    else:
        pred, raws = best_of_n_request(client, model, prompt, max_tokens, n, enable_thinking)
        raw = raws[0]

    ok = is_correct(pred, expected, rec_type=row.get("type", "polynomial"))

    print(f"  input    : {expr[:80]}")
    print(f"  expected : {expected}")
    print(f"  predicted: {pred}")
    print(f"  correct  : {ok}")
    print(f"  response : {raw[:300].replace(chr(10), ' ')}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="LLMsHub/LLMMath-Eval")
    parser.add_argument("--prompt_dir", default="prompts")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--best_of_n", type=int, default=4)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="none")

    check_server(client, args.model)

    dd = datasets.load_dataset(args.dataset)
    prompts = load_prompts(args.prompt_dir)

    results = []

    for split_name, split_ds in dd.items():
        split_type = "arithmetic" if "arithmetic" in split_name else "polynomial"
        rows = list(split_ds.select(range(min(args.samples, len(split_ds)))))

        for prompt_name, template in prompts.items():
            if split_type not in prompt_name:
                continue

            for decode_mode in ["greedy"]:
                print(f"\n[{split_name}] [{prompt_name}] [{decode_mode}]")
                for row in rows:
                    ok = run_sample(
                        client, args.model, row,
                        prompt_name, template,
                        decode_mode, args.best_of_n, args.max_tokens,
                        args.enable_thinking,
                    )
                    results.append(ok)

    total = len(results)
    correct = sum(results)
    print(f"\n{correct}/{total} correct ({correct/total:.1%})" if total else "\nNo samples ran.")


if __name__ == "__main__":
    main()
