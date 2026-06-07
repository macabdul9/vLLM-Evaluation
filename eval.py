"""
eval.py
Zero-shot evaluation of vLLM-served models on LLMMath HuggingFace evaluation splits.

Output structure:
    zero-shot-evals/
        Qwen3.5-0.8B/
            greedy_arithmetic_standard/   results.jsonl  metrics.json
            greedy_polynomial_mod/        results.jsonl  metrics.json
            best_of_n_arithmetic_mod/     results.jsonl  metrics.json
            ...
        Qwen3.5-2B/
            ...
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datasets
from math_verify import parse, verify
from openai import OpenAI


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompt_dir: str) -> dict[str, str]:
    """Return {stem: template_string} for every .txt file in prompt_dir."""
    prompts = {}
    for path in sorted(Path(prompt_dir).glob("*.txt")):
        prompts[path.stem] = path.read_text()
    return prompts


def build_prompt(template: str, expression: str, mod_base: int = 10000) -> str:
    return template.format(expression=expression, mod_base=mod_base, result="{result}")


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

MOD = 10000


def extract_answer(text: str) -> str | None:
    """Extract the content of \\boxed{...} from model output.

    Uses balanced-brace matching so that exponents like x^{10} inside the
    boxed expression don't prematurely terminate the match.
    After extraction, normalises x^{n} → x^n to match the dataset format.
    """
    idx = text.find("\\boxed{")
    if idx != -1:
        start = idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            content = text[start:i - 1].strip()
            return _normalise(content)

    # fallback: last line starting with "Answer:"
    for line in reversed(text.strip().splitlines()):
        if line.lower().startswith("answer:"):
            return _normalise(line.split(":", 1)[1].strip())

    # fallback: model output is just the bare answer (single non-empty line)
    stripped = text.strip()
    if stripped and "\n" not in stripped:
        return _normalise(stripped)

    return None


def _normalise(expr: str) -> str:
    """Normalise a polynomial/arithmetic string extracted from model output."""
    # x^{10} → x^10  (LaTeX braces around exponent)
    expr = re.sub(r"\^\{(\d+)\}", r"^\1", expr)
    # 3x → 3*x  (missing multiplication sign between coefficient and variable)
    expr = re.sub(r"(\d)(x)", r"\1*\2", expr)
    return expr.strip()


def _apply_mod(predicted: str, rec_type: str) -> str | None:
    """Reduce a predicted answer mod 10000 to match final_answer ground truth."""
    if rec_type == "arithmetic":
        try:
            return str(int(predicted.strip()) % MOD)
        except ValueError:
            return None
    else:
        # polynomial: mod each coefficient via sympy
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import (
                parse_expr, standard_transformations, convert_xor,
            )
            x = sp.Symbol("x")
            transforms = standard_transformations + (convert_xor,)
            expanded = sp.expand(parse_expr(predicted, transformations=transforms,
                                             local_dict={"x": x}))
            poly = sp.Poly(expanded, x)
            degree = poly.degree()
            coeffs = [int(c) % MOD for c in poly.all_coeffs()]
            terms = []
            for i, c in enumerate(coeffs):
                d = degree - i
                if c == 0:
                    continue
                if d == 0:
                    terms.append(str(c))
                elif d == 1:
                    terms.append("x" if c == 1 else f"{c}*x")
                else:
                    terms.append(f"x^{d}" if c == 1 else f"{c}*x^{d}")
            return " + ".join(terms) if terms else "0"
        except Exception:
            return None


def is_correct(predicted: str | None, expected: str,
               rec_type: str = "polynomial") -> bool:
    if predicted is None:
        return False

    def _compare(a: str, b: str) -> bool:
        try:
            return verify(parse(f"${b}$"), parse(f"${a}$"))
        except Exception:
            return a.strip() == b.strip()

    # Direct comparison first (model produced the mod answer correctly)
    if _compare(predicted, expected):
        return True

    # Fallback: apply mod 10000 to predicted and compare.
    # Handles the case where the model gives the mathematically correct raw
    # answer without applying modular reduction.
    pred_mod = _apply_mod(predicted, rec_type)
    if pred_mod is not None:
        return _compare(pred_mod, expected)

    return False


# ---------------------------------------------------------------------------
# vLLM client helpers
# ---------------------------------------------------------------------------

def greedy_request(
    client: OpenAI, model: str, prompt: str, max_tokens: int,
    enable_thinking: bool = False,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    return resp.choices[0].message.content


def best_of_n_request(
    client: OpenAI, model: str, prompt: str, max_tokens: int, n: int,
    enable_thinking: bool = False,
) -> tuple[str | None, list[str]]:
    """Return (majority-voted extracted answer, list of all raw response texts)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.8,
        n=n,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    contents = [c.message.content for c in resp.choices]
    answers = [extract_answer(c) for c in contents]
    votes: dict[str, int] = {}
    for a in answers:
        if a is not None:
            votes[a] = votes.get(a, 0) + 1
    best = max(votes, key=lambda k: votes[k]) if votes else None
    return best, contents


# ---------------------------------------------------------------------------
# Single split evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    client: OpenAI,
    model_name: str,
    split_ds: datasets.Dataset,
    split_name: str,
    prompt_name: str,
    prompt_template: str,
    decode_mode: str,
    n: int,
    max_tokens: int,
    workers: int,
    out_dir: Path,
    enable_thinking: bool = False,
    max_samples: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists():
        print(f"  [skip] {out_dir} already done")
        return json.loads(metrics_path.read_text())

    rows = list(split_ds)
    if max_samples is not None:
        rows = rows[:max_samples]

    results: list[dict | None] = [None] * len(rows)
    correct = 0

    def run_one(row):
        expr = row["input"]
        expected = row["final_answer"]
        prompt = build_prompt(prompt_template, expr)

        if decode_mode == "greedy":
            raw_output = greedy_request(client, model_name, prompt, max_tokens, enable_thinking)
            pred = extract_answer(raw_output)
        else:
            pred, raw_output = best_of_n_request(client, model_name, prompt, max_tokens, n, enable_thinking)

        ok = is_correct(pred, expected, rec_type=row.get("type", "polynomial"))
        return {
            "input": expr,
            "expected": expected,
            "predicted": pred,
            "correct": ok,
            "raw_output": raw_output,
            "split": split_name,
            "prompt": prompt_name,
            "decode": decode_mode,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, row): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            if result["correct"]:
                correct += 1

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    metrics = {
        "model": model_name,
        "split": split_name,
        "prompt": prompt_name,
        "decode": decode_mode,
        "n": n if decode_mode == "best_of_n" else 1,
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  {decode_mode}/{prompt_name}/{split_name}: {correct}/{len(results)} = {metrics['accuracy']:.3f}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True, help="vLLM server port")
    parser.add_argument("--model", required=True, help="served-model-name as set in vLLM")
    parser.add_argument("--dataset", default="LLMsHub/LLMMath-Eval", help="HF dataset repo id")
    parser.add_argument("--prompt_dir", default="prompts")
    parser.add_argument("--out_dir", default="zero-shot-evals")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Subset of splits to run. Default: all.")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Subset of prompt names to run. Default: all.")
    parser.add_argument("--decode", nargs="+", default=["greedy"], choices=["greedy", "best_of_n"],
                        help="Decoding strategy to use. Default: greedy.")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable chain-of-thought thinking mode (disabled by default).")
    parser.add_argument("--best_of_n", type=int, default=None,
                        help="Number of samples for best-of-n decoding (default: 4 for best_of_n)")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel threads per split (vLLM handles concurrency server-side)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Max examples per split (default: full split).")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="none")

    print(f"Loading dataset {args.dataset} ...")
    dd = datasets.load_dataset(args.dataset)

    prompt_templates = load_prompts(args.prompt_dir)
    if args.prompts:
        prompt_templates = {k: v for k, v in prompt_templates.items() if k in args.prompts}

    splits_to_run = args.splits or list(dd.keys())
    model_dir = Path(args.out_dir) / args.model

    all_metrics = []

    for split_name in splits_to_run:
        if split_name not in dd:
            print(f"Warning: split '{split_name}' not in dataset, skipping")
            continue
        split_ds = dd[split_name]

        for prompt_name, template in prompt_templates.items():
            # skip prompt/split type mismatches (arithmetic prompt on polynomial split etc.)
            split_type = "arithmetic" if "arithmetic" in split_name else "polynomial"
            if split_type not in prompt_name:
                continue

            for decode_mode in args.decode:
                tag = f"{decode_mode}_{prompt_name}"
                out_dir = model_dir / tag / split_name

                metrics = evaluate_split(
                    client=client,
                    model_name=args.model,
                    split_ds=split_ds,
                    split_name=split_name,
                    prompt_name=prompt_name,
                    prompt_template=template,
                    decode_mode=decode_mode,
                    n=args.best_of_n,
                    max_tokens=args.max_tokens,
                    workers=args.workers,
                    out_dir=out_dir,
                    enable_thinking=args.enable_thinking,
                    max_samples=args.samples,
                )
                all_metrics.append(metrics)

    # summary
    summary_path = model_dir / "summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
