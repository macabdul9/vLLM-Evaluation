"""
visualize.py
Bootstrap bar charts + radar chart for zero-shot / SFT evaluation results.

Usage:
    python visualize.py
    python visualize.py --results_dir zero-shot-evals --sample_size 500 --n_bootstrap 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Palette & constants
# ---------------------------------------------------------------------------

MOD_COLOR  = "#0072B2"   # blue       — mod prompt
STD_COLOR  = "#D55E00"   # vermillion — standard prompt
POLY_COLOR = "#009E73"   # green      — polynomial mod

SERIES_STYLE = {
    ("arithmetic_mod",      "zero-shot"): dict(color=MOD_COLOR,  hatch="///",  label="Arith. mod"),
    ("arithmetic_mod",      "sft"):       dict(color=MOD_COLOR,  hatch="\\\\", label="Arith. mod (SFT)"),
    ("arithmetic_standard", "zero-shot"): dict(color=STD_COLOR,  hatch="...",  label="Arith. standard"),
    ("arithmetic_standard", "sft"):       dict(color=STD_COLOR,  hatch="xx",   label="Arith. standard (SFT)"),
    ("polynomial_mod",      "zero-shot"): dict(color=POLY_COLOR, hatch="///",  label="Poly. mod"),
    ("polynomial_mod",      "sft"):       dict(color=POLY_COLOR, hatch="\\\\", label="Poly. mod (SFT)"),
    ("polynomial_standard", "zero-shot"): dict(color=STD_COLOR,  hatch="...",  label="Poly. standard"),
    ("polynomial_standard", "sft"):       dict(color=STD_COLOR,  hatch="xx",   label="Poly. standard (SFT)"),
}

MODEL_ORDER  = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"]
MODEL_LABELS = {"Qwen3.5-0.8B": "0.8B", "Qwen3.5-2B": "2B",
                "Qwen3.5-4B":   "4B",   "Qwen3.5-9B": "9B"}
MODEL_COLORS = {"Qwen3.5-0.8B": "#0072B2", "Qwen3.5-2B": "#D55E00",
                "Qwen3.5-4B":   "#009E73", "Qwen3.5-9B": "#CC79A7"}

SPLIT_ORDER = [
    "iid_arithmetic", "iid_easy", "iid_medium", "iid_hard",
    "ood_arithmetic",  "ood_coefficient", "ood_degree", "ood_terms",
]
SPLIT_LABELS = {
    "iid_arithmetic":  "IID\nArith.", "iid_easy":   "IID\nEasy",
    "iid_medium":      "IID\nMed.",   "iid_hard":   "IID\nHard",
    "ood_arithmetic":  "OOD\nArith.", "ood_coefficient": "OOD\nCoef.",
    "ood_degree":      "OOD\nDeg.",   "ood_terms":  "OOD\nTerms",
}


# ---------------------------------------------------------------------------
# Data loading & bootstrap
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> list[dict]:
    records = []
    base = Path(results_dir)
    if not base.exists():
        return records
    for path in base.rglob("results.jsonl"):
        parts      = path.parts
        model      = parts[-4]
        tag        = parts[-3]
        split      = parts[-2]
        variant    = "sft" if model.endswith("-SFT") else "zero-shot"
        model_base = model.removesuffix("-SFT")

        decode, *prompt_parts = tag.split("_")
        prompt = "_".join(prompt_parts)

        corrects = [int(json.loads(l)["correct"]) for l in open(path)]
        records.append({"model": model, "model_base": model_base, "variant": variant,
                        "decode": decode, "prompt": prompt, "split": split,
                        "corrects": corrects})
    return records


def bootstrap(corrects: list[int], sample_size: int, n_bootstrap: int,
              rng: np.random.Generator) -> tuple[float, float]:
    arr = np.array(corrects, dtype=float)
    sample_size = min(sample_size, len(arr))
    accs = np.array([arr[rng.integers(0, len(arr), size=sample_size)].mean()
                     for _ in range(n_bootstrap)])
    return float(accs.mean()), float(accs.std())


def mean_acc(records, model_base, variant, split) -> float | None:
    matching = [r for r in records
                if r["model_base"] == model_base
                and r["variant"] == variant
                and r["split"] == split]
    if not matching:
        return None
    return float(np.mean([np.mean(r["corrects"]) for r in matching]))


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def plot_bars(records: list[dict], args) -> None:
    all_splits = sorted({r["split"] for r in records})
    iid_splits = [s for s in all_splits if s.startswith("iid")]
    ood_splits  = [s for s in all_splits if s.startswith("ood")]
    n_cols      = max(len(iid_splits), len(ood_splits), 1)
    n_rows      = 1 if not (iid_splits and ood_splits) else 2
    row_splits  = [iid_splits, ood_splits] if n_rows == 2 else [iid_splits or ood_splits]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows), sharey=False)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for ax in axes.flat:
        ax.set_visible(False)

    seen_labels: dict[str, object] = {}

    for row_idx, split_group in enumerate(row_splits):
        for col_idx, split in enumerate(split_group):
            ax = axes[row_idx, col_idx]
            ax.set_visible(True)

            split_recs     = [r for r in records if r["split"] == split]
            models_present = [m for m in MODEL_ORDER
                              if any(r["model_base"] == m for r in split_recs)]
            series_present = sorted(
                {(r["prompt"], r["variant"]) for r in split_recs},
                key=lambda pv: (pv[0], pv[1] == "sft"),
            )
            n_series    = len(series_present)
            bar_width   = 0.22
            group_width = n_series * bar_width + 0.45
            x           = np.arange(len(models_present)) * group_width
            offsets     = (np.arange(n_series) - (n_series - 1) / 2) * bar_width

            for si, (prompt, variant) in enumerate(series_present):
                style = SERIES_STYLE.get((prompt, variant),
                                         dict(color="gray", hatch="", label=f"{prompt} ({variant})"))
                xs, means, stds = [], [], []
                for xi, model_base in enumerate(models_present):
                    match = [r for r in split_recs
                             if r["model_base"] == model_base
                             and r["prompt"] == prompt
                             and r["variant"] == variant]
                    if not match:
                        continue
                    xs.append(xi + offsets[si])
                    means.append(match[0]["mean"])
                    stds.append(match[0]["std"])

                if not means:
                    continue

                bar = ax.bar(xs, means, width=bar_width,
                             color=style["color"], hatch=style["hatch"],
                             edgecolor="black", zorder=3, label=style["label"])
                ax.errorbar(xs, means, yerr=stds, fmt="none",
                            color="black", capsize=3, linewidth=1.0, zorder=4)
                for xpos, mean in zip(xs, means):
                    ax.text(xpos, mean + 0.015, f"{mean*100:.1f}",
                            ha="center", va="bottom", fontsize=7)
                if style["label"] not in seen_labels:
                    seen_labels[style["label"]] = bar

            ax.set_title(split.replace("_", " "), fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present], fontsize=11)
            ax.set_xlabel("Model size", fontsize=12)
            ax.set_ylabel("Accuracy (↑)", fontsize=12)
            ax.set_ylim(0, 1.18)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)

    row_labels = (["IID", "OOD"] if n_rows == 2
                  else ["IID"] if iid_splits else ["OOD"])
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].annotate(
            label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-0.22, 0.5), textcoords="axes fraction",
            fontsize=14, fontweight="bold", va="center", ha="center", rotation=90,
        )

    has_sft = any(r["variant"] == "sft" for r in records)
    title   = "Zero-shot" + (" vs SFT" if has_sft else "")
    title  += f"  |  bootstrap n={args.n_bootstrap}, sample={args.sample_size}"

    fig.legend(seen_labels.values(), seen_labels.keys(),
               loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=len(seen_labels), frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=12, y=1.10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved → {args.out}")


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

def plot_radar(records: list[dict], args) -> None:
    variants_present = sorted({r["variant"] for r in records})
    splits_present   = [s for s in SPLIT_ORDER
                        if any(r["split"] == s for r in records)]
    N      = len(splits_present)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    n_panels = len(variants_present)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                             subplot_kw=dict(polar=True))
    if n_panels == 1:
        axes = [axes]

    variant_titles = {"zero-shot": "Zero-shot", "sft": "SFT"}

    for ax, variant in zip(axes, variants_present):
        for model_base in MODEL_ORDER:
            values = [mean_acc(records, model_base, variant, s) for s in splits_present]
            if all(v is None for v in values):
                continue
            values = [v if v is not None else 0.0 for v in values]
            vals   = values + values[:1]
            color  = MODEL_COLORS.get(model_base, "gray")
            label  = MODEL_LABELS.get(model_base, model_base)

            ax.plot(angles, vals, color=color, linewidth=2.0, zorder=3, label=label)
            ax.fill(angles, vals, color=color, alpha=0.08, zorder=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits_present], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.4)
        ax.spines["polar"].set_visible(False)
        ax.set_title(variant_titles.get(variant, variant),
                     fontsize=13, fontweight="bold", pad=18)

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in MODEL_ORDER if m in MODEL_COLORS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), frameon=False, fontsize=11)
    fig.suptitle("Accuracy across splits  (avg. over prompt types)",
                 fontsize=13, y=1.03)
    fig.tight_layout()

    radar_out = Path(args.out).with_stem(Path(args.out).stem + "_radar")
    fig.savefig(radar_out, dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved → {radar_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="zero-shot-evals")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out",         default="results.pdf")
    args = parser.parse_args()

    rng     = np.random.default_rng(args.seed)
    records = load_results(args.results_dir)

    if not records:
        print("No results.jsonl files found.")
        return

    for rec in records:
        rec["mean"], rec["std"] = bootstrap(
            rec["corrects"], args.sample_size, args.n_bootstrap, rng)

    plot_bars(records, args)
    plot_radar(records, args)

    print(f"\n{'Model':<25} {'Variant':<12} {'Prompt':<25} {'Split':<20} {'Mean':>7} {'±Std':>7}  N")
    print("-" * 100)
    for r in sorted(records, key=lambda r: (r["split"], r["prompt"], r["variant"],
                                             MODEL_ORDER.index(r["model_base"])
                                             if r["model_base"] in MODEL_ORDER else 99)):
        print(f"{r['model']:<25} {r['variant']:<12} {r['prompt']:<25} {r['split']:<20} "
              f"{r['mean']:>6.1%} {r['std']:>6.2%}  {len(r['corrects'])}")


if __name__ == "__main__":
    main()
