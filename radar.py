"""
radar.py
Radar (spider) chart comparing zero-shot vs SFT across all evaluation splits.

Each panel shows one variant; lines = model sizes; axes = splits.
Accuracy is averaged across prompt types (mod + standard) per split.

Usage:
    python radar.py
    python radar.py --results_dir zero-shot-evals --out radar.pdf
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# CUD colorblind-friendly — one color per model size
MODEL_COLORS = {
    "Qwen3.5-0.8B": "#0072B2",   # blue
    "Qwen3.5-2B":   "#D55E00",   # vermillion
    "Qwen3.5-4B":   "#009E73",   # green
    "Qwen3.5-9B":   "#CC79A7",   # reddish purple
}
MODEL_ORDER  = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"]
MODEL_LABELS = {"Qwen3.5-0.8B": "0.8B", "Qwen3.5-2B": "2B",
                "Qwen3.5-4B":   "4B",   "Qwen3.5-9B": "9B"}

SPLIT_ORDER = [
    "iid_arithmetic", "iid_easy", "iid_medium", "iid_hard",
    "ood_arithmetic",  "ood_coefficient", "ood_degree", "ood_terms",
]
SPLIT_LABELS = {
    "iid_arithmetic":  "IID\nArith.",
    "iid_easy":        "IID\nEasy",
    "iid_medium":      "IID\nMed.",
    "iid_hard":        "IID\nHard",
    "ood_arithmetic":  "OOD\nArith.",
    "ood_coefficient": "OOD\nCoef.",
    "ood_degree":      "OOD\nDeg.",
    "ood_terms":       "OOD\nTerms",
}


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

        _, *prompt_parts = tag.split("_")
        prompt = "_".join(prompt_parts)

        corrects = [int(json.loads(l)["correct"])
                    for l in open(path)]
        records.append({"model_base": model_base, "variant": variant,
                        "prompt": prompt, "split": split, "corrects": corrects})
    return records


def mean_accuracy(records, model_base, variant, split):
    """Average accuracy across all prompt types for a given (model, variant, split)."""
    matching = [r for r in records
                if r["model_base"] == model_base
                and r["variant"] == variant
                and r["split"] == split]
    if not matching:
        return None
    return np.mean([np.mean(r["corrects"]) for r in matching])


def draw_radar(ax, values, angles, color, label, linestyle="-"):
    vals = values + values[:1]          # close the polygon
    ax.plot(angles, vals, color=color, linewidth=2.0,
            linestyle=linestyle, zorder=3, label=label)
    ax.fill(angles, vals, color=color, alpha=0.08, zorder=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="zero-shot-evals")
    parser.add_argument("--out",         default="radar.pdf")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        print("No results.jsonl files found.")
        return

    variants_present = sorted({r["variant"] for r in records})
    splits_present   = [s for s in SPLIT_ORDER
                        if any(r["split"] == s for r in records)]
    N      = len(splits_present)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close

    n_panels = len(variants_present)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.5 * n_panels, 4.5),
                             subplot_kw=dict(polar=True))
    if n_panels == 1:
        axes = [axes]

    variant_titles = {"zero-shot": "Zero-shot", "sft": "SFT"}

    for ax, variant in zip(axes, variants_present):
        for model_base in MODEL_ORDER:
            values = [mean_accuracy(records, model_base, variant, s)
                      for s in splits_present]
            if all(v is None for v in values):
                continue
            values = [v if v is not None else 0.0 for v in values]

            draw_radar(ax, values, angles,
                       color=MODEL_COLORS.get(model_base, "gray"),
                       label=MODEL_LABELS.get(model_base, model_base))

        # axes labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits_present],
                           fontsize=10)
        # radial grid
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.4)
        ax.spines["polar"].set_visible(False)

        ax.set_title(variant_titles.get(variant, variant),
                     fontsize=13, fontweight="bold", pad=18)

    # shared legend below
    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in MODEL_ORDER if m in MODEL_COLORS]
    fig.legend(handles=handles,
               loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), frameon=False, fontsize=11)

    fig.suptitle("Accuracy across splits  (avg. over prompt types)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=600, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
