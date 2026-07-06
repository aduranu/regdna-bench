"""plots + markdown report for the zero-shot embedding variant-effect task.

reads the json summary written by run.py (--method embedding) and emits:
  - auroc_by_class.png       (sig-vs-background discrimination per CRE class)
  - auprc_by_class.png       (with the per-class positive-rate baseline)
  - correlation_by_class.png (spearman/pearson of cosine distance vs |beta|)
  - report.md                (numbers + drop accounting + interpretation)

usage:
    python examples/d3/plot_embedding_zero_shot.py \
        --results results/embedding_zero_shot/results_embedding_by_class.json \
        --outdir results/embedding_zero_shot
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# overall is drawn first and tinted differently from the per-class bars
OVERALL_KEY = "overall"
OVERALL_COLOR = "#444444"
CLASS_COLOR = "#4C72B0"
BASELINE_COLOR = "#C44E52"


def scored_classes(results):
    # per-class entries that actually carry metrics, sorted by descending n so
    # the better-supported classes read left-to-right.
    items = [
        (k, v) for k, v in results.items()
        if k != OVERALL_KEY and "metrics" in v
    ]
    items.sort(key=lambda kv: kv[1]["n"], reverse=True)

    return items


def ordered_entries(results):
    # overall first, then the scored classes by support.
    ordered = [(OVERALL_KEY, results[OVERALL_KEY])]
    ordered.extend(scored_classes(results))

    return ordered


def get(entry, group, metric):
    return entry.get("metrics", {}).get(group, {}).get(metric)


def bar_colors(labels):
    return [OVERALL_COLOR if k == OVERALL_KEY else CLASS_COLOR for k in labels]


def plot_auroc(entries, outpath):
    labels = [k for k, _ in entries]
    values = [get(v, "classification", "auroc") for _, v in entries]
    counts = [v["n"] for _, v in entries]

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.5))
    bars = ax.bar(labels, values, color=bar_colors(labels))

    # 0.5 is chance for auroc regardless of class balance
    ax.axhline(0.5, color=BASELINE_COLOR, linestyle="--", linewidth=1, label="chance (0.5)")

    for bar, val, n in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("Zero-shot embedding VEP: significant vs background by CRE class")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_auprc(entries, outpath):
    labels = [k for k, _ in entries]
    values = [get(v, "classification", "auprc") for _, v in entries]
    counts = [v["n"] for _, v in entries]

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.5))
    bars = ax.bar(labels, values, color=bar_colors(labels))

    for bar, val, n in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.3f}\n(n={n})", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("AUPRC")
    ax.set_title("Zero-shot embedding VEP: average precision by CRE class")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_correlation(entries, outpath):
    labels = [k for k, _ in entries]
    spearman = [get(v, "correlation", "spearman") for _, v in entries]
    pearson = [get(v, "correlation", "pearson") for _, v in entries]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.5))
    ax.bar(x - width / 2, spearman, width, label="spearman", color=CLASS_COLOR)
    ax.bar(x + width / 2, pearson, width, label="pearson", color="#55A868")

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("correlation (cosine distance vs |beta|)")
    ax.set_title("Zero-shot embedding VEP: effect-size correlation by CRE class")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fmt(value, places=4):
    return "n/a" if value is None else f"{value:.{places}f}"


def build_report(summary, plot_names):
    results = summary["results"]
    drops = summary.get("drops", {})
    backend = summary.get("genome_backend", "unknown")

    lines = []
    lines.append("# Zero-shot embedding variant-effect prediction (DART-Eval task 5)")
    lines.append("")
    lines.append("Model: zoonomia D3 (DNA discrete diffusion). Score: cosine distance "
                 "between mean-pooled last-layer ref/alt embeddings. A useful model "
                 "assigns larger distances to significant caQTLs than to background "
                 "SNPs in accessible peaks.")
    lines.append("")
    lines.append(f"- Genome backend: `{backend}`")
    lines.append(f"- Variant set: African caQTLs, held-out test chromosomes (chr22, chrX)")
    lines.append(f"- Element-anchored 350 bp windows (cCRE midpoint +/-175)")
    lines.append("")

    lines.append("## Results by CRE class")
    lines.append("")
    lines.append("| Class | n | AUROC | AUPRC | Pearson | Spearman |")
    lines.append("|---|---|---|---|---|---|")

    for key, entry in ordered_entries(results):
        auroc = get(entry, "classification", "auroc")
        auprc = get(entry, "classification", "auprc")
        pear = get(entry, "correlation", "pearson")
        spear = get(entry, "correlation", "spearman")
        lines.append(f"| {key} | {entry['n']} | {fmt(auroc)} | {fmt(auprc)} | "
                     f"{fmt(pear)} | {fmt(spear)} |")

    skipped = [(k, v) for k, v in results.items() if "skipped" in v]
    if skipped:
        lines.append("")
        lines.append("### Skipped classes (too few variants)")
        lines.append("")
        for key, entry in sorted(skipped, key=lambda kv: kv[1]["n"], reverse=True):
            lines.append(f"- `{key}`: n={entry['n']} ({entry['skipped']})")

    lines.append("")
    lines.append("## Variant drop accounting")
    lines.append("")
    if drops:
        for reason, count in sorted(drops.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Dataset (input)")
    lines.append("")
    lines.append("Distributions of the African caQTL set and the funnel down to the eval "
                 "subset are in `dataset_stats.md`; the figure below covers label balance, "
                 "effect size, significance, peak distance, and per-chromosome counts.")
    lines.append("")
    lines.append("![dataset_eda.png](dataset_eda.png)")
    lines.append("")

    lines.append("## Per-class metric figures")
    lines.append("")
    captions = {
        "auroc_by_class.png": "AUROC per CRE class (sig vs background), chance = 0.5.",
        "auprc_by_class.png": "Average precision per CRE class.",
        "correlation_by_class.png": "Cosine distance vs |beta| correlation per class.",
    }
    for name in plot_names:
        lines.append(f"![{name}]({name})")
        lines.append(f"*{captions.get(name, name)}*")
        lines.append("")

    lines.append("## Embedding space and cosine-distance score")
    lines.append("")
    lines.append("PCA/t-SNE of the mean-pooled ref embeddings (colored by CRE class, label, "
                 "and cosine distance), plus the cosine-distance distributions the score is "
                 "built from.")
    lines.append("")
    lines.append("![embeddings_lowdim.png](embeddings_lowdim.png)")
    lines.append("")

    lines.append("## Where the metrics come from")
    lines.append("")
    lines.append("ROC and precision-recall curves (overall + per class) behind AUROC/AUPRC, "
                 "the sig-vs-background score separation AUROC integrates, and the "
                 "cosine-distance vs |beta| scatter behind spearman.")
    lines.append("")
    lines.append("![classification_diag.png](classification_diag.png)")
    lines.append("")

    overall = results[OVERALL_KEY]
    overall_auroc = get(overall, "classification", "auroc")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"Overall AUROC is {fmt(overall_auroc)} on n={overall['n']} variants. "
                 "Values near 0.5 indicate the unsupervised embedding-distance prior "
                 "does not separate significant from background caQTLs on this set; "
                 "per-class entries show where, if anywhere, the signal concentrates.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="plots + report for embedding zero-shot VEP")
    parser.add_argument("--results", default="results/embedding_zero_shot/results_embedding_by_class.json")
    parser.add_argument("--outdir", default="results/embedding_zero_shot")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as handle:
        summary = json.load(handle)

    entries = ordered_entries(summary["results"])

    plot_names = ["auroc_by_class.png", "auprc_by_class.png", "correlation_by_class.png"]
    plot_auroc(entries, outdir / plot_names[0])
    plot_auprc(entries, outdir / plot_names[1])
    plot_correlation(entries, outdir / plot_names[2])

    report = build_report(summary, plot_names)
    report_path = outdir / "report.md"
    with open(report_path, "w") as handle:
        handle.write(report)

    print(f"wrote {report_path} and {len(plot_names)} plots to {outdir}")


if __name__ == "__main__":
    main()
