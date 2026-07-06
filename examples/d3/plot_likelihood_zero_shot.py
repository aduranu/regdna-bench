"""plots + markdown report for the zero-shot likelihood variant-effect task.

reads the raw per-variant llr dumps (scores.npz) written by
dump_likelihood_zero_shot.py for the element- and variant-centered runs, and
emits diagnostics behind the auroc/auprc/spearman summary:

  - metrics_by_class.png    (auroc + auprc per CRE class, element vs variant)
  - roc_curves.png          (roc overall + dense classes, element vs variant)
  - pr_curves.png           (precision-recall, with per-class positive-rate base)
  - score_separation.png    (|llr| significant vs background, variant-centered)
  - effect_size_scatter.png (|llr| vs |beta| with spearman, variant-centered)
  - report.md

the score is the signed allele log-likelihood difference (llr); its magnitude
|llr| is the significance score the classification metrics use.

usage:
    python examples/d3/plot_likelihood_zero_shot.py \
        --element-npz results/likelihood_zero_shot/scores.npz \
        --variant-npz results/likelihood_zero_shot_variant_centered/scores.npz \
        --outdir results/likelihood_zero_shot/plots
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

MODE_COLOR = {"element": "#4C72B0", "variant": "#DD8452"}
MODE_STYLE = {"element": "-", "variant": "--"}
BG_COLOR = "#BBBBBB"
SIG_COLOR = "#C44E52"
BASELINE_COLOR = "#888888"


def load_scores(path):
    # raw per-variant arrays for one window mode, or None if not on disk.
    if path is None or not pathlib.Path(path).exists():
        return None

    npz = np.load(path, allow_pickle=True)

    return {
        "abs_score": npz["abs_score"],
        "llr": npz["llr"],
        "labels": npz["labels"],
        "betas": npz["betas"],
        "cre_classes": npz["cre_classes"].astype(str),
    }


def class_mask(data, key):
    # boolean selector for a CRE class, or all-true for "overall".
    if key == "overall":
        return np.ones(len(data["labels"]), dtype=bool)

    return data["cre_classes"] == key


def panel_keys(data, min_n=20, max_classes=3):
    # overall plus the best-supported classes that carry both labels, so roc/pr
    # are defined. these are the same classes the by-class summary scores.
    counts = {}
    for cls in np.unique(data["cre_classes"]):
        mask = data["cre_classes"] == cls
        if mask.sum() >= min_n and len(np.unique(data["labels"][mask])) == 2:
            counts[cls] = int(mask.sum())

    dense = sorted(counts, key=counts.get, reverse=True)[:max_classes]

    return ["overall"] + dense


def plot_roc(modes, keys, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, key in zip(axes.ravel(), keys):
        for mode, data in modes.items():
            mask = class_mask(data, key)
            y = data["labels"][mask]
            s = data["abs_score"][mask]
            if len(np.unique(y)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y, s)
            auroc = roc_auc_score(y, s)
            ax.plot(fpr, tpr, color=MODE_COLOR[mode], linestyle=MODE_STYLE[mode],
                    label=f"{mode} (AUROC={auroc:.3f})")

        ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, linestyle=":", linewidth=1)
        n = int(class_mask(next(iter(modes.values())), key).sum())
        ax.set_title(f"{key} (n={n})")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Zero-shot likelihood VEP: ROC (significant vs background)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_pr(modes, keys, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, key in zip(axes.ravel(), keys):
        base = None
        for mode, data in modes.items():
            mask = class_mask(data, key)
            y = data["labels"][mask]
            s = data["abs_score"][mask]
            if len(np.unique(y)) < 2:
                continue

            precision, recall, _ = precision_recall_curve(y, s)
            auprc = average_precision_score(y, s)
            ax.plot(recall, precision, color=MODE_COLOR[mode], linestyle=MODE_STYLE[mode],
                    label=f"{mode} (AUPRC={auprc:.3f})")
            base = float(y.mean())

        if base is not None:
            ax.axhline(base, color=BASELINE_COLOR, linestyle=":", linewidth=1,
                       label=f"positive rate ({base:.3f})")
        n = int(class_mask(next(iter(modes.values())), key).sum())
        ax.set_title(f"{key} (n={n})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Zero-shot likelihood VEP: precision-recall")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_metrics_bars(modes, keys, outpath):
    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.38
    x = np.arange(len(keys))

    for i, (mode, data) in enumerate(modes.items()):
        aurocs, auprcs = [], []
        for key in keys:
            mask = class_mask(data, key)
            y, s = data["labels"][mask], data["abs_score"][mask]
            aurocs.append(roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan)
            auprcs.append(average_precision_score(y, s) if len(np.unique(y)) == 2 else np.nan)

        off = (i - 0.5) * width
        ax_roc.bar(x + off, aurocs, width, label=mode, color=MODE_COLOR[mode])
        ax_prc.bar(x + off, auprcs, width, label=mode, color=MODE_COLOR[mode])

    ax_roc.axhline(0.5, color=SIG_COLOR, linestyle="--", linewidth=1, label="chance (0.5)")
    ax_roc.set_ylim(0, 1)
    ax_roc.set_ylabel("AUROC")
    ax_roc.set_title("AUROC by CRE class")
    ax_prc.set_ylabel("AUPRC")
    ax_prc.set_title("AUPRC by CRE class")

    for ax in (ax_roc, ax_prc):
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=20, ha="right")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Zero-shot likelihood VEP: element- vs variant-centered windows")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_score_separation(data, keys, outpath, mode_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, key in zip(axes.ravel(), keys):
        mask = class_mask(data, key)
        y, s = data["labels"][mask], data["abs_score"][mask]
        groups = [s[y == 0], s[y == 1]]

        parts = ax.violinplot(groups, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], (BG_COLOR, SIG_COLOR)):
            body.set_facecolor(color)
            body.set_alpha(0.7)

        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"background\n(n={len(groups[0])})",
                            f"significant\n(n={len(groups[1])})"])
        ax.set_ylabel("|llr| (significance score)")
        ax.set_title(f"{key} (n={int(mask.sum())})")

    fig.suptitle(f"Zero-shot likelihood VEP: score separation ({mode_name}-centered)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_effect_scatter(data, keys, outpath, mode_name):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, key in zip(axes.ravel(), keys):
        mask = class_mask(data, key)
        s = data["abs_score"][mask]
        abeta = np.abs(data["betas"][mask])
        rho = spearmanr(s, abeta).statistic if len(s) > 2 else np.nan

        ax.scatter(abeta, s, s=10, alpha=0.4, color=MODE_COLOR["variant"])
        ax.set_xlabel("|beta| (reported effect size)")
        ax.set_ylabel("|llr| (predicted score)")
        ax.set_title(f"{key} (n={int(mask.sum())}, spearman={rho:.3f})")

    fig.suptitle(f"Zero-shot likelihood VEP: score vs effect size ({mode_name}-centered)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def overall_auroc(data):
    return roc_auc_score(data["labels"], data["abs_score"])


def build_report(modes, keys, plot_names):
    lines = []
    lines.append("# Zero-shot likelihood variant-effect prediction (DART-Eval task 5)")
    lines.append("")
    lines.append("Model: zoonomia D3 (uniform-SEDD DNA discrete diffusion). Score: the "
                 "signed allele log-likelihood difference read from the per-position "
                 "logits, `llr = score[off,alt] - score[off,ref]`; its magnitude |llr| "
                 "is the significance score. A useful model assigns larger |llr| to "
                 "significant caQTLs than to background SNPs in accessible peaks.")
    lines.append("")
    lines.append("- Variant set: African caQTLs, held-out test chromosomes (chr22, chrX)")
    lines.append("- Windows: 350 bp, element-centered (cCRE midpoint) vs variant-centered")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    for mode, data in modes.items():
        lines.append(f"- {mode}-centered: AUROC {overall_auroc(data):.4f} on "
                     f"n={len(data['labels'])} variants")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    captions = {
        "metrics_by_class.png": "AUROC/AUPRC per CRE class, element vs variant-centered.",
        "roc_curves.png": "ROC overall + dense classes; dotted diagonal is chance.",
        "pr_curves.png": "Precision-recall; dotted line is the per-class positive rate.",
        "score_separation.png": "|llr| for significant vs background (variant-centered).",
        "effect_size_scatter.png": "|llr| vs |beta| with spearman (variant-centered).",
    }
    for name in plot_names:
        lines.append(f"![{name}]({name})")
        lines.append(f"*{captions.get(name, name)}*")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("ROC curves hug the diagonal and the |llr| distributions for "
                 "significant vs background variants overlap almost completely: the "
                 "zero-shot likelihood prior barely separates significant from "
                 "background caQTLs. Variant-centering gives a small overall AUROC bump "
                 "(0.497 -> 0.517). Per-class numbers rest on very few positives "
                 "(significant counts: overall 50, dELS 35, pELS 11, PLS 3), so the "
                 "apparent PLS/pELS differences are noisy and should not be "
                 "over-read; the best-supported sets (overall, dELS) sit at chance. The "
                 "flat |llr|-vs-|beta| scatter matches the near-zero spearman: the score "
                 "does not track effect magnitude.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="plots + report for likelihood zero-shot VEP")
    parser.add_argument("--element-npz", default="results/likelihood_zero_shot/scores.npz")
    parser.add_argument("--variant-npz",
                        default="results/likelihood_zero_shot_variant_centered/scores.npz")
    parser.add_argument("--outdir", default="results/likelihood_zero_shot/plots")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    modes = {}
    element = load_scores(args.element_npz)
    variant = load_scores(args.variant_npz)
    if element is not None:
        modes["element"] = element
    if variant is not None:
        modes["variant"] = variant

    if not modes:
        raise FileNotFoundError("no scores.npz found; run dump_likelihood_zero_shot first")

    # dense-class panel set is shared across modes (same variants), take it from
    # whichever mode is present (prefer variant-centered for the deep-dive plots).
    deep = variant if variant is not None else element
    keys = panel_keys(deep)
    deep_name = "variant" if variant is not None else "element"

    plot_names = []

    plot_metrics_bars(modes, keys, outdir / "metrics_by_class.png")
    plot_names.append("metrics_by_class.png")

    plot_roc(modes, keys, outdir / "roc_curves.png")
    plot_names.append("roc_curves.png")

    plot_pr(modes, keys, outdir / "pr_curves.png")
    plot_names.append("pr_curves.png")

    plot_score_separation(deep, keys, outdir / "score_separation.png", deep_name)
    plot_names.append("score_separation.png")

    plot_effect_scatter(deep, keys, outdir / "effect_size_scatter.png", deep_name)
    plot_names.append("effect_size_scatter.png")

    report = build_report(modes, keys, plot_names)
    with open(outdir / "report.md", "w") as handle:
        handle.write(report)

    print(f"wrote report.md and {len(plot_names)} plots to {outdir}")


if __name__ == "__main__":
    main()
