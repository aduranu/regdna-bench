"""exploratory plots for the African caQTL dataset that feeds the zero-shot task.

shows what the raw tsv contains and how the zero-shot eval subset is carved out
of it (IsUsed & in_peaks & held-out test chroms). one multi-panel figure plus a
small stats markdown.

usage:
    python examples/d3/plot_dataset_eda.py \
        --variants data/caqtl/Afr.CaQTLS.tsv --outdir results/embedding_zero_shot
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIG_COLOR = "#C44E52"
BG_COLOR = "#4C72B0"
TEST_CHROMS = ["chr22", "chrX"]


def norm_chrom(chrom):
    chrom = str(chrom)

    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def eval_mask(df):
    # the zero-shot subset before cCRE containment: QTL-tested, in an accessible
    # peak, on a held-out test chrom.
    mask = pd.Series(True, index=df.index)

    if "IsUsed" in df.columns:
        mask &= df["IsUsed"].astype(bool)
    if "in_peaks" in df.columns:
        mask &= df["in_peaks"].astype(bool)

    mask &= df["chr_hg38"].map(norm_chrom).isin(TEST_CHROMS)

    return mask


def panel_label_balance(ax, df):
    counts = df["label"].astype(bool).value_counts()
    n_bg = int(counts.get(False, 0))
    n_sig = int(counts.get(True, 0))

    bars = ax.bar(["background", "significant"], [n_bg, n_sig], color=[BG_COLOR, SIG_COLOR])
    total = n_bg + n_sig

    for bar, val in zip(bars, [n_bg, n_sig]):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val}\n({100 * val / total:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("variants")
    ax.set_title("label balance (full dataset)")
    ax.set_ylim(0, total * 1.15)


def panel_beta_by_label(ax, df):
    sig = df.loc[df["label"].astype(bool), "beta"].to_numpy()
    bg = df.loc[~df["label"].astype(bool), "beta"].to_numpy()

    bins = np.linspace(-1.5, 1.5, 61)
    ax.hist(bg, bins=bins, density=True, alpha=0.6, color=BG_COLOR, label="background")
    ax.hist(sig, bins=bins, density=True, alpha=0.6, color=SIG_COLOR, label="significant")

    ax.set_xlabel("beta (effect size)")
    ax.set_ylabel("density")
    ax.set_title("effect size by label")
    ax.legend()


def panel_abs_beta(ax, df):
    sig = df.loc[df["label"].astype(bool), "beta"].abs().to_numpy()
    bg = df.loc[~df["label"].astype(bool), "beta"].abs().to_numpy()

    ax.violinplot([bg, sig], showmedians=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["background", "significant"])
    ax.set_ylabel("|beta|")
    ax.set_title("absolute effect size by label")


def panel_pval(ax, df):
    # -log10 p; significant variants pile up at large values by construction
    neglog = -np.log10(np.clip(df["pval"].to_numpy(), 1e-50, 1.0))
    sig_mask = df["label"].astype(bool).to_numpy()

    bins = np.linspace(0, np.percentile(neglog, 99.5), 60)
    ax.hist(neglog[~sig_mask], bins=bins, density=True, alpha=0.6, color=BG_COLOR, label="background")
    ax.hist(neglog[sig_mask], bins=bins, density=True, alpha=0.6, color=SIG_COLOR, label="significant")

    ax.set_xlabel("-log10(pval)")
    ax.set_ylabel("density")
    ax.set_title("QTL significance")
    ax.legend()


def panel_distance(ax, df):
    # distance to peak summit; -1 is the missing sentinel, dropped here
    dist = df.loc[df["distance"] >= 0, "distance"].to_numpy()

    ax.hist(dist, bins=60, color="#55A868")
    ax.set_xlabel("distance to peak summit (bp)")
    ax.set_ylabel("variants")
    ax.set_title("variant-to-peak distance")


def panel_chrom(ax, df):
    order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    chrom = df["chr_hg38"].map(norm_chrom)
    counts = chrom.value_counts()
    present = [c for c in order if c in counts.index]
    vals = [counts[c] for c in present]

    colors = [SIG_COLOR if c in TEST_CHROMS else "#999999" for c in present]
    ax.bar(present, vals, color=colors)

    ax.set_ylabel("variants")
    ax.set_title("per-chromosome counts (red = held-out test)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)


def panel_filters(ax, df):
    n_total = len(df)
    n_used = int(df["IsUsed"].astype(bool).sum()) if "IsUsed" in df.columns else n_total
    n_peaks = int(df["in_peaks"].astype(bool).sum()) if "in_peaks" in df.columns else n_total
    n_eval = int(eval_mask(df).sum())

    labels = ["all", "IsUsed", "in_peaks", "eval subset\n(test chroms)"]
    vals = [n_total, n_used, n_peaks, n_eval]

    bars = ax.bar(labels, vals, color=["#999999", "#999999", "#999999", SIG_COLOR])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val}", ha="center", va="bottom", fontsize=8)

    ax.set_yscale("log")
    ax.set_ylabel("variants (log)")
    ax.set_title("filter funnel to eval subset")
    ax.tick_params(axis="x", labelsize=8)


def write_stats(df, outpath):
    sig = df["label"].astype(bool)
    eval_mask = eval_mask(df)
    eval_df = df[eval_mask]

    lines = []
    lines.append("# African caQTL dataset summary")
    lines.append("")
    lines.append(f"- Total variants: {len(df)}")
    lines.append(f"- Significant (label=True): {int(sig.sum())} ({100 * sig.mean():.2f}%)")
    lines.append(f"- Background (label=False): {int((~sig).sum())} ({100 * (~sig).mean():.2f}%)")
    lines.append(f"- Chromosomes present: {df['chr_hg38'].map(norm_chrom).nunique()} "
                 f"(no chrX/chrY in this file)")
    lines.append("")
    lines.append("## Effect size (beta)")
    lines.append(f"- range [{df['beta'].min():.3f}, {df['beta'].max():.3f}], "
                 f"mean {df['beta'].mean():.4f}, std {df['beta'].std():.4f}")
    lines.append(f"- median |beta|: significant {df.loc[sig, 'beta'].abs().median():.4f} "
                 f"vs background {df.loc[~sig, 'beta'].abs().median():.4f}")
    lines.append("")
    lines.append("## Eval subset (IsUsed & in_peaks & test chroms)")
    lines.append(f"- Variants: {len(eval_df)}")
    if len(eval_df):
        lines.append(f"- Significant: {int(eval_df['label'].astype(bool).sum())} "
                     f"({100 * eval_df['label'].astype(bool).mean():.2f}%)")
        lines.append(f"- Chroms: {sorted(eval_df['chr_hg38'].map(norm_chrom).unique())}")
    lines.append("")
    lines.append("Note: the runner further drops variants not contained in a cCRE or "
                 "falling outside the central 350 bp window, yielding the n reported in "
                 "the scoring results.")
    lines.append("")

    with open(outpath, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="EDA plots for the caQTL dataset")
    parser.add_argument("--variants", default="data/caqtl/Afr.CaQTLS.tsv")
    parser.add_argument("--outdir", default="results/embedding_zero_shot")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.variants, sep="\t")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    panel_label_balance(axes[0, 0], df)
    panel_beta_by_label(axes[0, 1], df)
    panel_abs_beta(axes[0, 2], df)
    panel_pval(axes[0, 3], df)
    panel_distance(axes[1, 0], df)
    panel_chrom(axes[1, 1], df)
    panel_filters(axes[1, 2], df)
    axes[1, 3].axis("off")

    fig.suptitle("African caQTL dataset (input to zero-shot embedding VEP)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "dataset_eda.png", dpi=150)
    plt.close(fig)

    write_stats(df, outdir / "dataset_stats.md")

    print(f"wrote dataset_eda.png and dataset_stats.md to {outdir}")


if __name__ == "__main__":
    main()
