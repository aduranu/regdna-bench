"""overlay significant-variant positions on DNase accessibility, per CRE class.

for the best- and worst-performing CRE classes (and dELS for context, the class
with the most significant variants), plots where significant vs background caQTLs
sit relative to the host cCRE center, overlaid with the mean GM12878 DNase
accessibility meta-profile (ENCSR000EMT, the track behind the dataset's chrombpnet
column). answers whether significant variants concentrate at the accessibility
summit and whether that differs between strong/weak classes.

run in an env with pyBigWig (e.g. tangermeme):
    ~/miniforge3/envs/tangermeme/bin/python examples/d3/plot_accessibility_overlay.py \
        --table results/embedding_zero_shot/variant_positions_genomewide.csv \
        --bigwig ~/scratch/caqtl_accessibility/ENCFF915DFR.bigWig \
        --outdir results/embedding_zero_shot
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig

SIG_COLOR = "#C44E52"
BG_COLOR = "#4C72B0"
ACC_COLOR = "#55A868"

# accessibility meta-profile half-window (bp) for context; the model only scores
# the central +/-175, marked on each panel.
ACC_HALF = 500
VAR_HALF = 175

# auroc per class from the embedding zero-shot eval (held-out chr22), for labeling
CLASS_AUROC = {"PLS": 0.621, "pELS": 0.423, "dELS": 0.485}

# best, worst, plus the highest-signal class for a readable distribution
PANEL_CLASSES = ["PLS", "pELS", "dELS"]

# cap unique cCREs read per class so the meta-profile stays fast; sampled evenly
MAX_CCRE = 5000


def meta_profile(bw, chroms, mids):
    # mean DNase signal over [mid-ACC_HALF, mid+ACC_HALF) across the class's cCREs.
    chrom_set = set(bw.chroms().keys())
    acc = np.zeros(2 * ACC_HALF, dtype=np.float64)
    used = 0

    for chrom, mid in zip(chroms, mids):
        if chrom not in chrom_set:
            continue

        lo = mid - ACC_HALF
        hi = mid + ACC_HALF
        if lo < 0 or hi > bw.chroms()[chrom]:
            continue

        vals = np.array(bw.values(chrom, lo, hi), dtype=np.float64)
        acc += np.nan_to_num(vals, nan=0.0)
        used += 1

    if used:
        acc /= used

    return acc, used


def unique_ccres(sub):
    # one row per host cCRE (variants share cCREs), capped for speed
    uniq = sub.drop_duplicates(subset=["chrom", "ccre_mid"])

    if len(uniq) > MAX_CCRE:
        step = len(uniq) // MAX_CCRE
        uniq = uniq.iloc[::step]

    return uniq


def panel(ax, df, cls, bw):
    sub = df[df["cre_class"] == cls]
    sig = sub[sub["label"] == 1]["offset_from_center"].to_numpy()
    bg = sub[sub["label"] == 0]["offset_from_center"].to_numpy()

    uniq = unique_ccres(sub)
    acc, used = meta_profile(bw, uniq["chrom"].to_numpy(), uniq["ccre_mid"].to_numpy())
    acc_x = np.arange(-ACC_HALF, ACC_HALF)

    # left axis: normalized variant-position densities within the scored window
    bins = np.linspace(-VAR_HALF, VAR_HALF, 36)
    ax.hist(bg, bins=bins, density=True, alpha=0.45, color=BG_COLOR,
            label=f"background (n={len(bg)})")
    ax.hist(sig, bins=bins, density=True, alpha=0.6, color=SIG_COLOR,
            label=f"significant (n={len(sig)})")
    ax.set_xlabel("position relative to cCRE center (bp)")
    ax.set_ylabel("variant density")
    ax.set_xlim(-ACC_HALF, ACC_HALF)

    # right axis: mean DNase accessibility meta-profile
    ax2 = ax.twinx()
    ax2.plot(acc_x, acc, color=ACC_COLOR, lw=2, label=f"DNase mean ({used} cCREs)")
    ax2.set_ylabel("mean DNase signal (read-depth norm.)", color=ACC_COLOR)
    ax2.tick_params(axis="y", labelcolor=ACC_COLOR)
    ax2.set_ylim(bottom=0)

    # mark the scored window edges
    for edge in (-VAR_HALF, VAR_HALF):
        ax.axvline(edge, color="grey", ls=":", lw=1)

    auroc = CLASS_AUROC.get(cls)
    title = f"{cls}" + (f"  (eval AUROC={auroc:.3f})" if auroc is not None else "")
    ax.set_title(title)

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper right")


def write_stats(df, outpath):
    lines = []
    lines.append("# Significant-variant position vs DNase accessibility (by CRE class)")
    lines.append("")
    lines.append("Positions are genome-wide (all autosomes + chrX) so the distributions "
                 "are well-sampled; the best/worst ranking is from the held-out chr22 eval. "
                 "Offset = variant position minus host cCRE center (the model's 350 bp "
                 "window anchor). Accessibility = GM12878 DNase (ENCSR000EMT).")
    lines.append("")
    lines.append("| Class | eval AUROC | n significant | n background | median abs-offset (sig) | median abs-offset (bg) | % sig within +/-50bp | % bg within +/-50bp |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for cls in PANEL_CLASSES:
        sub = df[df["cre_class"] == cls]
        sig = sub[sub["label"] == 1]["offset_from_center"].to_numpy()
        bg = sub[sub["label"] == 0]["offset_from_center"].to_numpy()
        auroc = CLASS_AUROC.get(cls)
        au = f"{auroc:.3f}" if auroc is not None else "n/a"
        sig_core = 100 * np.mean(np.abs(sig) <= 50) if len(sig) else float("nan")
        bg_core = 100 * np.mean(np.abs(bg) <= 50) if len(bg) else float("nan")
        lines.append(f"| {cls} | {au} | {len(sig)} | {len(bg)} | "
                     f"{np.median(np.abs(sig)):.1f} | {np.median(np.abs(bg)):.1f} | "
                     f"{sig_core:.1f}% | {bg_core:.1f}% |")

    lines.append("")
    with open(outpath, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="accessibility overlay per CRE class")
    parser.add_argument("--table", default="results/embedding_zero_shot/variant_positions_genomewide.csv")
    parser.add_argument("--bigwig", required=True)
    parser.add_argument("--outdir", default="results/embedding_zero_shot")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table)
    bw = pyBigWig.open(args.bigwig)

    fig, axes = plt.subplots(1, len(PANEL_CLASSES), figsize=(7 * len(PANEL_CLASSES), 5.5))
    for ax, cls in zip(axes, PANEL_CLASSES):
        panel(ax, df, cls, bw)

    fig.suptitle("Significant caQTL positions vs DNase accessibility, by CRE class "
                 "(best -> worst eval AUROC: PLS, dELS, pELS)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "accessibility_overlay.png", dpi=150)
    plt.close(fig)

    bw.close()

    write_stats(df, outdir / "accessibility_stats.md")

    print(f"wrote accessibility_overlay.png and accessibility_stats.md to {outdir}")


if __name__ == "__main__":
    main()
