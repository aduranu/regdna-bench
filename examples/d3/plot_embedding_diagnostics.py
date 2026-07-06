"""low-dim embedding + classification plots for the zero-shot embedding VEP task.

reads embeddings.npz (ref/alt mean-pooled embeddings, cosine distances, labels,
betas, CRE classes) and emits two figures:

  embeddings_lowdim.png      PCA + t-SNE of the ref embeddings, colored by CRE
                             class / label / cosine distance, plus the cosine
                             distance distributions that feed every metric.
  classification_diag.png    ROC and PR curves (overall + per class), the
                             sig-vs-background score separation, and the
                             cosine-distance vs |beta| scatter behind spearman.

usage:
    python examples/d3/plot_embedding_diagnostics.py \
        --npz results/embedding_zero_shot/embeddings.npz \
        --outdir results/embedding_zero_shot
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

SIG_COLOR = "#C44E52"
BG_COLOR = "#4C72B0"
MIN_PER_CLASS = 10


def load(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    return {
        "ref_emb": data["ref_emb"],
        "alt_emb": data["alt_emb"],
        "distance": data["cosine_distance"],
        "labels": data["labels"],
        "betas": data["betas"],
        "classes": data["cre_classes"].astype(str),
    }


def pca_2d(emb):
    scaled = StandardScaler().fit_transform(emb)

    return PCA(n_components=2, random_state=0).fit_transform(scaled)


def tsne_2d(emb):
    # pca-reduce first for a stable, faster t-SNE on the high-dim embeddings
    scaled = StandardScaler().fit_transform(emb)
    pre = PCA(n_components=min(50, emb.shape[1]), random_state=0).fit_transform(scaled)
    perplexity = min(30, max(5, (len(emb) - 1) // 3))

    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=0).fit_transform(pre)


def scatter_by_category(ax, xy, categories, title):
    for cat in sorted(set(categories)):
        m = categories == cat
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, label=f"{cat} (n={int(m.sum())})")

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(fontsize=7, markerscale=1.5)


def scatter_by_label(ax, xy, labels, title):
    bg = labels == 0
    ax.scatter(xy[bg, 0], xy[bg, 1], s=10, alpha=0.5, color=BG_COLOR,
               label=f"background (n={int(bg.sum())})")
    ax.scatter(xy[~bg, 0], xy[~bg, 1], s=14, alpha=0.7, color=SIG_COLOR,
               label=f"significant (n={int((~bg).sum())})")

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(fontsize=8)


def scatter_by_distance(ax, xy, distance, title):
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.7, c=distance, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="cosine distance (ref vs alt)")

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")


def dist_by_label(ax, distance, labels):
    bg = distance[labels == 0]
    sig = distance[labels == 1]

    bins = np.linspace(float(distance.min()), float(distance.max()), 40)
    ax.hist(bg, bins=bins, density=True, alpha=0.6, color=BG_COLOR, label="background")
    ax.hist(sig, bins=bins, density=True, alpha=0.6, color=SIG_COLOR, label="significant")

    ax.set_xlabel("cosine distance (ref vs alt)")
    ax.set_ylabel("density")
    ax.set_title("score distribution by label")
    ax.legend()


def dist_by_class(ax, distance, classes):
    cats = sorted(set(classes))
    data = [distance[classes == c] for c in cats]

    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cosine distance")
    ax.set_title("cosine distance by CRE class")


def plot_lowdim(data, outpath):
    pca = pca_2d(data["ref_emb"])
    tsne = tsne_2d(data["ref_emb"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    scatter_by_category(axes[0, 0], pca, data["classes"], "PCA of ref embeddings by CRE class")
    scatter_by_label(axes[0, 1], pca, data["labels"], "PCA of ref embeddings by label")
    scatter_by_distance(axes[0, 2], pca, data["distance"], "PCA colored by cosine distance")

    scatter_by_category(axes[1, 0], tsne, data["classes"], "t-SNE of ref embeddings by CRE class")
    dist_by_label(axes[1, 1], data["distance"], data["labels"])
    dist_by_class(axes[1, 2], data["distance"], data["classes"])

    fig.suptitle("Embedding space and cosine-distance score (zero-shot VEP)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def scored_class_masks(classes, labels):
    # classes with enough variants and both labels present, for per-class curves
    masks = {}
    for c in sorted(set(classes)):
        m = classes == c
        if m.sum() >= MIN_PER_CLASS and len(np.unique(labels[m])) == 2:
            masks[c] = m

    return masks


def plot_roc(ax, distance, labels, classes):
    fpr, tpr, _ = roc_curve(labels, distance)
    auroc = roc_auc_score(labels, distance)
    ax.plot(fpr, tpr, color="black", lw=2, label=f"overall (AUROC={auroc:.3f})")

    for c, m in scored_class_masks(classes, labels).items():
        fpr_c, tpr_c, _ = roc_curve(labels[m], distance[m])
        au_c = roc_auc_score(labels[m], distance[m])
        ax.plot(fpr_c, tpr_c, lw=1.3, alpha=0.8, label=f"{c} (AUROC={au_c:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC: significant vs background")
    ax.legend(fontsize=7, loc="lower right")


def plot_pr(ax, distance, labels, classes):
    prec, rec, _ = precision_recall_curve(labels, distance)
    auprc = average_precision_score(labels, distance)
    base = labels.mean()
    ax.plot(rec, prec, color="black", lw=2, label=f"overall (AUPRC={auprc:.3f})")
    ax.axhline(base, ls="--", color="grey", lw=1, label=f"baseline ({base:.3f})")

    for c, m in scored_class_masks(classes, labels).items():
        prec_c, rec_c, _ = precision_recall_curve(labels[m], distance[m])
        ap_c = average_precision_score(labels[m], distance[m])
        ax.plot(rec_c, prec_c, lw=1.3, alpha=0.8, label=f"{c} (AUPRC={ap_c:.3f})")

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("precision-recall")
    ax.legend(fontsize=7, loc="upper right")


def plot_score_box(ax, distance, labels):
    bg = distance[labels == 0]
    sig = distance[labels == 1]

    ax.boxplot([bg, sig], labels=["background", "significant"], showfliers=False)
    ax.scatter(np.random.normal(1, 0.04, len(bg)), bg, s=5, alpha=0.2, color=BG_COLOR)
    ax.scatter(np.random.normal(2, 0.04, len(sig)), sig, s=8, alpha=0.4, color=SIG_COLOR)

    ax.set_ylabel("cosine distance")
    ax.set_title("score separation (AUROC integrates this)")


def plot_spearman(ax, distance, betas):
    abs_beta = np.abs(betas)
    rho, p = spearmanr(distance, abs_beta)

    ax.scatter(abs_beta, distance, s=10, alpha=0.4, color="#55A868")
    ax.set_xlabel("|beta| (effect size)")
    ax.set_ylabel("cosine distance")
    ax.set_title(f"effect-size correlation (spearman={rho:.3f}, p={p:.2g})")


def plot_classification(data, outpath):
    distance = data["distance"]
    labels = data["labels"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    plot_roc(axes[0, 0], distance, labels, data["classes"])
    plot_pr(axes[0, 1], distance, labels, data["classes"])
    plot_score_box(axes[1, 0], distance, labels)
    plot_spearman(axes[1, 1], distance, data["betas"])

    fig.suptitle("Where the metrics come from: classification + correlation diagnostics",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="embedding + classification diagnostics")
    parser.add_argument("--npz", default="results/embedding_zero_shot/embeddings.npz")
    parser.add_argument("--outdir", default="results/embedding_zero_shot")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load(args.npz)

    plot_lowdim(data, outdir / "embeddings_lowdim.png")
    plot_classification(data, outdir / "classification_diag.png")

    print(f"wrote embeddings_lowdim.png and classification_diag.png to {outdir}")


if __name__ == "__main__":
    main()
