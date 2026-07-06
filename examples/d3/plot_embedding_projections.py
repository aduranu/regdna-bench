"""PCA / t-SNE / UMAP projections of the ref and alt (var) allele embeddings.

reads embeddings.npz and lays out a 2x3 grid: rows are the reference-allele and
the variant-allele mean-pooled embeddings, columns are PCA, t-SNE and UMAP, every
panel colored by host CRE class. shows whether the embedding space organizes by
regulatory element type and whether the single-base alt substitution changes that
organization (it barely does, since ref and alt differ by one nucleotide in 350).

run in an env with umap-learn (e.g. autotune_ntv3):
    ~/miniforge3/envs/autotune_ntv3/bin/python examples/d3/plot_embedding_projections.py \
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
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# fixed seed everywhere so ref/var panels are comparable run-to-run
SEED = 0


def pca_2d(emb):
    scaled = StandardScaler().fit_transform(emb)

    return PCA(n_components=2, random_state=SEED).fit_transform(scaled)


def tsne_2d(emb):
    # pca-reduce first for a stable, faster t-SNE on the 768-d embeddings
    scaled = StandardScaler().fit_transform(emb)
    pre = PCA(n_components=min(50, emb.shape[1]), random_state=SEED).fit_transform(scaled)
    perplexity = min(30, max(5, (len(emb) - 1) // 3))

    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=SEED).fit_transform(pre)


def umap_2d(emb):
    scaled = StandardScaler().fit_transform(emb)

    return umap.UMAP(n_components=2, random_state=SEED).fit_transform(scaled)


def scatter_by_class(ax, xy, classes, title):
    for cls in sorted(set(classes)):
        m = classes == cls
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, label=f"{cls} (n={int(m.sum())})")

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")


def main():
    parser = argparse.ArgumentParser(description="PCA/t-SNE/UMAP of ref and var embeddings")
    parser.add_argument("--npz", default="results/embedding_zero_shot/embeddings.npz")
    parser.add_argument("--outdir", default="results/embedding_zero_shot")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    classes = data["cre_classes"].astype(str)
    embeddings = {"ref": data["ref_emb"], "var": data["alt_emb"]}
    methods = {"PCA": pca_2d, "t-SNE": tsne_2d, "UMAP": umap_2d}

    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    for row, (name, emb) in enumerate(embeddings.items()):
        for col, (mname, fn) in enumerate(methods.items()):
            xy = fn(emb)
            scatter_by_class(axes[row, col], xy, classes, f"{mname} - {name} embeddings")

    # one shared legend (classes identical across panels)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9, markerscale=1.6,
               title="CRE class")

    fig.suptitle("Low-dim projections of ref vs var (alt) allele embeddings, by CRE class",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    fig.savefig(outdir / "embedding_projections_by_class.png", dpi=150)
    plt.close(fig)

    print(f"wrote embedding_projections_by_class.png to {outdir}")


if __name__ == "__main__":
    main()
