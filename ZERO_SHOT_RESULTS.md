# Zero-shot variant-effect experiments — results summary

DART-Eval **task 5** (zero-shot variant-effect prediction) run on the branch
`zero-shot`. Four experiments were run: two scoring methods (**embedding** and
**likelihood**) crossed with two windowing modes (**element-centered** and
**variant-centered**). All results are unsupervised (no training on the caQTL
labels) — the model is used only as a zero-shot prior.

Bottom line: **no configuration separates significant from background caQTLs on
this set.** Overall AUROC stays at chance (0.496–0.517) in every run.

---

## Model

- **Model:** zoonomia D3 — DNA discrete diffusion (uniform-SEDD), hg38.
- **Checkpoint:** `model-epoch=207-val_loss=395.0214.ckpt` (zoonomia `best/`).
- **Genome backend:** zarr (`zoonomia_v3.zarr`), cCREs from `GRCh38-cCREs.bed`.

## Dataset

African caQTL set (`afr` caQTL TSV, hg38; columns `chr_hg38`/`pos_hg38`/`allele1`/`allele2`/`label`/`beta`).

| Property | Value |
|---|---|
| Total variants | 219,382 |
| Significant (`label=True`) | 11,098 (5.06%) |
| Background (`label=False`) | 208,284 (94.94%) |
| Chromosomes | 22 autosomes (no chrX/chrY in file) |
| Effect size `beta` | range [-1.395, 2.081], mean -0.0034, std 0.2621 |
| median \|beta\| | significant 0.6706 vs background 0.1130 |

**Eval funnel (held-out test):** `IsUsed & in_peaks & test chrom (chr22)` -> 1,103
variants (57 significant, 5.17%). The runner then drops variants not contained
in a cCRE (`no_ccre`: 177) or falling outside the central 350 bp window,
yielding **n = 926 scored variants** in every experiment.

## CRE classes and windows

- **cCRE classes** (from the cCRE BED, host element of each variant): `PLS`,
  `pELS`, `dELS`, `CA-CTCF`, `CA-H3K4me3`.
- Classes with fewer than 10 eval variants are **skipped**: `CA-CTCF` (n=7),
  `CA-H3K4me3` (n=2). Reported classes: `dELS` (402), `PLS` (306), `pELS` (209).
- **Window:** 350 bp in all runs. Two anchoring modes:
  - **element-centered** — window is the host cCRE midpoint ±175 bp.
  - **variant-centered** — window is centered on the variant itself.
  - N-padding only at chromosome edges (never used to skip real flanking genome).

## Scores and metrics

- **Embedding score:** cosine distance between mean-pooled last-layer ref/alt
  embeddings. Larger distance = predicted more significant.
- **Likelihood score:** signed allele log-likelihood difference from per-position
  logits, `llr = score[off,alt] - score[off,ref]`; the magnitude `|llr|` is the
  significance score.
- **Classification** (significant vs background): **AUROC**, **AUPRC**.
- **Correlation** of score vs `|beta|`: **Pearson**, **Spearman**.

---

## Overall results (n = 926, 4 experiments)

| Method | Window | AUROC | AUPRC | Pearson | Spearman |
|---|---|---|---|---|---|
| embedding | element | 0.4983 | 0.0566 | 0.0005 | -0.0078 |
| embedding | variant | 0.4979 | 0.0558 | 0.0144 | 0.0184 |
| likelihood | element | 0.4965 | 0.0708 | 0.0437 | 0.0103 |
| likelihood | variant | **0.5165** | 0.0762 | 0.0350 | 0.0075 |

Chance AUROC = 0.5; positive rate ~0.057 sets the AUPRC baseline. Every run is
within noise of chance. Variant-centering gives the likelihood method a small
bump (0.497 -> 0.517) and negligible change for embeddings.

## Per-class results

### Embedding — element-centered
| Class | n | AUROC | AUPRC | Pearson | Spearman |
|---|---|---|---|---|---|
| dELS | 402 | 0.4854 | 0.0955 | 0.0054 | -0.0225 |
| PLS | 306 | 0.6210 | 0.0209 | 0.0863 | 0.0217 |
| pELS | 209 | 0.4229 | 0.0529 | -0.0714 | -0.0643 |

### Embedding — variant-centered
| Class | n | AUROC | AUPRC | Pearson | Spearman |
|---|---|---|---|---|---|
| dELS | 402 | 0.4321 | 0.0803 | -0.0109 | -0.0302 |
| PLS | 306 | 0.8069 | 0.0334 | 0.0918 | 0.0379 |
| pELS | 209 | 0.4993 | 0.0633 | -0.0100 | 0.0211 |

### Likelihood — element-centered
| Class | n | AUROC | AUPRC | Pearson | Spearman |
|---|---|---|---|---|---|
| dELS | 402 | 0.4747 | 0.0934 | 0.0232 | 0.0026 |
| PLS | 306 | 0.5259 | 0.0139 | 0.1451 | 0.1045 |
| pELS | 209 | 0.5110 | 0.1491 | -0.0239 | -0.1184 |

### Likelihood — variant-centered
| Class | n | AUROC | AUPRC | Pearson | Spearman |
|---|---|---|---|---|---|
| dELS | 402 | 0.4844 | 0.0995 | -0.0084 | -0.0521 |
| PLS | 306 | 0.5765 | 0.0267 | 0.1630 | 0.1094 |
| pELS | 209 | 0.5583 | 0.1145 | 0.0101 | -0.0322 |

**Caveat on per-class numbers.** They rest on very few positives (significant
counts: overall ~50, dELS 35, pELS 11, PLS 3). The eye-catching PLS AUROC
(0.62 -> 0.81 embedding, 0.53 -> 0.58 likelihood under variant-centering) is
driven by 3 positive variants and should not be over-read. The best-supported
sets (overall, dELS) sit at chance.

## Position / accessibility diagnostic

Offset = significant-variant position minus host cCRE center (the model's window
anchor); accessibility = GM12878 DNase (ENCSR000EMT), genome-wide positions.

| Class | eval AUROC | median \|offset\| sig | median \|offset\| bg | % sig within ±50bp | % bg within ±50bp |
|---|---|---|---|---|---|
| PLS | 0.621 | 59.0 | 69.0 | 43.5% | 36.2% |
| pELS | 0.423 | 46.0 | 63.0 | 53.4% | 40.0% |
| dELS | 0.485 | 48.0 | 67.0 | 52.0% | 37.1% |

Significant variants sit slightly closer to the cCRE center than background, but
not enough to explain the per-class AUROC ranking.

## Interpretation

The zero-shot zoonomia D3 prior — whether read as embedding cosine distance or as
allele log-likelihood ratio — **does not separate significant caQTLs from
background SNPs** in accessible peaks on this set. ROC curves hug the diagonal,
`|llr|` distributions for significant vs background overlap almost completely,
and score-vs-`|beta|` correlations are near zero (the score does not track
effect magnitude). Variant-centering slightly helps the likelihood method
overall. Any per-class signal (notably PLS) is on too few positives to trust.

---

## Reproducing / artifacts

Result artifacts live under `results/` (gitignored):

- `results/embedding_zero_shot/` — element-centered embedding + full plot set
  and `report.md`, `dataset_stats.md`, `accessibility_stats.md`.
- `results/embedding_zero_shot_variant_centered/`
- `results/likelihood_zero_shot/` — element-centered likelihood + `plots/report.md`.
- `results/likelihood_zero_shot_variant_centered/`

Run scripts: `examples/d3/run_{embedding,likelihood}_zero_shot{,_variant_centered}.sbatch`
(dataset paths/columns in `examples/d3/config.py`).
