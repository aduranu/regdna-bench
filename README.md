# regdna-bench

A generalizable benchmark for regulatory-DNA generative models. The core is
**model-agnostic**: tasks are written against capability interfaces, and any
model plugs in by implementing the minimal capability it supports. This document
is the current overview of the repo structure and the most recent experiments
(the `zero-shot` line of work). Full result tables live in
[`ZERO_SHOT_RESULTS.md`](ZERO_SHOT_RESULTS.md).

---

## Architecture at a glance

The only model-specific code is a thin **adapter** that lives outside the
installed package (in `examples/`). Everything the benchmark ships is agnostic to
the model, and tasks never import a concrete model.

```
model adapter (example)  ->  capability interfaces  ->  Task  ->  metrics
   examples/d3/wrapper.py     regdna_bench.base        tasks/     metrics.py
```

- A **model** implements one or more capability interfaces (`BenchModel.forward`,
  `EmbeddingModel.embed`, `LikelihoodModel.predict_logits` + `vocab_index`).
- A **task** declares the capabilities it `requires`; the **runner** checks them
  and fails fast before doing any work, then runs `prepare -> run -> metrics`.
- All tasks/datasets speak the canonical DNA alphabet `CANON` (`N,A,C,G,T`); each
  model owns the mapping from CANON to its own vocab, so datasets stay
  model-agnostic.

## Repo structure

```
src/regdna_bench/                 installable, model-agnostic core
    base.py                       capability ABCs (BenchModel / EmbeddingModel /
                                  LikelihoodModel) + CANON alphabet
    registry.py                   register_task / get_task / list_tasks
    runner.py                     run_task: capability check -> prepare/run/metrics
    metrics.py                    pure scoring (correlation, classification); no torch
    cli.py                        `regdna-bench` entry point (list-tasks / run)
    data/
        base.py                   Dataset ABC + VariantSet container
        caqtl.py                  CaqtlVariantDataset loader
        ccre.py                   cCRE BED handling / host-class assignment
        genome_store.py           zarr genome backend for window extraction
    tasks/
        base.py                   Task ABC + stratification helpers (group_indices, subset)
        _scoring.py               shared model-touching score extractors
        short_ctx/                short-context task family (~350 bp windows)
            vep.py                task 5: zero-shot variant-effect (embedding + likelihood)
            probing.py            frozen-embedding probing (ridge + CNN heads)

examples/d3/                      D3 (DNA discrete diffusion) example, not installed
    wrapper.py                    D3 model adapter (implements the capability interfaces
                                  + probing hook API)
    config.py                     dataset paths + TSV column names (kept out of the core)
    run.py                        thin runner around the CLI
    run_probing.py                probing entry point
    dump_*_zero_shot.py(.sbatch)  score-dumping jobs (embedding / likelihood)
    build_variant_position_table.py
    plot_*.py                     dataset EDA, embedding projections/diagnostics,
                                  accessibility overlay, embedding/likelihood results
    run_*_zero_shot*.sbatch       the 4 VEP experiment configs
```

Package name `regdna-bench` (v0.1.0). Core deps are light (`numpy`, `scipy`,
`scikit-learn`); `torch`, `pandas`, `zarr`, `h5py` and model backbones (e.g.
`d3_dna`) are environment-provided and imported lazily.

## Task families

### short-context (`tasks/short_ctx/`)

Tasks scored on single short windows (~350 bp). Two subsets:

- **zero-shot variant-effect (`vep.py`)** — DART-Eval task 5. Scores how a variant
  changes regulatory activity from paired ref/alt allele windows. Two zero-shot
  methods:
  - `vep-embedding`: cosine distance between mean-pooled last-layer ref/alt
    embeddings (needs `EmbeddingModel`).
  - `vep-likelihood`: signed allele log-likelihood difference from per-position
    logits, `|llr|` as the significance score (needs `LikelihoodModel`).
- **probing (`probing.py`)** — trains a lightweight probe on frozen last-layer
  embeddings (`ridge`: mean-pool + RidgeCV; `cnn`: RepCNN conv stack). Measures how
  decodable a model's representations are for a downstream activity target.
  Currently function-based (`run_probing` / `run_cnn_probing`); porting it to the
  `Task` ABC is a tracked follow-up.

## Running

```bash
# list registered tasks
regdna-bench list-tasks            # -> vep-embedding, vep-likelihood

# run a task (model supplied as an import spec that returns a BenchModel)
regdna-bench run vep-likelihood \
    --model examples/d3/wrapper.py:D3Model.from_pretrained \
    --variants <caqtl.tsv> --window-mode variant \
    --checkpoint <ckpt> --ccre-bed <bed> --zarr <zarr> \
    --chroms chr22,chrX --out results.json
```

`--window-mode` is `element` (host cCRE midpoint +/-175) or `variant`
(centered on the variant). The D3 example scripts under `examples/d3/` wrap this
for SLURM.

---

## Most recent experiments — zero-shot VEP (DART-Eval task 5)

Four runs: two scoring methods x two windowing modes, model = **zoonomia D3**
(uniform-SEDD DNA discrete diffusion, hg38, epoch 207).

- **Dataset:** African caQTLs (219,382 variants; 5.06% significant). Held-out
  chr22 eval funnels to **n = 926** scored variants after cCRE/window drops.
- **CRE classes:** reported for `PLS` / `pELS` / `dELS`; `CA-CTCF` / `CA-H3K4me3`
  skipped (<10 eval variants).
- **Windows:** 350 bp, element- vs variant-centered.
- **Metrics:** AUROC / AUPRC (significant vs background) + Pearson/Spearman of
  score vs `|beta|`.

| Method | Window | AUROC | AUPRC | Pearson | Spearman |
|---|---|---|---|---|---|
| embedding | element | 0.4983 | 0.0566 | 0.0005 | -0.0078 |
| embedding | variant | 0.4979 | 0.0558 | 0.0144 | 0.0184 |
| likelihood | element | 0.4965 | 0.0708 | 0.0437 | 0.0103 |
| likelihood | variant | 0.5165 | 0.0762 | 0.0350 | 0.0075 |

**Takeaway:** the zero-shot D3 prior does not separate significant from
background caQTLs (overall AUROC at chance in every run). Variant-centering gives
the likelihood method a small bump; per-class signal rests on too few positives
to trust. Per-class tables, dataset funnel, and the accessibility/offset
diagnostic are in [`ZERO_SHOT_RESULTS.md`](ZERO_SHOT_RESULTS.md).

## Status / still to do

- **Port probing to the `Task` ABC** so both short-context subsets share the
  runner/registry/metrics (currently function-based).
- **Additional short-context task families** (activity prediction, sequence
  distribution) not yet designed against the `Task` ABC.
- **Broader eval + baselines** — n=926 with ~50 positives is small; no supervised
  baseline yet to contextualize the chance-level zero-shot numbers.
- **Tests + CI** for the core (registry/runner/metrics/data loaders).
