"""task 5: variant effect prediction (zero-shot).

scores how a variant changes regulatory activity from paired ref/alt allele
windows (caQTL / dsQTL data). two zero-shot methods, matching DART-Eval:

- embedding: cosine distance between mean-pooled ref/alt last-layer embeddings
  (needs only the EmbeddingModel capability).
- likelihood: the allele log-likelihood difference from the model's per-position
  scores (needs the LikelihoodModel capability).

a good model puts a larger score on significant variants (real QTLs) than on
background variants. results are reported overall and stratified by host CRE
class. headline metric is auroc/auprc; spearman vs |effect size| is secondary.
the model is scored once over all variants, then metrics are stratified by
subsetting, so per-class numbers match a per-class re-score exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from regdna_bench.base import EmbeddingModel, LikelihoodModel
from regdna_bench.metrics import classification_metrics, correlation_metrics
from regdna_bench.registry import register_task
from regdna_bench.tasks._scoring import (
    extract_allele_score_difference,
    extract_embeddings,
)
from regdna_bench.tasks.base import Task, group_indices, subset


def variant_metrics(score, effect_sizes, labels):
    # the score is unsigned: it correlates with |effect size| and doubles as the
    # significance score for the sig-vs-background classification.
    entry = {}

    if effect_sizes is not None:
        target = np.abs(np.asarray(effect_sizes, dtype=np.float64))
        entry["correlation"] = correlation_metrics(score, target)

    if labels is not None:
        entry["classification"] = classification_metrics(score, labels)

    return entry


def stratified_metrics(score, groups, effect_sizes, labels, min_per_class):
    # overall entry plus one per CRE class; sparse classes are skipped (not
    # reported noisy). auroc is undefined on a single label, so classes with one
    # label drop labels and keep only the correlation metric.
    results = {"overall": {"n": len(score),
                           "metrics": variant_metrics(score, effect_sizes, labels)}}

    for cls, idx in group_indices(groups).items():
        n = len(idx)

        if n < min_per_class:
            results[cls] = {"n": n, "skipped": f"fewer than {min_per_class} variants"}
            continue

        cls_labels = subset(labels, idx)
        if cls_labels is not None and len(np.unique(np.asarray(cls_labels))) < 2:
            cls_labels = None

        results[cls] = {"n": n, "metrics": variant_metrics(
            subset(score, idx), subset(effect_sizes, idx), cls_labels)}

    return results


@register_task
class VEPEmbeddingTask(Task):
    """cosine distance between mean-pooled ref/alt embeddings."""

    name = "vep-embedding"
    requires = (EmbeddingModel,)

    def __init__(self, batch_size=256, min_per_class=10):
        self.batch_size = batch_size
        self.min_per_class = min_per_class

    def prepare(self, dataset):
        return dataset.load()

    def run(self, model, variant_set):
        ref_emb = extract_embeddings(model, variant_set.ref_ids, self.batch_size)
        alt_emb = extract_embeddings(model, variant_set.alt_ids, self.batch_size)

        cos = torch.nn.functional.cosine_similarity(ref_emb, alt_emb, dim=1)
        distance = (1.0 - cos).numpy()

        return {"score": distance, "cosine_distance": distance}

    def metrics(self, variant_set, output):
        return stratified_metrics(output["score"], variant_set.groups,
                                  variant_set.effect_sizes, variant_set.labels,
                                  self.min_per_class)


@register_task
class VEPLikelihoodTask(Task):
    """allele log-likelihood difference from per-position scores."""

    name = "vep-likelihood"
    requires = (LikelihoodModel,)

    def __init__(self, batch_size=256, min_per_class=10):
        self.batch_size = batch_size
        self.min_per_class = min_per_class

    def prepare(self, dataset):
        return dataset.load()

    def run(self, model, variant_set):
        # the signed value can carry effect direction; |llr| is the significance
        # score, mirroring the unsigned cosine distance of the embedding method.
        llr = extract_allele_score_difference(
            model, variant_set.ref_ids, variant_set.offsets,
            variant_set.ref_base, variant_set.alt_base, self.batch_size,
        )

        return {"score": np.abs(llr), "llr": llr}

    def metrics(self, variant_set, output):
        return stratified_metrics(output["score"], variant_set.groups,
                                  variant_set.effect_sizes, variant_set.labels,
                                  self.min_per_class)
