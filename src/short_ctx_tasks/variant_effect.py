"""task 5: variant effect prediction (zero-shot).

scores how a variant changes regulatory activity from paired reference/alternate
allele sequences (caQTL / dsQTL data). two zero-shot methods, matching DART-Eval:

- embedding-based: cosine distance between mean-pooled ref/alt last-layer
  embeddings (uses only the BenchModel embedding hook).
- likelihood-based: the allele log-likelihood difference read from the model's
  per-position scores (requires the model to expose predict_logits).

a good model puts a larger score on significant variants (real QTLs) than on
background variants (SNPs in accessible peaks that do not affect accessibility).
results are reported overall and stratified by host CRE class. headline metric is
auroc/auprc (significant vs background); spearman vs |effect size| is secondary.
"""

from __future__ import annotations

import numpy as np
import torch

from short_ctx_tasks.scoring import (
    classification_metrics,
    correlation_metrics,
    extract_allele_score_difference,
    extract_embeddings,
)


def _variant_metrics(distance, effect_sizes, labels):
    # cosine distance is unsigned: it correlates with |effect size| and doubles
    # as the significance score for the sig-vs-background classification.
    entry = {}

    if effect_sizes is not None:
        target = np.abs(np.asarray(effect_sizes, dtype=np.float64))
        entry["correlation"] = correlation_metrics(distance, target)

    if labels is not None:
        entry["classification"] = classification_metrics(distance, labels)

    return entry


def run_variant_zero_shot_embedding(model, ref_seqs, alt_seqs,
                                    effect_sizes=None, labels=None, batch_size=256,
                                    verbose=True):
    # cosine distance between mean-pooled ref/alt embeddings.
    ref_emb = extract_embeddings(model, ref_seqs, batch_size=batch_size)
    alt_emb = extract_embeddings(model, alt_seqs, batch_size=batch_size)

    cos = torch.nn.functional.cosine_similarity(ref_emb, alt_emb, dim=1)
    distance = (1.0 - cos).numpy()

    results = {"cosine_distance": distance}

    if effect_sizes is not None or labels is not None:
        results["metrics"] = _variant_metrics(distance, effect_sizes, labels)

    # verbose is off when called per-class, so the wrapper prints one summary
    if verbose:
        print("variant zero-shot embedding complete")

    return results


def _group_indices_by_class(cre_classes):
    # {class label: [row indices]}, preserving first-seen order so per-class
    # slices stay aligned with the input arrays.
    groups = {}
    for i, cls in enumerate(cre_classes):
        groups.setdefault(cls, []).append(i)

    return groups


def _subset(values, idx):
    # index any of the per-variant containers we pass around: lists/tuples by
    # comprehension, arrays/tensors by fancy index, None passes through.
    if values is None:
        return None

    if isinstance(values, (list, tuple)):
        return [values[i] for i in idx]

    return values[idx]


def run_variant_zero_shot_by_class(model, ref_seqs, alt_seqs, cre_classes,
                                   effect_sizes=None, labels=None, batch_size=256,
                                   min_per_class=10):
    # stratifies the embedding variant score by host CRE class (plus an overall
    # entry). per-class auroc/auprc shows where the generative prior captures
    # functional variant effects; restricting to D3's cCRE classes keeps the
    # eval in-distribution. sparse classes are skipped rather than reported noisy.
    overall = run_variant_zero_shot_embedding(
        model, ref_seqs, alt_seqs, effect_sizes=effect_sizes, labels=labels,
        batch_size=batch_size, verbose=False,
    )
    overall["n"] = len(ref_seqs)

    results = {"overall": overall}
    groups = _group_indices_by_class(cre_classes)

    scored, skipped = 0, 0
    for cls, idx in groups.items():
        n = len(idx)

        if n < min_per_class:
            results[cls] = {"n": n, "skipped": f"fewer than {min_per_class} variants"}
            skipped += 1
            continue

        cls_labels = _subset(labels, idx)
        # auroc is undefined on a single label; drop labels for this class so the
        # scorer still returns the correlation metric without raising.
        if cls_labels is not None and len(np.unique(np.asarray(cls_labels))) < 2:
            cls_labels = None

        cls_result = run_variant_zero_shot_embedding(
            model, _subset(ref_seqs, idx), _subset(alt_seqs, idx),
            effect_sizes=_subset(effect_sizes, idx), labels=cls_labels,
            batch_size=batch_size, verbose=False,
        )
        cls_result["n"] = n
        results[cls] = cls_result
        scored += 1

    print(f"variant zero-shot by class complete: {scored} classes scored, {skipped} skipped")

    return results


def run_variant_zero_shot_likelihood(model, windows, offsets, ref_tokens, alt_tokens,
                                     effect_sizes=None, labels=None, batch_size=256,
                                     verbose=True):
    # allele log-likelihood difference from the model's per-position scores. the
    # signed value can carry effect direction; |llr| is the significance score,
    # mirroring the unsigned cosine distance of the embedding method.
    llr = extract_allele_score_difference(
        model, windows, offsets, ref_tokens, alt_tokens, batch_size=batch_size,
    )
    score = np.abs(llr)

    results = {"score": llr, "abs_score": score}

    if effect_sizes is not None or labels is not None:
        results["metrics"] = _variant_metrics(score, effect_sizes, labels)

    # verbose is off when called per-class, so the wrapper prints one summary
    if verbose:
        print("variant zero-shot likelihood complete")

    return results


def run_variant_zero_shot_likelihood_by_class(model, windows, offsets, ref_tokens,
                                              alt_tokens, cre_classes,
                                              effect_sizes=None, labels=None,
                                              batch_size=256, min_per_class=10):
    # likelihood counterpart of run_variant_zero_shot_by_class: same stratify +
    # skip-sparse + single-label-guard logic, scoring with the allele llr instead
    # of the embedding cosine distance.
    overall = run_variant_zero_shot_likelihood(
        model, windows, offsets, ref_tokens, alt_tokens,
        effect_sizes=effect_sizes, labels=labels, batch_size=batch_size, verbose=False,
    )
    overall["n"] = len(windows)

    results = {"overall": overall}
    groups = _group_indices_by_class(cre_classes)

    scored, skipped = 0, 0
    for cls, idx in groups.items():
        n = len(idx)

        if n < min_per_class:
            results[cls] = {"n": n, "skipped": f"fewer than {min_per_class} variants"}
            skipped += 1
            continue

        cls_labels = _subset(labels, idx)
        if cls_labels is not None and len(np.unique(np.asarray(cls_labels))) < 2:
            cls_labels = None

        cls_result = run_variant_zero_shot_likelihood(
            model, _subset(windows, idx), _subset(offsets, idx),
            _subset(ref_tokens, idx), _subset(alt_tokens, idx),
            effect_sizes=_subset(effect_sizes, idx), labels=cls_labels,
            batch_size=batch_size, verbose=False,
        )
        cls_result["n"] = n
        results[cls] = cls_result
        scored += 1

    print(f"variant zero-shot likelihood by class complete: {scored} classes scored, {skipped} skipped")

    return results
