"""runner: zero-shot variant-effect prediction per CRE class, for D3.

assembles 350bp ref/alt windows for caQTL variants and scores them with one of
the zero-shot tasks, reporting auroc/auprc (significant vs background) overall
and per cCRE class.

requires `pip install -e .` (regdna_bench on the path). wrapper.py and config.py
are siblings, imported directly.

usage:
    python examples/d3/run.py --variants /path/Afr.CaQTLS.tsv \
        --method likelihood --window-mode variant
"""

from __future__ import annotations

import argparse
import json

from config import DEFAULT_BED, DEFAULT_CKPT, DEFAULT_H5, DEFAULT_ZARR
from wrapper import D3Model

from regdna_bench.data.caqtl import CaqtlVariantDataset
from regdna_bench.registry import get_task
from regdna_bench.runner import run_task

# map the historical --method flag to task names
METHOD_TASKS = {"embedding": "vep-embedding", "likelihood": "vep-likelihood"}


def print_summary(summary, drops):
    print("drop reasons:", dict(drops) if drops else {})

    for key in ["overall"] + sorted(k for k in summary if k != "overall"):
        entry = summary[key]
        if "skipped" in entry:
            print(f"  {key}: n={entry['n']} skipped ({entry['skipped']})")
            continue

        clf = entry.get("metrics", {}).get("classification", {})
        line = f"  {key}: n={entry['n']}"
        if "auroc" in clf:
            line += f" auroc={clf['auroc']:.4f} auprc={clf['auprc']:.4f}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="zero-shot variant-effect prediction per CRE class for D3")
    parser.add_argument("--variants", required=True, help="caQTL tsv (hg38)")
    parser.add_argument("--method", choices=["embedding", "likelihood"], default="embedding")
    parser.add_argument("--window-mode", choices=["element", "variant"], default="element",
                        help="350bp window anchored on the cCRE midpoint (element) or the variant")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--ccre-bed", default=DEFAULT_BED)
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--chroms", default="chr22,chrX", help="held-out test chroms")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--out", default=None, help="optional json summary path")
    args = parser.parse_args()

    dataset = CaqtlVariantDataset(
        variants=args.variants, ccre_bed=args.ccre_bed,
        zarr_path=args.zarr, h5_path=args.h5,
        chroms=args.chroms.split(","), window_mode=args.window_mode,
    )

    model = D3Model.from_pretrained(args.checkpoint, device=args.device)

    task = get_task(METHOD_TASKS[args.method], min_per_class=args.min_per_class)

    _prepared, _output, summary = run_task(task, model, dataset)

    print_summary(summary, dataset.drops)

    if args.out is not None:
        with open(args.out, "w") as handle:
            json.dump({"method": args.method, "window_mode": args.window_mode,
                       "genome_backend": dataset.genome_backend,
                       "drops": dict(dataset.drops), "results": summary},
                      handle, indent=2)


if __name__ == "__main__":
    main()
