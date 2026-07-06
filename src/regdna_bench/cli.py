"""command-line entry point: list tasks, run a task against a model.

the core stays model-agnostic: a model is supplied as an import spec
("module:Attr.factory" or "path/to/file.py:Attr.factory") that returns a
BenchModel. the example adapters live outside the installed package, so this is
how the CLI reaches them without the core importing any concrete model.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pathlib

from regdna_bench.data.caqtl import CaqtlVariantDataset
from regdna_bench.registry import get_task, list_tasks
from regdna_bench.runner import run_task

# zoonomia training defaults; model, genome store and cCREs are all hg38.
DEFAULT_CKPT = "/grid/koo/home/shared/d3/trained_weights/zoonomia/best/model-epoch=207-val_loss=395.0214.ckpt"
DEFAULT_BED = "/grid/koo/home/shared/d3/data/zoonomia/GRCh38-cCREs.bed"
DEFAULT_ZARR = "~/scratch/d3-dna/zoonomia/zoonomia_v3.zarr"
DEFAULT_H5 = "/grid/koo/home/shared/d3/data/zoonomia/zoonomia_241.h5"


def _resolve(spec):
    # "module:attr" or "file.py:attr"; attr may be dotted (Class.classmethod).
    mod_part, _, attr_part = spec.partition(":")
    if not attr_part:
        raise ValueError(f"model spec must be 'module:attr', got {spec!r}")

    if mod_part.endswith(".py") or pathlib.Path(mod_part).exists():
        module_spec = importlib.util.spec_from_file_location("_regdna_model", mod_part)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)

    obj = module
    for name in attr_part.split("."):
        obj = getattr(obj, name)

    return obj


def summarize(drops, genome_backend, window_mode, task_name, summary):
    return {
        "task": task_name,
        "window_mode": window_mode,
        "genome_backend": genome_backend,
        "drops": dict(drops) if drops else {},
        "results": summary,
    }


def print_summary(summary, drops):
    print("drop reasons:", dict(drops) if drops else {})

    keys = ["overall"] + sorted(k for k in summary if k != "overall")
    for key in keys:
        entry = summary[key]
        if "skipped" in entry:
            print(f"  {key}: n={entry['n']} skipped ({entry['skipped']})")
            continue

        clf = entry.get("metrics", {}).get("classification", {})
        line = f"  {key}: n={entry['n']}"
        if "auroc" in clf:
            line += f" auroc={clf['auroc']:.4f} auprc={clf['auprc']:.4f}"
        print(line)


def cmd_list_tasks(_args):
    for name in list_tasks():
        print(name)


def cmd_run(args):
    factory = _resolve(args.model)
    model = factory(args.checkpoint, device=args.device)

    dataset = CaqtlVariantDataset(
        variants=args.variants, ccre_bed=args.ccre_bed,
        zarr_path=args.zarr, h5_path=args.h5,
        chroms=args.chroms.split(","), window_mode=args.window_mode,
    )

    task = get_task(args.task, batch_size=args.batch_size, min_per_class=args.min_per_class)

    _prepared, _output, summary = run_task(task, model, dataset)

    print_summary(summary, dataset.drops)

    if args.out is not None:
        payload = summarize(dataset.drops, dataset.genome_backend, args.window_mode,
                            args.task, summary)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(prog="regdna-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-tasks", help="list registered tasks")

    run = sub.add_parser("run", help="run a task against a model")
    run.add_argument("task", help="task name (see list-tasks)")
    run.add_argument("--model", required=True,
                     help="import spec returning a BenchModel, e.g. examples/d3/wrapper.py:D3Model.from_pretrained")
    run.add_argument("--variants", required=True, help="caQTL tsv (hg38)")
    run.add_argument("--window-mode", choices=["element", "variant"], default="element")
    run.add_argument("--checkpoint", default=DEFAULT_CKPT)
    run.add_argument("--ccre-bed", default=DEFAULT_BED)
    run.add_argument("--zarr", default=DEFAULT_ZARR)
    run.add_argument("--h5", default=DEFAULT_H5)
    run.add_argument("--chroms", default="chr22,chrX", help="held-out test chroms")
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--min-per-class", type=int, default=10)
    run.add_argument("--out", default=None, help="optional json summary path")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-tasks":
        cmd_list_tasks(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
