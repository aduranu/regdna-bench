"""task runner: capability check, then prepare -> run -> metrics."""

from __future__ import annotations

from regdna_bench.base import CapabilityError


def run_task(task, model, dataset):
    # fail fast if the model lacks a capability the task needs, before any work.
    for cap in task.requires:
        if not isinstance(model, cap):
            raise CapabilityError(
                f"{type(model).__name__} lacks {cap.__name__} required by task {task.name!r}")

    prepared = task.prepare(dataset)
    output = task.run(model, prepared)

    return prepared, output, task.metrics(prepared, output)
