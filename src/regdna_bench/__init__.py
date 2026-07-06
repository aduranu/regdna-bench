from regdna_bench.base import (
    CANON,
    BenchModel,
    CapabilityError,
    EmbeddingModel,
    LikelihoodModel,
)
from regdna_bench.data import Dataset, VariantSet
from regdna_bench.registry import get_task, list_tasks, register_task
from regdna_bench.runner import run_task
from regdna_bench.tasks import Task

# importing regdna_bench.tasks registers the built-in tasks

__all__ = [
    "CANON",
    "BenchModel",
    "CapabilityError",
    "EmbeddingModel",
    "LikelihoodModel",
    "Dataset",
    "VariantSet",
    "Task",
    "get_task",
    "list_tasks",
    "register_task",
    "run_task",
]
