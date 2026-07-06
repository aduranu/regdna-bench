from regdna_bench.tasks.base import Task

# import task families so they self-register with the registry on package import
from regdna_bench.tasks import short_ctx  # noqa: F401

__all__ = ["Task"]
