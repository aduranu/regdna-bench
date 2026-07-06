from regdna_bench.tasks.base import Task

# import task modules so they self-register with the registry on package import
from regdna_bench.tasks import vep  # noqa: F401

__all__ = ["Task"]
