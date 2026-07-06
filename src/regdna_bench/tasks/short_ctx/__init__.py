"""short-context task family: tasks scored on single short windows (~350 bp).

covers the zero-shot variant-effect tasks (vep) and frozen-embedding probing.
importing this package registers the vep tasks with the registry; probing is
still function-based (run_probing / run_cnn_probing) and invoked directly.
"""

from regdna_bench.tasks.short_ctx import vep  # noqa: F401  registers vep tasks
from regdna_bench.tasks.short_ctx import probing  # noqa: F401

__all__ = ["vep", "probing"]
