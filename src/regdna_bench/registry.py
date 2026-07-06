"""task registry: name -> Task class. tasks self-register with @register_task."""

from __future__ import annotations

TASKS = {}


def register_task(cls):
    TASKS[cls.name] = cls

    return cls


def get_task(name, **kwargs):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; have {list_tasks()}")

    return TASKS[name](**kwargs)


def list_tasks():
    return sorted(TASKS)
