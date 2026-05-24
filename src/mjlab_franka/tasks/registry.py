"""Task registry — maps task names to env config factories.

Mirrors ``aloy.tasks.registry``.
"""

from __future__ import annotations

from typing import Callable, Protocol

from mjlab.envs import ManagerBasedRlEnvCfg


class TaskFactory(Protocol):
    def __call__(self, num_envs: int, play: bool = False) -> ManagerBasedRlEnvCfg: ...


task_registry: dict[str, TaskFactory] = {}


def register_task(name: str) -> Callable[[TaskFactory], TaskFactory]:
    """Decorator to register a task config factory."""

    def decorator(factory: TaskFactory) -> TaskFactory:
        if name in task_registry:
            raise ValueError(f"Task '{name}' already registered")
        task_registry[name] = factory
        return factory

    return decorator
