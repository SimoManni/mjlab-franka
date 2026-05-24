"""Task package — importing this triggers registration of all built-in tasks."""

from mjlab_franka.tasks.registry import register_task, task_registry

# Import for the @register_task decorator side-effect.
from mjlab_franka.tasks.reaching import franka_reaching_spline  # noqa: F401

__all__ = ["register_task", "task_registry"]
