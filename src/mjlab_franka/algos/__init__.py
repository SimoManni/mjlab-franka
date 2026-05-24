"""Algorithm registry — register all algos here."""

from mjlab_franka.algos.registry import algo_registry, register_algo

# Imported so @register_algo decorators run.
from mjlab_franka.algos import ppo_skrl  # noqa: F401

__all__ = ["algo_registry", "register_algo"]
