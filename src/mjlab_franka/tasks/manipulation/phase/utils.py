"""Utility functions for wrapping, computing, and phase-masking rewards and terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, Union

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def phase_masked_reward(
    env: ManagerBasedRlEnv,
    reward_fn_or_class: Union[Callable, type],
    active_phases: Sequence[int],
    phase_command_name: str = "phase_command",
    **kwargs,
) -> torch.Tensor:
    """Wrapper function to compute a reward (via function or class instance)
    and mask it out if the current environment's phase is not in the active phases list.
    """
    if isinstance(reward_fn_or_class, type):
        cache_key = f"_reward_instance_{reward_fn_or_class.__name__}"
        if not hasattr(env, cache_key):
            # Pass kwargs (which include source_cfg, target_cfg, distance_type, etc.) directly into the class!
            instance = reward_fn_or_class(None, env, **kwargs)
            setattr(env, cache_key, instance)
        reward_callable = getattr(env, cache_key)
    else:
        reward_callable = reward_fn_or_class

    # Compute underlying reward values
    if hasattr(reward_callable, "__call__") and isinstance(reward_fn_or_class, type):
        raw_rewards = reward_callable(env)
    else:
        raw_rewards = reward_callable(env, **kwargs)

    # Fetch phase command and mask rewards outside active phases
    phase_term = env.command_manager.get_term(phase_command_name)
    in_active_phase = phase_term.is_in_phases(active_phases)

    masked_rewards = torch.where(in_active_phase, raw_rewards, torch.zeros_like(raw_rewards))
    return masked_rewards


def phase_masked_termination(
    env: ManagerBasedRlEnv,
    term_fn: Callable,
    active_phases: Sequence[int],
    phase_command_name: str = "phase_command",
    **kwargs,
) -> torch.Tensor:
    """Wrapper function to evaluate a termination condition function

    and suppress it (return False) if the environment is not in an active phase.
    """
    # 1. Evaluate the underlying termination condition
    raw_terminations = term_fn(env, **kwargs)

    # 2. Fetch the phase command term and check active status
    phase_term = env.command_manager.get_term(phase_command_name)
    in_active_phase = phase_term.is_in_phases(active_phases)

    # 3. Only permit termination if active AND condition is met
    masked_terminations = raw_terminations & in_active_phase
    return masked_terminations