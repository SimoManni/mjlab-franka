"""Lightweight env factory mirroring ``aloy.envs.factory.make_env``."""

from __future__ import annotations

import logging
import random

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_franka.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab_franka.tasks import task_registry

log = logging.getLogger(__name__)


def make_env(
    task: str,
    num_envs: int = 4096,
    device: str = "cuda:0",
    seed: int = -1,
    play: bool = False,
) -> ManagerBasedRlEnv:
    """Build a :class:`mjlab.envs.ManagerBasedRlEnv` from the task registry."""
    if task not in task_registry:
        raise KeyError(f"Unknown task '{task}'. Available: {sorted(task_registry)}")

    seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
    cfg: ManagerBasedRlEnvCfg = task_registry[task](num_envs=num_envs, play=play)
    cfg.seed = seed

    log.info(
        "Task: %s | num_envs=%d | device=%s | seed=%d", task, num_envs, device, seed
    )
    env = ManagerBasedRlEnv(cfg, device=device)
    log.info(
        "Obs space: %s | Action space: %s",
        env.single_observation_space,
        env.single_action_space,
    )
    return env
