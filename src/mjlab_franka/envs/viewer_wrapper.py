"""Thin env wrapper exposing the viewer's expected interface.

mjlab's ``ManagerBasedRlEnv`` doesn't expose ``get_observations()``, which is
required by :class:`mjlab.viewer.ViserPlayViewer`. This wrapper supplies it
without pulling in the heavier rsl_rl/skrl wrappers (we want the raw obs dict
so a checkpoint-loaded policy can pick out the actor group).
"""

from __future__ import annotations

from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv


class ViewerEnvWrapper:
    """Wrap a ``ManagerBasedRlEnv`` for the viewer / playback policy.

    The wrapped env exposes ``get_observations()`` returning the actor obs
    tensor for the configured ``obs_group`` (defaults to ``"policy"``).
    All other attributes pass through to the underlying env.
    """

    def __init__(self, env: ManagerBasedRlEnv, obs_group: str = "policy") -> None:
        self._env = env
        self._obs_group = obs_group
        self._last_obs: torch.Tensor | None = None

    # --- viewer expected api ---

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device | str:
        return self._env.device

    @property
    def cfg(self) -> Any:
        return self._env.cfg

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self._env

    def get_observations(self) -> torch.Tensor:
        if self._last_obs is None:
            self.reset()
        assert self._last_obs is not None
        return self._last_obs

    def reset(self) -> Any:
        obs, info = self._env.reset()
        self._last_obs = obs[self._obs_group]
        return obs, info

    def step(self, action: torch.Tensor) -> Any:
        obs, rew, terminated, truncated, info = self._env.step(action)
        self._last_obs = obs[self._obs_group]
        return obs, rew, terminated, truncated, info

    def close(self) -> None:
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
