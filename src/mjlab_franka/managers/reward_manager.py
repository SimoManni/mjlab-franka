"""Reward manager that exposes raw, unweighted term values for metrics logging.

Ported from aloy/managers/reward_manager.py (the Copyable/SMPC bits removed —
mjlab_franka does not need them).
"""

from __future__ import annotations

from typing import Any

import torch
from mjlab.managers.reward_manager import RewardManager as MjlabRewardManager


class RewardManager(MjlabRewardManager):
    """Reward manager that mirrors mjlab's scaling but always computes every
    term (including weight=0.0) and caches per-step raw values for metrics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._step_raw = torch.zeros(
            (self.num_envs, len(self._term_names)),
            dtype=torch.float,
            device=self.device,
        )
        self._dt_scale = self._env.step_dt if self._scale_by_dt else 1.0

    @property
    def step_raw(self) -> torch.Tensor:
        """Raw per-step values for each reward term (unweighted, unscaled)."""
        return self._step_raw

    def compute(self, dt: float) -> torch.Tensor:
        """Compute rewards and cache raw term values for metrics."""
        self._reward_buf[:] = 0.0

        for term_idx, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=True)
        ):
            raw_value = term_cfg.func(self._env, **term_cfg.params)
            raw_value = torch.nan_to_num(raw_value, nan=0.0, posinf=0.0, neginf=0.0)

            self._step_raw[:, term_idx] = raw_value

            reward = raw_value * term_cfg.weight * self._dt_scale
            self._reward_buf += reward
            self._episode_sums[name] += reward
            self._step_reward[:, term_idx] = reward

        return self._reward_buf

    def get_active_iterable_terms(self, env_idx: int) -> list[tuple[str, list[float]]]:
        """Return per-term step rewards (weighted, scaled like episode sums)."""
        scale = self._env.step_dt if self._scale_by_dt else 1.0
        terms: list[tuple[str, list[float]]] = []
        for idx, (name, term_cfg) in enumerate(
            zip(self._term_names, self._term_cfgs, strict=True)
        ):
            value = (
                (self._step_raw[env_idx, idx] * term_cfg.weight * scale).cpu().item()
            )
            terms.append((name, [value]))
        return terms

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, Any]:
        """Return per-env episode rewards for EpisodeStats to aggregate."""
        if env_ids is None:
            env_ids = slice(None)

        extras: dict[str, Any] = {}

        if isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0:
            return extras

        episode_rewards: dict[str, torch.Tensor] = {}

        for name, term_cfg in zip(self._term_names, self._term_cfgs, strict=True):
            if term_cfg.weight != 0.0:
                episode_rewards[name] = self._episode_sums[name][env_ids].clone()
            self._episode_sums[name][env_ids] = 0.0

        extras["_episode_env_ids"] = (
            env_ids if isinstance(env_ids, torch.Tensor) else None
        )
        extras["_episode_rewards"] = episode_rewards

        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

        return extras
