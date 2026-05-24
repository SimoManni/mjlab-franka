"""Metrics manager that records raw reward outputs and user-defined metrics.

Ported from aloy/managers/metrics_manager.py (Copyable/SMPC bits removed).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerBase
from mjlab.managers.metrics_manager import MetricsTermCfg
from prettytable import PrettyTable

from mjlab_franka.managers.reward_manager import RewardManager


class MetricsManager(ManagerBase):
    """Accumulates per-step raw reward outputs + user-defined metric terms.

    For every step:
      - raw reward outputs (unweighted) are read from ``reward_manager.step_raw``
      - user-defined ``MetricsTermCfg.func(env, **params)`` are evaluated
    Episode sums per env are returned at ``reset()`` under ``_episode_metrics``.
    """

    _env: ManagerBasedRlEnv

    def __init__(
        self,
        cfg: dict[str, MetricsTermCfg],
        env: ManagerBasedRlEnv,
        *,
        reward_manager: RewardManager,
    ) -> None:
        self._metric_term_names: list[str] = []
        self._metric_term_cfgs: list[MetricsTermCfg] = []
        self._metric_class_term_cfgs: list[MetricsTermCfg] = []

        self._reward_manager = reward_manager
        self.cfg = deepcopy(cfg)
        super().__init__(env=env)

        self._reward_term_names = list(self._reward_manager.active_terms)
        overlap = set(self._reward_term_names) & set(self._metric_term_names)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"Metrics terms overlap with reward terms: {names}")

        self._term_names = self._reward_term_names + self._metric_term_names
        self._episode_sums: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )

        self._step_values = torch.zeros(
            (self.num_envs, len(self._term_names)),
            dtype=torch.float,
            device=self.device,
        )

    def __str__(self) -> str:
        msg = f"<MetricsManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Metrics Terms"
        table.field_names = ["Index", "Name", "Source"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._term_names):
            source = "reward" if index < len(self._reward_term_names) else "metric"
            table.add_row([index, name, source])
        msg += table.get_string()
        msg += "\n"
        return msg

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    def reset(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> dict[str, torch.Tensor]:
        """Return per-env episode metrics for EpisodeStats to aggregate."""
        if env_ids is None:
            env_ids = slice(None)
        extras: dict[str, torch.Tensor] = {}

        if isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0:
            return extras

        episode_metrics: dict[str, torch.Tensor] = {}
        for key in self._term_names:
            episode_metrics[key] = self._episode_sums[key][env_ids].clone()
            self._episode_sums[key][env_ids] = 0.0

        extras["_episode_metrics"] = episode_metrics

        for term_cfg in self._metric_class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

        return extras

    def compute_substep(self) -> None:
        """No-op: per-substep metrics are not supported."""

    def compute(self) -> None:
        """Compute all metric terms for the current step and accumulate sums."""
        if not self._term_names:
            return

        # Reward terms: read raw values cached by the reward manager.
        if self._reward_term_names:
            reward_step_values = self._reward_manager.step_raw
            for idx, name in enumerate(self._reward_term_names):
                value = reward_step_values[:, idx]
                self._episode_sums[name] += value
                self._step_values[:, idx] = value

        # Metric-only terms.
        offset = len(self._reward_term_names)
        for term_idx, (name, term_cfg) in enumerate(
            zip(self._metric_term_names, self._metric_term_cfgs, strict=True)
        ):
            value = term_cfg.func(self._env, **term_cfg.params)
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            self._episode_sums[name] += value
            self._step_values[:, offset + term_idx] = value

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        terms: list[tuple[str, list[float]]] = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_values[env_idx, idx].cpu().item()]))
        return terms

    def _prepare_terms(self) -> None:
        for term_name, term_cfg in self.cfg.items():
            if term_cfg is None:
                continue
            self._resolve_common_term_cfg(term_name, term_cfg)
            self._metric_term_names.append(term_name)
            self._metric_term_cfgs.append(term_cfg)
            if hasattr(term_cfg.func, "reset") and callable(term_cfg.func.reset):
                self._metric_class_term_cfgs.append(term_cfg)
