"""EpisodeStats — tracks episode-level stats across parallel envs.

Ported near-verbatim from aloy/core/stats.py.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch
from torch import Tensor


class EpisodeStats:
    """Tracks episode-level stats for parallel envs.

    Per-term episode sums (rewards / metrics / terminations / curriculum) are
    consumed from each manager's ``reset(env_ids)`` extras and averaged across
    envs once every env has seen at least one episode end.
    """

    def __init__(
        self, num_envs: int, device: str, buffer_size: int | None = None
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.buffer_size = buffer_size if buffer_size is not None else 2 * num_envs

        self._running_lengths = torch.zeros(num_envs, device=device)
        self._last_lengths = torch.full(
            (num_envs,), float("nan"), dtype=torch.float, device=device
        )

        self._metrics_terms: dict[str, torch.Tensor] = {}
        self._rewards_terms: dict[str, torch.Tensor] = {}
        self._termination_terms: dict[str, torch.Tensor] = {}
        self._curriculum_terms: dict[str, torch.Tensor] = {}

        self._last_env_ids: Tensor | None = None
        self._skip_first_reset = True
        self.logged_rewards_once = False

    def update(
        self,
        env_ids: Tensor | None = None,
        episode_metrics: dict[str, Tensor] | None = None,
        episode_rewards: dict[str, Tensor] | None = None,
        episode_terminations: dict[str, Tensor] | None = None,
        episode_curriculum: dict[str, Tensor] | None = None,
    ) -> None:
        self._running_lengths += 1
        # mjlab keeps the env_ids tensor object between resets; identity check
        # avoids counting the same episode twice.
        if env_ids is self._last_env_ids or env_ids is None:
            return
        self._last_env_ids = env_ids

        if self._skip_first_reset:
            self._skip_first_reset = False
            return

        self._last_lengths[env_ids] = self._running_lengths[env_ids]
        self._running_lengths[env_ids] = 0

        if episode_metrics:
            for name, values in episode_metrics.items():
                self._set_term_values(name, values, self._metrics_terms, env_ids)
        if episode_rewards:
            for name, values in episode_rewards.items():
                self._set_term_values(name, values, self._rewards_terms, env_ids)
        if episode_terminations:
            for name, values in episode_terminations.items():
                self._set_term_values(name, values, self._termination_terms, env_ids)
        if episode_curriculum:
            for name, values in episode_curriculum.items():
                self._set_term_values(name, values, self._curriculum_terms, env_ids)

    def get_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}

        length_stats = self._masked_stats(self._last_lengths)
        if length_stats is not None:
            length_values, _std = length_stats
            stats.update(
                {
                    "Episode / Total timesteps (min)": length_values.min().item(),
                    "Episode / Total timesteps (max)": length_values.max().item(),
                    "Episode / Total timesteps (mean)": length_values.mean().item(),
                    "Episode / count": length_values.numel(),
                }
            )

        for name, buffer in self._metrics_terms.items():
            term_stats = self._masked_stats(buffer)
            if term_stats is None:
                continue
            values, std = term_stats
            stats[f"Metrics / {name}"] = values.mean().item()
            stats[f"Metrics (std) / {name}"] = std

        for name, buffer in self._rewards_terms.items():
            term_stats = self._masked_stats(buffer)
            if term_stats is None:
                continue
            self.logged_rewards_once = True
            values, std = term_stats
            value = values.mean().item()
            stats[f"Rewards / {name}"] = value
            stats[f"Rewards (std) / {name}"] = std
            current_total = stats.get("Episode / Total Reward", 0.0)
            stats["Episode / Total Reward"] = current_total + value

        for name, buffer in self._termination_terms.items():
            term_stats = self._masked_stats(buffer)
            if term_stats is None:
                continue
            values, std = term_stats
            stats[f"Terminations / {name}"] = values.mean().item()
            stats[f"Terminations (std) / {name}"] = std

        for name, buffer in self._curriculum_terms.items():
            term_stats = self._masked_stats(buffer)
            if term_stats is None:
                continue
            values, _std = term_stats
            stats[f"Curriculum / {name}"] = values.mean().item()

        return stats

    def _set_term_values(
        self,
        name: str,
        value: float | Tensor,
        collection: dict[str, torch.Tensor],
        env_ids: Tensor,
    ) -> None:
        if name not in collection:
            collection[name] = torch.full(
                (self.num_envs,), float("nan"), dtype=torch.float, device=self.device
            )
        if isinstance(value, Tensor):
            collection[name][env_ids] = value.to(self.device)
        elif isinstance(value, Real):
            collection[name][env_ids] = float(value)
        else:
            raise ValueError(
                f"Unsupported logging value type for term '{name}': {type(value)}"
            )

    def _masked_stats(self, values: torch.Tensor) -> tuple[torch.Tensor, float] | None:
        mask = ~torch.isnan(values)
        if not mask.all():
            return None
        valid = values[mask]
        std = valid.std(unbiased=False).item() if valid.numel() > 1 else 0.0
        return valid, std


def skrl_agent_metrics(agent: Any) -> dict[str, float]:
    """Extract metrics from a skrl agent's ``tracking_data`` and clear it."""
    metrics: dict[str, float] = {}
    if not agent.tracking_data:
        return metrics
    for key, values in agent.tracking_data.items():
        if not values:
            continue
        metrics[key] = sum(values) / len(values)
    agent._track_rewards.clear()  # noqa: SLF001
    agent._track_timesteps.clear()  # noqa: SLF001
    agent.tracking_data.clear()
    return metrics
