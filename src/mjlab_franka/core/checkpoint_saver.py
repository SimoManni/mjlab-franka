"""Checkpoint saving: periodic saves + single-slot best-model tracking.

Simplified from aloy/core/checkpoint_saver.py — dropped ONNX/Exporter and
multi-slot best tracking (we only need a single ``best_agent.pt``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from mjlab_franka.config.base import CheckpointConfig
from mjlab_franka.core.checkpoint import checkpoint_filename, save_training_state

log = logging.getLogger(__name__)


class CheckpointSaver:
    """Periodic + best-model checkpoint writer."""

    def __init__(
        self,
        checkpoint_cfg: CheckpointConfig,
        checkpoint_dir: Path,
        agent_modules: dict,
        algo_config: dict,
        *,
        start_step: int = 0,
    ) -> None:
        self._cfg = checkpoint_cfg
        self._checkpoint_dir = checkpoint_dir
        self._agent_modules = agent_modules
        self._algo_config = algo_config
        self._last_save_step = start_step

        if self._cfg.best_metric_mode not in {"max", "min"}:
            raise ValueError(
                f"Unsupported best_metric_mode: {self._cfg.best_metric_mode}"
            )
        self._best_value = (
            float("-inf") if self._cfg.best_metric_mode == "max" else float("inf")
        )
        self._is_best = False

    @property
    def last_save_step(self) -> int:
        return self._last_save_step

    @property
    def save_final(self) -> bool:
        return self._cfg.save_interval > 0

    def check_best(self, metrics: dict[str, float], logged_rewards_once: bool) -> None:
        if self._cfg.best_metric is None:
            return
        current = metrics.get(self._cfg.best_metric)
        if current is None:
            if logged_rewards_once:
                log.warning(
                    f"Requested best_metric '{self._cfg.best_metric}' not in tracked metrics."
                )
            return

        is_max = self._cfg.best_metric_mode == "max"
        is_better = current > self._best_value if is_max else current < self._best_value
        if is_better:
            self._best_value = current
            self._is_best = True
            log.info(f"New best {self._cfg.best_metric}: {current:.4f}")

    def maybe_save(self, step: int) -> list[Path]:
        if step - self._last_save_step < self._cfg.save_interval:
            return []
        self._last_save_step = step
        paths = self._write(step)
        if self._is_best:
            paths.extend(self._write(step, filename="best_agent.pt"))
            self._is_best = False
        return paths

    def save(self, step: int) -> list[Path]:
        self._last_save_step = step
        return self._write(step)

    def _write(self, step: int, filename: str | None = None) -> list[Path]:
        if filename is None:
            filename = checkpoint_filename(step)
        path = self._checkpoint_dir / filename
        state = {
            "step": step,
            "agent": {name: m.state_dict() for name, m in self._agent_modules.items()},
            "algo_config": self._algo_config,
        }
        save_training_state(state, path)
        return [path]
