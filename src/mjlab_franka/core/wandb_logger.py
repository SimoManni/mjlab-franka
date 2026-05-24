"""Wandb logger — init, metric logging, file upload, finish.

Ported from aloy/core/wandb_logger.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import wandb

from mjlab_franka.config.base import WandbConfig

log = logging.getLogger(__name__)


class WandbLogger:
    """Owns the wandb run lifecycle for a Tracker."""

    def __init__(
        self,
        wandb_cfg: WandbConfig,
        experiment_name: str,
        task: str,
        log_dir: Path,
        train_config: dict,
    ) -> None:
        project = wandb_cfg.project if wandb_cfg.project is not None else task
        self._log_dir = log_dir
        self._run = wandb.init(
            project=project,
            entity=wandb_cfg.entity,
            group=wandb_cfg.group,
            name=experiment_name,
            tags=list(wandb_cfg.tags) if wandb_cfg.tags else None,
            dir=str(log_dir),
            config=train_config,
            id=wandb_cfg.run_id,
            resume=wandb_cfg.resume,
        )
        log.info(f"WandB: {self._run.url}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update(config)

    def save_file(self, path: Path) -> None:
        """Upload a file to the active wandb run immediately."""
        wandb.save(str(path), base_path=str(self._log_dir), policy="now")

    def finish(self) -> None:
        self._run.finish()
