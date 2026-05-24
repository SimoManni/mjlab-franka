"""Tracker — orchestrates training bookkeeping.

Composes :class:`WandbLogger`, :class:`CheckpointSaver`, and :class:`EpisodeStats`.
Ported from aloy/core/tracker.py with ONNX export and per-phase best-slot
mechanics removed.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from mjlab_franka.config.base import TrainConfig
from mjlab_franka.core.checkpoint import init_from_checkpoint, resume_training
from mjlab_franka.core.checkpoint_saver import CheckpointSaver
from mjlab_franka.core.stats import EpisodeStats
from mjlab_franka.core.wandb_logger import WandbLogger

log = logging.getLogger(__name__)


class Tracker:
    """Training bookkeeping orchestrator."""

    def __init__(
        self,
        train_cfg: TrainConfig,
        device: str,
        *,
        agent_modules: dict,
        env: Any,
    ) -> None:
        tracker_cfg = train_cfg.tracker
        if train_cfg.name is None:
            raise ValueError("train_cfg.name must be set before creating Tracker")

        self.num_envs = env.num_envs

        # Resume or warm-start from a checkpoint.
        self.start_step = 0
        if train_cfg.resume:
            self.start_step = resume_training(
                train_cfg.resume,
                device,
                agent_modules,
                wandb_filename=train_cfg.wandb_filename,
            )
            env.unwrapped.common_step_counter = self.start_step
        elif train_cfg.init_from:
            init_from_checkpoint(
                train_cfg.init_from,
                device,
                agent_modules,
                wandb_filename=train_cfg.wandb_filename,
            )

        # Log dir: {log_dir}/{experiment_name}/{timestamp}/
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = (
            Path(tracker_cfg.log_dir).expanduser() / train_cfg.name / timestamp
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.log_dir / "checkpoints"

        self.stats = EpisodeStats(env.num_envs, device)

        # WandbLogger (None = disabled).
        self._wandb: WandbLogger | None = None
        if tracker_cfg.wandb is not None:
            train_cfg_dict = OmegaConf.to_container(
                OmegaConf.structured(train_cfg), resolve=True
            )
            self._wandb = WandbLogger(
                tracker_cfg.wandb,
                train_cfg.name,
                train_cfg.env.task,
                self.log_dir,
                train_cfg_dict,
            )

        # CheckpointSaver (None = disabled).
        self._saver: CheckpointSaver | None = None
        if tracker_cfg.checkpoint is not None:
            algo_config = OmegaConf.to_container(
                OmegaConf.structured(train_cfg.algo), resolve=True
            )
            self._saver = CheckpointSaver(
                tracker_cfg.checkpoint,
                self.checkpoint_dir,
                agent_modules,
                algo_config,
                start_step=self.start_step,
            )

        self._log_interval = tracker_cfg.log_interval
        self._env_steps = 0
        self._last_log_step = self.start_step
        self._last_log_step_time = time.perf_counter()
        self._last_step = self.start_step
        self._term_counts: dict[str, int] = {}
        self._term_episodes: int = 0

        log.info(f"Logging to: {self.log_dir}")

    @staticmethod
    def _get_curriculum_from_extras(extras: dict) -> dict[str, Any]:
        episode_curriculum: dict[str, Any] = {}
        for key, value in extras.items():
            if "Curriculum" not in key:
                continue
            name = key.split("/")[-1]
            episode_curriculum[name] = value
        return episode_curriculum

    @staticmethod
    def _get_terminations_from_extras(extras: dict) -> dict[str, int]:
        result: dict[str, int] = {}
        for key, value in extras.items():
            if not key.startswith("Episode_Termination/"):
                continue
            name = key.split("/", 1)[-1]
            result[name] = int(value)
        return result

    # --- Public API ---

    def track_config(self, config: dict[str, Any]) -> None:
        if self._wandb is not None:
            self._wandb.log_config(config)

    def record(self, extras: dict | None = None) -> None:
        """Call after each ``env.step()`` to accumulate episode stats."""
        self._env_steps += 1

        env_ids = extras.get("_episode_env_ids") if extras else None
        episode_metrics = extras.get("_episode_metrics") if extras else None
        episode_rewards = extras.get("_episode_rewards") if extras else None
        episode_terminations = extras.get("_episode_terminations") if extras else None
        episode_curriculum = (
            self._get_curriculum_from_extras(extras) if extras else None
        )

        if extras and env_ids is not None and len(env_ids) > 0:
            term_counts = self._get_terminations_from_extras(extras)
            if term_counts:
                self._term_episodes += len(env_ids)
                for name, count in term_counts.items():
                    self._term_counts[name] = self._term_counts.get(name, 0) + count

        self.stats.update(
            env_ids,
            episode_metrics,
            episode_rewards,
            episode_terminations,
            episode_curriculum,
        )

    def flush(
        self,
        step: int,
        progress_bar: Any | None = None,
        algo_stats: dict | None = None,
    ) -> None:
        """Call after each agent update to log metrics and save checkpoints."""
        self._last_step = step

        if step - self._last_log_step >= self._log_interval:
            now = time.perf_counter()
            log_interval_steps = step - self._last_log_step
            log_interval_time = now - self._last_log_step_time
            self._last_log_step = step
            self._last_log_step_time = now

            metrics = self.stats.get_stats()
            metrics["Charts / Total Experiences"] = self._env_steps * self.num_envs
            metrics["Charts / Experiences per Second"] = (
                (log_interval_steps * self.num_envs) / log_interval_time
                if log_interval_time > 0
                else 0.0
            )

            if self._term_episodes > 0:
                for name, count in self._term_counts.items():
                    metrics[f"Terminations / {name}"] = count / self._term_episodes
                self._term_counts.clear()
                self._term_episodes = 0

            if algo_stats:
                metrics.update(algo_stats)

            if metrics and self._wandb is not None:
                self._wandb.log(metrics, step)

            if progress_bar is not None and metrics:
                total = metrics.get("Episode / Total Reward")
                if total is not None:
                    progress_bar.set_postfix_str(f"reward={total:.3f}", refresh=False)

            if self._saver is not None:
                self._saver.check_best(metrics, self.stats.logged_rewards_once)

        if self._saver is not None:
            save_paths = self._saver.maybe_save(step)
            for path in save_paths:
                if self._wandb is not None:
                    self._wandb.save_file(path)

            if save_paths and progress_bar is not None:
                progress_bar.set_postfix_str(f"last save = {step}", refresh=False)

    def finish(self) -> None:
        """Save a final checkpoint and close the wandb run."""
        if self._saver is not None and self._last_step > self._saver.last_save_step:
            for path in self._saver.save(self._last_step):
                if self._wandb is not None:
                    self._wandb.save_file(path)
        if self._wandb is not None:
            self._wandb.finish()
        log.info(f"Training complete. Checkpoints in: {self.checkpoint_dir}")
