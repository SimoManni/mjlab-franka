"""Base configuration dataclasses for mjlab_franka training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class WandbConfig:
    """Wandb logging configuration."""

    project: str | None = None  # Defaults to task name if unset
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    run_id: str | None = None
    resume: str | None = None


@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""

    save_interval: int = 2000
    """Save a checkpoint every N timesteps."""
    best_metric: str | None = None
    """Metric name used for tracking the best model (e.g. ``Metrics / ee_traj_distance_m``)."""
    best_metric_mode: str = "max"
    """``"max"`` or ``"min"`` — direction in which higher/lower is better."""


@dataclass
class TrackerConfig:
    """Logging / checkpointing configuration."""

    log_dir: str = "~/.cache/mjlab_franka/runs/"
    experiment_name: str | None = None  # Defaults to algo_type if unset
    log_interval: int = 100
    wandb: WandbConfig | None = field(default_factory=WandbConfig)
    checkpoint: CheckpointConfig | None = field(default_factory=CheckpointConfig)


@dataclass
class EnvConfig:
    """Environment configuration (task, num_envs, seed, device)."""

    task: str = MISSING  # Looked up in `mjlab_franka.tasks.task_registry`
    num_envs: int = 4096
    seed: int = -1  # negative => random
    device: str = "cuda:0"
    play: bool = False


@dataclass
class TrainConfig:
    """Top-level training config (composed by Hydra)."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"optional algo": None},
            "_self_",
            {"optional experiment": None},
        ]
    )

    name: str | None = None
    log_level: str = "INFO"
    algo: Any = MISSING
    env: EnvConfig = field(default_factory=EnvConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualize: bool = True  # Launch viser visualizer during training
    resume: str | None = None  # Path or wandb URL to resume training from
    init_from: str | None = None  # Path or wandb URL to warm-start weights only
    wandb_filename: str = "latest"  # Checkpoint filename when pulling from wandb

    def __post_init__(self) -> None:
        if self.name is None and hasattr(self.algo, "algo_type"):
            self.name = self.algo.algo_type


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
