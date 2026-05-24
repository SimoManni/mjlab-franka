"""Hydra preset: Franka trajectory-tracking with SKRL PPO."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from mjlab_franka.algos.ppo_skrl import PPOSkrlConfig
from mjlab_franka.config.base import EnvConfig, TrackerConfig, TrainConfig


@dataclass
class FrankaTrajectoryTrackPPOExperiment(TrainConfig):
    """Experiment defaults: Franka trajectory tracking with PPO."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"override /algo": "ppo_skrl"},
            "_self_",
        ]
    )

    name: str | None = "franka_trajectory_track_ppo"
    env: EnvConfig = field(
        default_factory=lambda: EnvConfig(
            task="franka_trajectory_track_spline",
            num_envs=4096,
            device="cuda:0",
        )
    )
    algo: PPOSkrlConfig = field(
        default_factory=lambda: PPOSkrlConfig(
            timesteps=150_000,
            rollouts=16,
            learning_rate=3e-4,
            mini_batches=8,
            learning_epochs=4,
            entropy_loss_scale=0.005,
        )
    )
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualize: bool = True


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    name="franka_trajectory_track_ppo",
    node=FrankaTrajectoryTrackPPOExperiment,
    package="_global_",
)
