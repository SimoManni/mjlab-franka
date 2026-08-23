"""Hydra preset: Franka trajectory-tracking with SKRL PPO."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from mjlab_franka.algos.ppo_phases import PPOPhasesConfig
from mjlab_franka.config.base import EnvConfig, TrackerConfig, TrainConfig


@dataclass
class FrankaCubeRelocationPPOPhasesExperiment(TrainConfig):
    """Experiment defaults: Franka cube relocation with PPO phases."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"override /algo": "ppo_phases"},
            "_self_",
        ]
    )

    name: str | None = "franka_cube_relocation_ppo_phases"
    env: EnvConfig = field(
        default_factory=lambda: EnvConfig(
            task="franka_cube_relocation",
            num_envs=4096,
            device="cuda:0",
        )
    )
    algo: PPOPhasesConfig = field(
        default_factory=lambda: PPOPhasesConfig(
            timesteps=150_000,
            rollouts=16,
            learning_rate=3e-4,
            mini_batches=8,
            learning_epochs=4,
            entropy_loss_scale=0.005,
            num_phases=4,
        )
    )
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualize: bool = True


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    name="franka_cube_relocation_ppo_phases",
    node=FrankaCubeRelocationPPOPhasesExperiment,
    package="_global_",
)
