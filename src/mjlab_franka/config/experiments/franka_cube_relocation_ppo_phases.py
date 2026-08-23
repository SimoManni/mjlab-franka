"""Hydra preset: Franka trajectory-tracking with SKRL PPO."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from mjlab_franka.algos.ppo_phases import PPOPhasesConfig, MLPCfg, ResidualCfg, MultiheadCfg
from mjlab_franka.config.base import EnvConfig, TrackerConfig, TrainConfig



@dataclass
class FrankaCubeRelocationBase(TrainConfig):
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"override /algo": "ppo_phases"},
            "_self_",
        ]
    )

    env: EnvConfig = field(
        default_factory=lambda: EnvConfig(
            task="franka_cube_relocation",
            num_envs=4096,
            device="cuda:0",
        )
    )

    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualize: bool = True

@dataclass
class FrankaCubeRelocationMLP(FrankaCubeRelocationBase):
    """Experiment defaults: Franka cube relocation with PPO phases."""

    name: str | None = "franka_cube_relocation_mlp_phases"
    algo: PPOPhasesConfig = field(
        default_factory=lambda: PPOPhasesConfig(
            timesteps=150_000,
            rollouts=16,
            learning_rate=3e-4,
            mini_batches=8,
            learning_epochs=4,
            entropy_loss_scale=0.005,
            num_phases=4,
            value_cfg=MLPCfg(
                value_hidden_dims=[256, 256],
            )
        )
    )


@dataclass
class FrankaCubeRelocationResidual(FrankaCubeRelocationBase):
    """Experiment defaults: Franka cube relocation with PPO phases."""

    name: str | None = "franka_cube_relocation_residual_phases"
    algo: PPOPhasesConfig = field(
        default_factory=lambda: PPOPhasesConfig(
            timesteps=150_000,
            rollouts=16,
            learning_rate=3e-4,
            mini_batches=8,
            learning_epochs=4,
            entropy_loss_scale=0.005,
            num_phases=4,
            value_cfg=ResidualCfg(
                value_hidden_dims=[256],
                base_hidden_dims=[128],
                head_hidden_dims=[64],
                residual_l2_weight=1e-4,
            )
        )
    )


@dataclass
class FrankaCubeRelocationMultihead(FrankaCubeRelocationBase):
    """Experiment defaults: Franka cube relocation with PPO phases."""

    name: str | None = "franka_cube_relocation_multihead_phases"
    algo: PPOPhasesConfig = field(
        default_factory=lambda: PPOPhasesConfig(
            timesteps=150_000,
            rollouts=16,
            learning_rate=3e-4,
            mini_batches=8,
            learning_epochs=4,
            entropy_loss_scale=0.005,
            num_phases=4,
            value_cfg=MultiheadCfg(
                value_hidden_dims=[256],
                head_hidden_dims=[64],
            )
        )
    )

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    name="franka_cube_relocation_mlp_phases",
    node=FrankaCubeRelocationMLP,
    package="_global_",
)

cs.store(
    group="experiment",
    name="franka_cube_relocation_residual_phases",
    node=FrankaCubeRelocationResidual,
    package="_global_",
)

cs.store(
    group="experiment",
    name="franka_cube_relocation_multihead_phases",
    node=FrankaCubeRelocationMultihead,
    package="_global_",
)