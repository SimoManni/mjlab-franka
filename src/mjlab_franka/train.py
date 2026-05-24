"""Training entry point.

Usage:
    uv run train experiment=franka_trajectory_track_ppo
    uv run train experiment=franka_trajectory_track_ppo env.num_envs=64 algo.timesteps=100
"""

from __future__ import annotations

# Must come first: sets PYTORCH_JIT=0 to avoid torch.jit.script segfault in mjlab.
import mjlab_franka._compat  # noqa: F401

import logging

import hydra

from mjlab_franka.algos import algo_registry
from mjlab_franka.config.base import TrainConfig
from mjlab_franka.config.experiments import available_experiments
from mjlab_franka.envs.factory import make_env

# Imported for its task registrations.
import mjlab_franka.tasks  # noqa: F401

log = logging.getLogger(__name__)


@hydra.main(config_path=None, config_name="train", version_base=None)
def main(train_cfg: TrainConfig) -> None:
    logging.getLogger("mjlab_franka").setLevel(getattr(logging, train_cfg.log_level))

    train_fn = algo_registry[train_cfg.algo.algo_type].train

    log.info(f"Algo: {train_cfg.algo.algo_type}")
    log.info(f"Task: {train_cfg.env.task}")
    log.info(f"Experiment: {train_cfg.name}")
    log.info(f"Available experiments: {available_experiments()}")

    env = make_env(
        task=train_cfg.env.task,
        num_envs=train_cfg.env.num_envs,
        device=train_cfg.env.device,
        seed=train_cfg.env.seed,
        play=train_cfg.env.play,
    )

    viz = None
    if train_cfg.visualize:
        from mjlab_franka.core.training_visualizer import (
            RecordingEnvWrapper,
            TrainingVisualizer,
        )

        viz = TrainingVisualizer(env)
        env = RecordingEnvWrapper(env, viz)  # type: ignore[assignment]

    try:
        train_fn(train_cfg, env)
    finally:
        if viz is not None:
            viz.stop()


if __name__ == "__main__":
    main()
