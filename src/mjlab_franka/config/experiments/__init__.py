"""Hydra experiment presets for mjlab_franka."""

from mjlab_franka.config.experiments import franka_trajectory_track_ppo  # noqa: F401


def available_experiments() -> list[str]:
    return ["franka_trajectory_track_ppo"]
