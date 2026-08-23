"""Hydra experiment presets for mjlab_franka."""

from mjlab_franka.config.experiments import franka_trajectory_track_ppo  # noqa: F401
from mjlab_franka.config.experiments import franka_cube_relocation_ppo_phases  # noqa: F401

def available_experiments() -> list[str]:
    return ["franka_trajectory_track_ppo", "franka_cube_relocation_ppo_phases"]
