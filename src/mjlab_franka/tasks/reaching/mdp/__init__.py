"""Reaching task MDP terms (commands, observations, rewards)."""

from mjlab_franka.tasks.reaching.mdp.actions import (
    QuinticSplineJointPositionAction,
    QuinticSplineJointPositionActionCfg,
)
from mjlab_franka.tasks.reaching.mdp.commands import (
    TrajectoryCommand,
    TrajectoryCommandCfg,
)
from mjlab_franka.tasks.reaching.mdp.metrics import ee_position_l2_metric
from mjlab_franka.tasks.reaching.mdp.observations import (
    ee_target_offset_w,
    ee_target_velocity_w,
    ee_velocity_w,
    ee_z_axis_w,
    trajectory_plane_normal_w,
)
from mjlab_franka.tasks.reaching.mdp.rewards import (
    ee_plane_perpendicular,
    ee_position_l2,
    ee_position_tracking,
    ee_velocity_tracking,
    wrist_position_tracking,
)

__all__ = [
    "QuinticSplineJointPositionAction",
    "QuinticSplineJointPositionActionCfg",
    "TrajectoryCommand",
    "TrajectoryCommandCfg",
    "ee_plane_perpendicular",
    "ee_position_l2",
    "ee_position_l2_metric",
    "ee_position_tracking",
    "ee_target_offset_w",
    "ee_target_velocity_w",
    "ee_velocity_tracking",
    "ee_velocity_w",
    "ee_z_axis_w",
    "trajectory_plane_normal_w",
    "wrist_position_tracking",
]
