"""Rewards specific to end-effector trajectory tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_franka.tasks.reaching.mdp.observations import _resolve_site_id

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def ee_position_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.1,
) -> torch.Tensor:
    """Exponential reward on EE position tracking error."""
    robot = env.scene[asset_cfg.name]
    site_id = _resolve_site_id(env, asset_cfg)
    ee_pos = robot.data.site_pos_w[:, site_id]
    cmd = env.command_manager.get_command(command_name)
    err = torch.norm(cmd[:, :3] - ee_pos, dim=-1)
    return torch.exp(-((err / std) ** 2))


def ee_velocity_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.5,
) -> torch.Tensor:
    """Exponential reward on EE linear-velocity tracking error."""
    robot = env.scene[asset_cfg.name]
    site_id = _resolve_site_id(env, asset_cfg)
    ee_vel = robot.data.site_vel_w[:, site_id, 3:6]
    cmd = env.command_manager.get_command(command_name)
    err = torch.norm(cmd[:, 3:6] - ee_vel, dim=-1)
    return torch.exp(-((err / std) ** 2))


def ee_position_l2(
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Negative L2 penalty on EE position tracking error (used with negative weight)."""
    robot = env.scene[asset_cfg.name]
    site_id = _resolve_site_id(env, asset_cfg)
    ee_pos = robot.data.site_pos_w[:, site_id]
    cmd = env.command_manager.get_command(command_name)
    return torch.sum(torch.square(cmd[:, :3] - ee_pos), dim=-1)


def ee_plane_perpendicular(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float = 0.1,
) -> torch.Tensor:
    """Exponential reward on alignment between the EE z-axis and the trajectory plane normal.

    Computes ``exp(-((1 - dot(ee_z_w, plane_normal_w)) / std) ** 2)``, where 1 means the
    EE z-axis (approach direction) is perfectly aligned with the plane normal
    and 0 means it lies flat in the trajectory plane.

    Both vectors are obtained from :class:`TrajectoryCommand` attributes.
    """
    from mjlab_franka.tasks.reaching.mdp.commands import TrajectoryCommand

    cmd_term: TrajectoryCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    similarity = torch.sum(cmd_term.ee_z_w * cmd_term.plane_normal_w, dim=-1)
    err = 1.0 - similarity
    return torch.exp(-((err / std) ** 2))


def wrist_position_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.05,
) -> torch.Tensor:
    """Exponential reward on wrist (link7) position tracking error.

    The wrist target is ``target_pos_w - plane_normal_w * wrist_offset``,
    where ``wrist_offset`` is set in :class:`TrajectoryCommandCfg`.
    This drives the wrist body to sit directly behind the EE along the
    trajectory plane normal.
    """
    from mjlab_franka.tasks.reaching.mdp.commands import TrajectoryCommand

    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, list) and len(body_ids) > 0:
        body_id = int(body_ids[0])
    else:
        names = asset_cfg.body_names
        if isinstance(names, str):
            names = [names]
        resolved, _ = robot.find_bodies(list(names))
        body_id = int(resolved[0])
    wrist_pos = robot.data.body_link_pos_w[:, body_id]
    cmd_term: TrajectoryCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    err = torch.norm(cmd_term.wrist_target_pos_w - wrist_pos, dim=-1)
    return torch.exp(-((err / std) ** 2))
