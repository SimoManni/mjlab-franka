"""Observations specific to the trajectory-tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _resolve_site_id(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> int:
    """Return the entity-local id of the first site referenced by ``asset_cfg``."""
    robot = env.scene[asset_cfg.name]
    site_ids = asset_cfg.site_ids
    if isinstance(site_ids, list) and len(site_ids) > 0:
        return int(site_ids[0])
    names = asset_cfg.site_names
    if names is None:
        raise ValueError(
            f"asset_cfg for entity '{asset_cfg.name}' must specify site_names"
        )
    if isinstance(names, str):
        names = [names]
    resolved, _ = robot.find_sites(list(names))
    return int(resolved[0])


def ee_target_offset_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    time_shift: float = 0.0,
) -> torch.Tensor:
    """Vector from EE to target, in world frame. Shape (num_envs, 3).

    When ``time_shift > 0`` the target position is evaluated ``time_shift``
    seconds ahead of the current step via :meth:`TrajectoryCommand.trajectory_at`.
    """
    robot = env.scene[asset_cfg.name]
    site_id = _resolve_site_id(env, asset_cfg)
    ee_pos = robot.data.site_pos_w[:, site_id]
    if time_shift == 0.0:
        cmd = env.command_manager.get_command(command_name)
        target = cmd[:, :3]
    else:
        from mjlab_franka.tasks.reaching.mdp.commands import TrajectoryCommand

        cmd_term: TrajectoryCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
        target, _ = cmd_term.trajectory_at(time_shift)
    return target - ee_pos


def ee_velocity_w(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """End-effector world-frame linear velocity (3) for the first site."""
    robot = env.scene[asset_cfg.name]
    site_id = _resolve_site_id(env, asset_cfg)
    # site_vel_w is (num_envs, num_sites, 6) — linear vel are last 3.
    return robot.data.site_vel_w[:, site_id, 3:6]


def ee_target_velocity_w(
    env: "ManagerBasedRlEnv",
    command_name: str,
    time_shift: float = 0.0,
) -> torch.Tensor:
    """Target EE linear velocity in world frame. Shape (num_envs, 3).

    When ``time_shift > 0`` the velocity is evaluated ``time_shift`` seconds
    ahead via :meth:`TrajectoryCommand.trajectory_at`.
    """
    if time_shift == 0.0:
        cmd = env.command_manager.get_command(command_name)
        return cmd[:, 3:6]
    from mjlab_franka.tasks.reaching.mdp.commands import TrajectoryCommand

    cmd_term: TrajectoryCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    _, vel_w = cmd_term.trajectory_at(time_shift)
    return vel_w


def ee_z_axis_w(
    env: "ManagerBasedRlEnv",
    command_name: str,
) -> torch.Tensor:
    """EE approach direction (z-axis of the site frame) in world frame. Shape (num_envs, 3).

    Delegates to :attr:`TrajectoryCommand.ee_z_w`.
    """
    from mjlab_franka.tasks.reaching.mdp.commands import TrajectoryCommand

    cmd_term: TrajectoryCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    return cmd_term.ee_z_w


def trajectory_plane_normal_w(
    env: "ManagerBasedRlEnv",
    command_name: str,
) -> torch.Tensor:
    """Normal of the active trajectory plane in world frame. Shape (num_envs, 3).

    Delegates to :attr:`TrajectoryCommand.plane_normal_w`.
    """
    from mjlab_franka.tasks.reaching.mdp.commands import TrajectoryCommand

    cmd_term: TrajectoryCommand = env.command_manager.get_term(command_name)  # type: ignore[assignment]
    return cmd_term.plane_normal_w
