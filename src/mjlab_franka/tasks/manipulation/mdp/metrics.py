"""Metrics specific to the manipulation tasks for performance evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def obj_to_target_distance(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Computes the 2D distance between the object and the goal target location on the ground."""
    obj = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_link_pos_w  # (num_envs, 3)

    command_term = env.command_manager.get_term(command_name)
    target_pos = command_term.command  # (num_envs, 2)

    return torch.norm(obj_pos_w[:, :2] - target_pos[:, :2], dim=-1)


def gripper_to_obj_distance(
    env: ManagerBasedRlEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Computes the 3D L2 distance between the robot's end-effector/gripper and the object."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Retrieve end-effector site position or fallback to body position
    if robot_cfg.site_names:
        site_id = robot.find_sites(robot_cfg.site_names)[0]
        ee_pos = robot.data.site_pos_w[..., site_id, :].squeeze(1)
    else:
        body_id = robot_cfg.body_ids[0] if robot_cfg.body_ids else 0
        ee_pos = robot.data.body_pos_w[..., body_id, :].squeeze(1)

    obj_pos = obj.data.root_link_pos_w

    return torch.norm(ee_pos - obj_pos, dim=-1).view(-1)