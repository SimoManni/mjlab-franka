"""Observations specific to the manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, matrix_from_quat, quat_conjugate, quat_mul

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def object_pos_local(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position of the object expressed in the robot base (or root) frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    obj_pos = obj.data.root_pos_w   
    # Transform global position offset into robot's local frame
    pos_diff = obj_pos - robot_pos
    return quat_apply_inverse(robot_quat, pos_diff)


def object_rotation_matrix_local(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Rotation matrix (flattened or raw components) of the object in the robot base frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    robot_quat = robot.data.root_quat_w
    obj_quat = obj.data.root_quat_w

    # Compute relative rotation quaternion, then convert to rotation matrix
    # For simplicity and standard RL usage, flattening the 3x3 rotation matrix (9D) or 6D representation works well.
    # Here we use relative orientation matrix flattened to (N, 9).
    rel_quat = quat_mul(quat_conjugate(robot_quat), obj_quat)
    rot_mat = matrix_from_quat(rel_quat)
    return rot_mat.reshape(env.num_envs, 9)


def object_vel_local(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linear and angular velocity of the object in the robot base frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    robot_quat = robot.data.root_quat_w
    obj_lin_vel_w = obj.data.root_lin_vel_w
    obj_ang_vel_w = obj.data.root_ang_vel_w

    # Rotate linear and angular velocities into the local frame
    lin_vel_local = quat_apply_inverse(robot_quat, obj_lin_vel_w)
    ang_vel_local = quat_apply_inverse(robot_quat, obj_ang_vel_w)

    return torch.cat([lin_vel_local, ang_vel_local], dim=-1)


def target_2D_location(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Retrieves the target goal location on the ground plane from a command term."""
    command_term = env.command_manager.get_term(command_name)
    # Assuming command outputs layout [x, y, yaw] or [x, y]
    return command_term.command[:, :2]