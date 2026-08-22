"""Reset events specific to the manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from mjlab.envs.mdp.events import reset_root_state_uniform, reset_joints_by_offset 

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def reset_robot_obj_scene(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_pos_range: dict[str, tuple[float, float]],
    robot_joint_pos_range: tuple[float, float] = (-0.5, 0.5),
    robot_joint_vel_range: tuple[float, float] = (-0.5, 0.5),
) -> None:
    """Resets the robot base, robot joints, and the randomized object position on the ground.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset. If None, resets all environments.
        robot_cfg: Scene configuration for the Franka robot.
        object_cfg: Scene configuration for the object entity.
        object_pos_range: Uniform ranges for the object's initial position, e.g., 
                        {"x": (0.4, 0.6), "y": (-0.2, 0.2), "z": (0.02, 0.02)}.
        robot_joint_pos_range: Uniform offset range for robot joint positions.
        robot_joint_vel_range: Uniform offset range for robot joint velocities.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    # 1. Reset Robot Base / Root Pose
    reset_root_state_uniform(
        env,
        env_ids,
        asset_cfg=robot_cfg,
    )

    # 2. Reset Robot Joints (Offsets from default configuration)
    reset_joints_by_offset(
        env,
        env_ids,
        position_range=robot_joint_pos_range,
        velocity_range=robot_joint_vel_range,
        asset_cfg=robot_cfg,
    )

    # 3. Reset Object Position on the Ground
    obj: Entity = env.scene[object_cfg.name]
    default_obj_state = obj.data.default_root_state
    assert default_obj_state is not None
    
    obj_states = default_obj_state[env_ids].clone()

    # Sample uniformly within specified coordinate bounds
    range_list = [
        object_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=env.device)
    obj_pose_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device
    )

    # Apply position offsets and add environment grid origins
    positions = obj_states[:, 0:3] + obj_pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
    
    # Keep default orientation or apply small randomized orientation changes if desired
    orientations = obj_states[:, 3:7]

    obj.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    # Zero out velocities for the object upon reset
    obj.write_root_link_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )