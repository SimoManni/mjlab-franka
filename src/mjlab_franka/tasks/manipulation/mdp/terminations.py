"""Termination conditions specific to the manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def obj_at_goal(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    pos_threshold: float = 0.05,
) -> torch.Tensor:
    """Terminates (succeeds) when the object's 2D position on the ground is within a threshold

    of the command target location.
    """
    # Retrieve object position in the world frame
    cube = env.scene[object_cfg.name]
    cube_pos_w = cube.data.root_link_pos_w  # (num_envs, 3)

    # Retrieve the 2D command target location from the command manager
    command_term = env.command_manager.get_term(command_name)
    target_pos = command_term.command  # (num_envs, 2)

    # Compute Euclidean distance in the 2D (x, y) ground plane
    distance = torch.norm(cube_pos_w[:, :2] - target_pos[:, :2], dim=-1)

    # Return a boolean tensor indicating whether each environment meets the threshold condition
    return distance < pos_threshold