"""User-defined metric terms for the trajectory-tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_franka.tasks.reaching.mdp.observations import _resolve_site_id

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def ee_position_l2_metric(
    env: "ManagerBasedRlEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 distance (meters) between EE site and the current trajectory target."""
    robot = env.scene[asset_cfg.name]
    site_id = _resolve_site_id(env, asset_cfg)
    ee_pos = robot.data.site_pos_w[:, site_id]
    target = env.command_manager.get_command(command_name)[:, :3]
    return torch.linalg.norm(target - ee_pos, dim=-1)
