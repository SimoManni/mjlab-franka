"""Modular, potential-based shaping rewards using distance improvement."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class DistanceImprovementReward:
    """Base stateful reward term that computes potential-based rewards

    based on the reduction of distance between simulation steps:
    Reward = prev_distance - curr_distance
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        self._num_envs = env.num_envs
        self._device = env.device
        
        # Buffer to keep track of the distance from the previous step for each environment
        self._prev_distances = torch.zeros(self._num_envs, device=self._device)
        self._initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

    def _compute_current_distances(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """To be overridden by subclasses to define what distance is being measured."""
        raise NotImplementedError

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """Computes distance improvement between the previous step and the current step."""
        curr_distances = self._compute_current_distances(env)

        # On the first step or after a reset, initialize prev_distances to current value
        uninit_mask = ~self._initialized
        if uninit_mask.any():
            self._prev_distances[uninit_mask] = curr_distances[uninit_mask]
            self._initialized[uninit_mask] = True

        # Difference: positive if we got closer, negative if we moved further away
        improvement = self._prev_distances - curr_distances

        # Update previous distance buffer for the next step
        self._prev_distances[:] = curr_distances

        return improvement

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Resets the initialization flags and buffers when environments reset."""
        if env_ids is None:
            self._initialized.zero_()
            self._prev_distances.zero_()
        else:
            self._initialized[env_ids] = False
            self._prev_distances[env_ids] = 0.0


class EntityEntityDistanceImprovement(DistanceImprovementReward):
    """Measures distance improvement between a source entity and a target entity."""

    def __init__(
        self,
        cfg,
        env: ManagerBasedRlEnv,
        source_cfg: SceneEntityCfg,
        target_cfg: SceneEntityCfg,
        distance_type: str = "l2",
        **kwargs
    ) -> None:
        super().__init__(cfg, env)
        self._source_cfg = source_cfg
        self._target_cfg = target_cfg
        self._distance_type = distance_type

    def _compute_current_distances(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        # Retrieve source entity position
        source_entity = env.scene[self._source_cfg.name]
        if self._source_cfg.site_names:
            site_id = source_entity.find_sites(self._source_cfg.site_names)[0]
            source_pos = source_entity.data.site_pos_w[..., site_id, :].squeeze(1)
        else:
            body_id = self._source_cfg.body_ids[0] if self._source_cfg.body_ids else 0
            source_pos = source_entity.data.body_pos_w[..., body_id, :].squeeze(1)

        # Retrieve target entity position
        target_entity = env.scene[self._target_cfg.name]
        if self._target_cfg.site_names:
            site_id = target_entity.find_sites(self._target_cfg.site_names)[0]
            target_pos = target_entity.data.site_pos_w[..., site_id, :].squeeze(1)
        else:
            target_pos = target_entity.data.root_link_pos_w

        if self._distance_type == "l2":
            dist = torch.norm(source_pos - target_pos, dim=-1)
        elif self._distance_type == "xy":
            dist = torch.norm(source_pos[..., :2] - target_pos[..., :2], dim=-1)
        else:
            raise ValueError(f"Unsupported distance type: {self._distance_type}")
        
        return dist.view(-1)

class EntityCommandDistanceImprovement(DistanceImprovementReward):
    """Measures distance improvement between an entity (e.g., cube) and a command target."""

    def __init__(
        self,
        cfg,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        distance_type: str = "xy",
        **kwargs
    ) -> None:
        super().__init__(cfg, env)
        self._asset_cfg = asset_cfg
        self._command_name = command_name
        self._distance_type = distance_type

    def _compute_current_distances(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        # Retrieve entity position
        entity = env.scene[self._asset_cfg.name]
        source_pos = entity.data.root_link_pos_w  # (num_envs, 3)

        # Retrieve target from command manager
        command_term = env.command_manager.get_term(self._command_name)
        target_command = command_term.command  # Expected shape (num_envs, 2) or (num_envs, 3)

        if self._distance_type == "xy":
            return torch.norm(source_pos[..., :2] - target_command[..., :2], dim=-1)
        elif self._distance_type == "l2":
            # If target command only has 2D info, pad Z with 0 or match dimensions
            target_pos_3d = torch.cat([target_command[..., :2], torch.zeros_like(target_command[..., :1])], dim=-1)
            return torch.norm(source_pos[..., :3] - target_pos_3d, dim=-1)
        else:
            raise ValueError(f"Unsupported distance type: {self._distance_type}")



def finger_object_contact_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Returns a binary or continuous tensor indicating whether the gripper

    fingers are currently in contact with the object, based on a contact sensor.
    """
    contact_sensor = env.scene.sensors[sensor_name]
    # contact_sensor.data.found usually contains a boolean or indicator tensor of shape (num_envs, num_slots)
    contact_data = contact_sensor.data.found
    
    # Reduce across contact slots to get a per-environment binary flag (1.0 if any contact, 0.0 otherwise)
    if contact_data.ndim > 1:
        has_contact = torch.any(contact_data > 0, dim=-1).float()
    else:
        has_contact = (contact_data > 0).float()
        
    return has_contact


def termination_triggered_reward(env: ManagerBasedRlEnv, termination_name: str) -> torch.Tensor:
    """Reward or penalize based on whether a specific termination condition was triggered."""
    # Check if the termination term exists in the manager's active flags or dones buffer
    if hasattr(env.termination_manager, "term_dones") and termination_name in env.termination_manager.term_dones:
        dones = env.termination_manager.term_dones[termination_name]
    elif hasattr(env.termination_manager, "_term_dones") and termination_name in env.termination_manager._term_dones:
        dones = env.termination_manager._term_dones[termination_name]
    else:
        # Fallback to checking the active termination flags buffer directly
        dones = env.termination_manager.get_term(termination_name)

    return dones.float()