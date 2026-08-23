import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_franka.tasks.manipulation.phase.base_phase import PhaseCommandTerm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

class FrankaCubePhaseCommand(PhaseCommandTerm):
    """Manages the 4 sequential task phases using configuration aliases."""

    def __init__(self, cfg: "FrankaCubePhaseCommandCfg", env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        self.env = env
        self.cfg: FrankaCubePhaseCommandCfg = cfg

        self.APPROACHING = cfg.APPROACHING
        self.GRIPPING = cfg.GRIPPING
        self.MOVING = cfg.MOVING
        self.OFFLOADING = cfg.OFFLOADING

    def _compute_phases(self, env_ids: torch.Tensor) -> torch.Tensor:
        robot = self.env.scene[self.cfg.robot_cfg.name]
        cube = self.env.scene[self.cfg.cube_cfg.name]
        target_term = self.env.command_manager.get_term(self.cfg.target_command_name)
        target_pos = target_term.command[env_ids]

        if self.cfg.robot_cfg.site_names:
            site_id = robot.find_sites(self.cfg.robot_cfg.site_names)[0]
            ee_pos = robot.data.site_xpos[env_ids, site_id, :]
        else:
            body_id = self.cfg.robot_cfg.body_ids[0] if self.cfg.robot_cfg.body_ids else 0
            ee_pos = robot.data.body_pos_w[env_ids, body_id, :]

        cube_pos = cube.data.root_pos_w[env_ids]

        ee_to_cube_dist = torch.norm(ee_pos - cube_pos, dim=-1)
        cube_height = cube_pos[:, 2]
        cube_to_goal_dist = torch.norm(cube_pos[:, :2] - target_pos[:, :2], dim=-1)

        current = self._current_phases[env_ids]
        next_phase = current.clone()

        # Phase transitions with hysteresis
        mask_p0 = (current == self.APPROACHING) & (ee_to_cube_dist < self.cfg.reach_threshold)
        next_phase[mask_p0] = self.GRIPPING

        mask_p1_advance = (current == self.GRIPPING) & (cube_height > self.cfg.lift_threshold)
        mask_p1_regress = (current == self.GRIPPING) & (ee_to_cube_dist > self.cfg.reach_hysteresis)
        next_phase[mask_p1_advance] = self.MOVING
        next_phase[mask_p1_regress] = self.APPROACHING

        mask_p2_advance = (current == self.MOVING) & (cube_to_goal_dist < self.cfg.goal_threshold)
        mask_p2_regress = (current == self.MOVING) & (cube_height < self.cfg.lift_hysteresis)
        next_phase[mask_p2_advance] = self.OFFLOADING
        next_phase[mask_p2_regress] = self.GRIPPING

        mask_p3_regress = (current == self.OFFLOADING) & (cube_to_goal_dist > self.cfg.goal_hysteresis)
        next_phase[mask_p3_regress] = self.MOVING
        return next_phase


@dataclass
class FrankaCubePhaseCommandCfg(CommandTermCfg):
    """Configuration for Franka cube task phases."""

    # Phase aliases
    APPROACHING: int = 0
    GRIPPING: int = 1
    MOVING: int = 2
    OFFLOADING: int = 3


    num_phases: int = 4
    robot_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("franka"))
    cube_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("cube"))
    target_command_name: str = "target_2D_location"
    
    # Hysteresis thresholds (Forward threshold to advance, backward to regress)
    reach_threshold: float = 0.08      # Advance to gripping when EE is < 0.08m
    reach_hysteresis: float = 0.12     # Fall back if EE > 0.12m
    
    lift_threshold: float = 0.04       # Advance to moving when cube z > 0.04m
    lift_hysteresis: float = 0.025     # Fall back if cube z < 0.025m
    
    goal_threshold: float = 0.05       # Advance to offloading when cube to goal < 0.05m
    goal_hysteresis: float = 0.08      # Fall back if distance > 0.08m

    def build(self, env):
        return FrankaCubePhaseCommand(self, env)