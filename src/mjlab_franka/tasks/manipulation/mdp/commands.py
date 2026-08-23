"""Command terms for generating 2D target locations on the ground."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class Target2DGroundCommandCfg(CommandTermCfg):
    """Configuration for generating 2D ground target positions."""

    x_range: tuple[float, float] = (0.3, 0.6)
    """Valid range for the target X position in the world frame (meters)."""
    y_range: tuple[float, float] = (-0.3, 0.3)
    """Valid range for the target Y position in the world frame (meters)."""

    def build(self, env: ManagerBasedRlEnv) -> Target2DGroundCommand:
        return Target2DGroundCommand(self, env)


class Target2DGroundCommand(CommandTerm):
    """Command term that samples 2D target positions on the ground plane."""

    cfg: Target2DGroundCommandCfg

    def __init__(self, cfg: Target2DGroundCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg=cfg, env=env)

        # Buffer holding current 2D targets for each environment: shape (num_envs, 2)
        self._location = torch.zeros(self.num_envs, 2, device=self.device)

        # Debug visualization attributes if needed
        self._metrics = {}

    def _update_command(self) -> None:
        # Commands stay static during an episode unless manually resampled,
        # handled by reset for specific environments.
        pass

    def _resample(self, env_ids: torch.Tensor) -> None:
        """Resample 2D target positions for specified environments."""
        n = len(env_ids)
        
        # Sample uniformly within the specified bounds
        x = torch.empty(n, device=self.device).uniform_(*self.cfg.x_range)
        y = torch.empty(n, device=self.device).uniform_(*self.cfg.y_range)

        self._location[env_ids, 0] = x
        self._location[env_ids, 1] = y

    @property
    def command(self) -> torch.Tensor:
        """Return the current 2D target positions for all environments."""
        return self._location

    def _update_metrics(self) -> None:
        return {}