"""Phase-based command terms for managing task progression and reward masking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PhaseCommandTerm(CommandTerm):
    """Base class for a phase-based command term.

    Keeps track of a discrete phase integer for each environment and outputs
    a one-hot encoded tensor representing the current phase as the command.
    """

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv) -> None:
        self.cfg = cfg
        self._num_envs = env.num_envs
        self._device = env.device

        # Number of total discrete phases
        self.num_phases = cfg.num_phases

        # Tensor tracking the current phase index per environment
        self._current_phases = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        
        # Command tensor (one-hot encoding of the current phase)
        self._one_hot = torch.zeros((self._num_envs, self.num_phases), dtype=torch.float32, device=self._device)
        self._update_one_hot()

        self.metrics = {}

    @property
    def command(self) -> torch.Tensor:
        """Returns the current one-hot encoded phase command for all environments."""
        return self._one_hot
        
    def _update_one_hot(self) -> None:
        """Updates the one-hot command buffer based on current phases."""
        self._one_hot.zero_()
        self._one_hot.scatter_(1, self._current_phases.unsqueeze(1), 1.0)

    def _compute_phases(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses to determine phase transitions."""
        raise NotImplementedError

    def compute(self, dt: float) -> None:
        """Called every simulation step to update phases."""
        # By default, compute for all environments
        env_ids = torch.arange(self._num_envs, device=self._device)
        self._current_phases[env_ids] = self._compute_phases(env_ids)
        self._update_one_hot()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Resets phases to 0 (initial phase) upon environment reset."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self._current_phases[env_ids] = 0
        self._update_one_hot()
        return {}

    def is_in_phases(self, active_phases: Sequence[int]) -> torch.Tensor:
        """Checks whether the current phase of each environment matches any in the active list.

        Useful for masking rewards dynamically.
        
        Returns:
            A boolean tensor of shape (num_envs,)
        """
        phases_tensor = torch.tensor(active_phases, dtype=torch.long, device=self._device)
        # Check matching across environments
        return torch.isin(self._current_phases, phases_tensor)

    def _update_metrics(self) -> None:
        return {}