"""Custom action terms for the reaching task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class QuinticSplineJointPositionActionCfg(BaseActionCfg):
    """Quintic spline action: smooth trajectory from current state to policy target.

    Every control cycle the policy outputs a desired joint position *increment*
    ``Δq`` and, optionally, a desired joint velocity ``v_f`` at the end of the
    horizon.  Given the robot's current ``(q₀, v₀, a₀)`` a 5th-order polynomial
    is solved analytically (boundary conditions: pos/vel/acc at both ends) and
    evaluated at each simulation sub-step so that the position target sent to
    the PD actuators varies smoothly — no step changes, no velocity discontinuities.

    Policy output layout (flattened, action_dim = J or 2J):
      - ``action[:J]``  — position increments  ``Δq`` (rad), scaled by ``pos_scale``
      - ``action[J:2J]`` — target velocities   ``v_f`` (rad/s), scaled by ``vel_scale``
                           (only present when ``include_velocity=True``)

    Polynomial (normalised time p = t/T ∈ [0, 1]):
      q(p) = a₀ + a₁p + a₂p² + a₃p³ + a₄p⁴ + a₅p⁵
    Coefficients:
      a₀ = q₀,  a₁ = v₀·T,  a₂ = a₀·T²/2
      Δq = q_f − a₀ − a₁ − a₂
      Δv = v_f·T − a₁ − 2a₂
      Δa = −2a₂
      a₃ = 10Δq − 4Δv + ½Δa
      a₄ = −15Δq + 7Δv − Δa
      a₅ = 6Δq − 3Δv + ½Δa
    """

    include_velocity: bool = True
    """If True the policy outputs 2J values (Δq + v_f); if False only J (Δq, v_f=0)."""

    pos_scale: float = 0.05
    """Scale applied to the position-increment outputs (rad per control step)."""

    vel_scale: float = 0.5
    """Scale applied to the velocity outputs (rad/s). Only used when
    ``include_velocity=True``."""
    # Inherited ``scale`` field from BaseActionCfg is unused; set to 1.0 to
    # avoid confusion.
    scale: float = field(default=1.0, init=False)

    def build(self, env: "ManagerBasedRlEnv") -> "QuinticSplineJointPositionAction":
        return QuinticSplineJointPositionAction(self, env)


class QuinticSplineJointPositionAction(BaseAction):
    """Executes a quintic spline between the current joint state and the policy target.

    See :class:`QuinticSplineJointPositionActionCfg` for the full description.
    """

    cfg: QuinticSplineJointPositionActionCfg

    def __init__(
        self,
        cfg: QuinticSplineJointPositionActionCfg,
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(cfg=cfg, env=env)

        N = self.num_envs
        J = self._num_targets
        self._decimation: int = env.cfg.decimation

        # Override action buffers for the 2J case.
        if cfg.include_velocity:
            self._action_dim = 2 * J
            self._raw_actions = torch.zeros(N, 2 * J, device=self.device)
            self._processed_actions = torch.zeros(N, 2 * J, device=self.device)

        # Spline coefficients (N, J, 6): [a0, a1, a2, a3, a4, a5].
        # Initialised to zero; first process_actions call fills them correctly.
        self._coeff = torch.zeros(N, J, 6, device=self.device)
        # Current simulation sub-step within the control cycle (0-indexed).
        self._substep: int = 0

        # Hard joint position limits for clamping (N, J).
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]  # (N, J, 2)
        self._joint_lower = limits[..., 0].clone()
        self._joint_upper = limits[..., 1].clone()

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        """Compute quintic spline coefficients from current robot state + policy target."""
        self._raw_actions[:] = actions

        J = self._num_targets
        T = self._env.step_dt  # control cycle duration (s)

        # --- decode policy outputs ---
        delta_q = actions[:, :J] * self.cfg.pos_scale  # (N, J) position increment
        if self.cfg.include_velocity:
            v_f = actions[:, J:] * self.cfg.vel_scale  # (N, J) target velocity
        else:
            v_f = torch.zeros(self.num_envs, J, device=self.device)

        # --- current robot state ---
        q_0 = self._entity.data.joint_pos[:, self._target_ids]  # (N, J)
        v_0 = self._entity.data.joint_vel[:, self._target_ids]  # (N, J)
        a_0 = self._entity.data.joint_acc[:, self._target_ids]  # (N, J)

        q_f = q_0 + delta_q  # absolute target position

        # --- coefficients in normalised time p = t/T ∈ [0, 1] ---
        a0 = q_0
        a1 = v_0 * T
        a2 = a_0 * (T * T) * 0.5

        dq = q_f - a0 - a1 - a2
        dv = v_f * T - a1 - 2.0 * a2
        da = -2.0 * a2

        # Analytical solution of the 3×3 boundary system.
        a3 = 10.0 * dq - 4.0 * dv + 0.5 * da
        a4 = -15.0 * dq + 7.0 * dv - da
        a5 = 6.0 * dq - 3.0 * dv + 0.5 * da

        self._coeff = torch.stack([a0, a1, a2, a3, a4, a5], dim=-1)  # (N, J, 6)
        self._substep = 0

    def apply_actions(self) -> None:
        """Evaluate the spline at the current sub-step and apply position target."""
        p = (self._substep + 1.0) / self._decimation  # normalised time in (0, 1]

        # Horner's method: a0 + p*(a1 + p*(a2 + p*(a3 + p*(a4 + p*a5))))
        c = self._coeff  # (N, J, 6)
        q_target = c[..., 0] + p * (
            c[..., 1]
            + p * (c[..., 2] + p * (c[..., 3] + p * (c[..., 4] + p * c[..., 5])))
        )  # (N, J)

        self._entity.set_joint_position_target(
            q_target.clamp(self._joint_lower, self._joint_upper),
            joint_ids=self._target_ids,
        )
        # Advance sub-step, saturate at decimation-1 to avoid index overflow if
        # apply_actions is called more times than expected.
        self._substep = min(self._substep + 1, self._decimation - 1)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            self._coeff.zero_()
        else:
            self._coeff[env_ids] = 0.0
        self._substep = 0
