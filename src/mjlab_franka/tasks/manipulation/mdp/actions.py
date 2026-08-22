"""Custom action terms for the reaching task (Arm + Gripper)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class QuinticSplineArmAndGripperActionCfg(BaseActionCfg):
    """Quintic spline action extended to control both the 7-DoF arm and the gripper.

    The policy output layout depends on ``include_velocity``:
      - If False: [delta_q_arm (7), delta_q_gripper (2)] -> Total dim = 9
      - If True:  [delta_q_arm (7), delta_q_gripper (2), v_f_arm (7), v_f_gripper (2)] -> Total dim = 18
    """

    include_velocity: bool = True
    """If True, the policy outputs target velocities alongside position increments."""

    # Arm scales
    arm_pos_scale: float = 0.05
    """Scale applied to arm position increments (rad per control step)."""
    arm_vel_scale: float = 0.5
    """Scale applied to arm target velocities (rad/s)."""

    # Gripper scales (typically smaller for fine control of fingers)
    gripper_pos_scale: float = 0.01
    """Scale applied to gripper position increments (m or rad per control step)."""
    gripper_vel_scale: float = 0.1
    """Scale applied to gripper target velocities."""

    # Inherited ``scale`` field from BaseActionCfg is unused; set to 1.0.
    scale: float = field(default=1.0, init=False)

    def build(self, env: "ManagerBasedRlEnv") -> "QuinticSplineArmAndGripperAction":
        return QuinticSplineArmAndGripperAction(self, env)


class QuinticSplineArmAndGripperAction(BaseAction):
    """Executes smooth quintic splines independently across arm joints and gripper joints."""

    cfg: QuinticSplineArmAndGripperActionCfg

    def __init__(
        self,
        cfg: QuinticSplineArmAndGripperActionCfg,
        env: "ManagerBasedRlEnv",
    ) -> None:
        super().__init__(cfg=cfg, env=env)

        N = self.num_envs
        J = self._num_targets  # Total targets (Arm joints + Gripper joints, e.g., 7 + 2 = 9)
        self._decimation: int = env.cfg.decimation

        # We assume the first 7 targets are the arm and the remaining are the gripper.
        # (Ensure your action term registration order aligns with this assumption)
        self._arm_dim = 7
        self._gripper_dim = J - self._arm_dim

        if self._gripper_dim < 0:
            raise ValueError(
                f"QuinticSplineArmAndGripperAction expected at least {self._arm_dim} targets, "
                f"but got {J}. Did you include the gripper joints in asset config?"
            )

        # Build scale tensors matched to target dimensions for vectorized broadcasting
        scales = []
        # 1. Arm position scale
        scales.extend([cfg.arm_pos_scale] * self._arm_dim)
        # 2. Gripper position scale
        scales.extend([cfg.gripper_pos_scale] * self._gripper_dim)

        if cfg.include_velocity:
            # 3. Arm velocity scale
            scales.extend([cfg.arm_vel_scale] * self._arm_dim)
            # 4. Gripper velocity scale
            scales.extend([cfg.gripper_vel_scale] * self._gripper_dim)
            self._action_dim = 2 * J
        else:
            self._action_dim = J

        self._action_scales = torch.tensor(scales, device=self.device, dtype=torch.get_default_dtype())

        # Action buffers
        self._raw_actions = torch.zeros(N, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros(N, self._action_dim, device=self.device)

        # Spline coefficients (N, J, 6): [a0, a1, a2, a3, a4, a5]
        self._coeff = torch.zeros(N, J, 6, device=self.device)
        self._substep: int = 0

        # Hard joint position limits for clamping (N, J)
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        self._joint_lower = limits[..., 0].clone()
        self._joint_upper = limits[..., 1].clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        J = self._num_targets
        T = self._env.step_dt

        # Scale raw policy outputs using our split scale vector
        scaled_actions = actions * self._action_scales

        if self.cfg.include_velocity:
            delta_q = scaled_actions[:, :J]
            v_f = scaled_actions[:, J:]
        else:
            delta_q = scaled_actions
            v_f = torch.zeros(self.num_envs, J, device=self.device)

        # Current robot state for all targeted joints (Arm + Gripper)
        q_0 = self._entity.data.joint_pos[:, self._target_ids]
        v_0 = self._entity.data.joint_vel[:, self._target_ids]
        a_0 = self._entity.data.joint_acc[:, self._target_ids]

        q_f = q_0 + delta_q

        # Quintic polynomial coefficients calculation
        a0 = q_0
        a1 = v_0 * T
        a2 = a_0 * (T * T) * 0.5

        dq = q_f - a0 - a1 - a2
        dv = v_f * T - a1 - 2.0 * a2
        da = -2.0 * a2

        a3 = 10.0 * dq - 4.0 * dv + 0.5 * da
        a4 = -15.0 * dq + 7.0 * dv - da
        a5 = 6.0 * dq - 3.0 * dv + 0.5 * da

        self._coeff = torch.stack([a0, a1, a2, a3, a4, a5], dim=-1)
        self._substep = 0

    def apply_actions(self) -> None:
        p = (self._substep + 1.0) / self._decimation

        c = self._coeff
        q_target = c[..., 0] + p * (
            c[..., 1]
            + p * (c[..., 2] + p * (c[..., 3] + p * (c[..., 4] + p * c[..., 5])))
        )

        self._entity.set_joint_position_target(
            q_target.clamp(self._joint_lower, self._joint_upper),
            joint_ids=self._target_ids,
        )
        self._substep = min(self._substep + 1, self._decimation - 1)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            self._coeff.zero_()
        else:
            self._coeff[env_ids] = 0.0
        self._substep = 0