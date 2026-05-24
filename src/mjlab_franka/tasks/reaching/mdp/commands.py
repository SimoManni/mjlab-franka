"""Parametric end-effector trajectory command.

Each environment samples a target trajectory (shape, size, plane orientation,
period) at every resample. At every step the command exposes the current target
position and target linear velocity in the world frame, which downstream
observations / rewards consume.

Supported shapes (string identifiers, sampled uniformly from
``cfg.shapes`` per env):

- ``line``      — segment that goes back and forth (triangle wave)
- ``circle``    — closed loop
- ``square``    — closed loop along 4 edges
- ``figure_8``  — Lissajous (1:2) figure-8
- ``sinusoid``  — segment with vertical sine modulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


# Shape identifiers and their integer codes for vectorised dispatch.
SHAPE_TO_ID: dict[str, int] = {
    "line": 0,
    "circle": 1,
    "square": 2,
    "figure_8": 3,
    "sinusoid": 4,
}
ID_TO_SHAPE: dict[int, str] = {v: k for k, v in SHAPE_TO_ID.items()}


def _sample_random_rotation(n: int, device: torch.device | str) -> torch.Tensor:
    """Sample (n, 3, 3) uniformly random rotation matrices (Shoemake).

    Uses the QR decomposition of a random Gaussian matrix, which gives a
    uniform distribution on SO(3).
    """
    g = torch.randn(n, 3, 3, device=device)
    q, r = torch.linalg.qr(g)
    # Ensure proper rotation (det = +1) by flipping a column if needed.
    diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag_sign[diag_sign == 0] = 1.0
    q = q * diag_sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    flip = (det < 0).float().unsqueeze(-1).unsqueeze(-1)
    # Flip last column where det == -1.
    q = q * (
        1.0 - 2.0 * flip * torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)
    )
    return q


def _shape_xy(
    shape_id: torch.Tensor,  # (n,) int
    phase: torch.Tensor,  # (n,) in [0, 1)
    size: torch.Tensor,  # (n,)
) -> torch.Tensor:
    """Return (n, 2) parametric coordinates in the local plane for each env."""
    n = shape_id.shape[0]
    out = torch.zeros(n, 2, device=shape_id.device)

    two_pi = 2.0 * torch.pi

    # line: triangle wave on x, y=0 — covers [-size, +size].
    is_line = shape_id == SHAPE_TO_ID["line"]
    if is_line.any():
        s = phase[is_line]
        tri = 4.0 * torch.abs(s - 0.5) - 1.0  # in [-1, 1]
        out[is_line, 0] = size[is_line] * tri
        out[is_line, 1] = 0.0

    # circle
    is_circle = shape_id == SHAPE_TO_ID["circle"]
    if is_circle.any():
        a = two_pi * phase[is_circle]
        out[is_circle, 0] = size[is_circle] * torch.cos(a)
        out[is_circle, 1] = size[is_circle] * torch.sin(a)

    # square: 4 edges of length 2*size, centered at origin.
    is_square = shape_id == SHAPE_TO_ID["square"]
    if is_square.any():
        sub = phase[is_square] * 4.0
        edge = sub.long().clamp(max=3)
        frac = sub - edge.float()
        sz = size[is_square]
        x = torch.empty_like(sz)
        y = torch.empty_like(sz)
        # edge 0: bottom, (-sz -> +sz, -sz)
        m0 = edge == 0
        x = torch.where(m0, -sz + 2.0 * sz * frac, x)
        y = torch.where(m0, -sz, y)
        # edge 1: right, (+sz, -sz -> +sz)
        m1 = edge == 1
        x = torch.where(m1, sz, x)
        y = torch.where(m1, -sz + 2.0 * sz * frac, y)
        # edge 2: top, (+sz -> -sz, +sz)
        m2 = edge == 2
        x = torch.where(m2, sz - 2.0 * sz * frac, x)
        y = torch.where(m2, sz, y)
        # edge 3: left, (-sz, +sz -> -sz)
        m3 = edge == 3
        x = torch.where(m3, -sz, x)
        y = torch.where(m3, sz - 2.0 * sz * frac, y)
        out[is_square, 0] = x
        out[is_square, 1] = y

    # figure-8 (Lissajous 1:2)
    is_eight = shape_id == SHAPE_TO_ID["figure_8"]
    if is_eight.any():
        a = two_pi * phase[is_eight]
        out[is_eight, 0] = size[is_eight] * torch.sin(a)
        out[is_eight, 1] = 0.5 * size[is_eight] * torch.sin(2.0 * a)

    # sinusoid: segment with vertical sine modulation.
    is_sin = shape_id == SHAPE_TO_ID["sinusoid"]
    if is_sin.any():
        s = phase[is_sin]
        tri = 4.0 * torch.abs(s - 0.5) - 1.0  # in [-1, 1]
        out[is_sin, 0] = size[is_sin] * tri
        out[is_sin, 1] = 0.5 * size[is_sin] * torch.sin(2.0 * torch.pi * 2.0 * s)

    return out


@dataclass(kw_only=True)
class TrajectoryCommandCfg(CommandTermCfg):
    """Configuration for :class:`TrajectoryCommand`."""

    entity_name: str
    """Name of the robot entity holding the end-effector site."""

    ee_site_name: str
    """Site to track (must exist on the entity)."""

    shapes: tuple[str, ...] = ("line", "circle", "square", "figure_8", "sinusoid")
    """Allowed trajectory shapes; one is sampled per env per resample."""

    @dataclass
    class CenterRange:
        x: tuple[float, float] = (0.35, 0.60)
        y: tuple[float, float] = (-0.25, 0.25)
        z: tuple[float, float] = (0.25, 0.60)

    center_range: CenterRange = field(default_factory=CenterRange)
    """Cuboid in the robot base frame from which the trajectory center is sampled."""

    size_range: tuple[float, float] = (0.08, 0.3)
    """Range from which the trajectory characteristic size (radius / half-side) is sampled."""

    period_range: tuple[float, float] = (10.0, 14.0)
    """Range from which the trajectory period (seconds for one full traversal) is sampled."""

    random_orientation: bool = True
    """If True, the plane of the trajectory is rotated by a uniform random SO(3) matrix.
    If False, the plane is horizontal (z = center_z)."""

    wrist_body_name: str = "link7"
    """Body name of the wrist link (one link behind the EE site)."""

    wrist_offset: float = 0.107
    """Distance from the wrist body origin to the EE site along the approach direction (m).
    Defaults to the standard Franka attachment-site offset."""

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (0.9, 0.2, 0.2, 0.9)
        path_color: tuple[float, float, float, float] = (0.2, 0.6, 0.9, 0.6)
        ee_color: tuple[float, float, float, float] = (0.2, 0.9, 0.2, 0.9)
        ee_z_color: tuple[float, float, float, float] = (0.1, 0.9, 0.5, 0.9)
        plane_normal_color: tuple[float, float, float, float] = (0.95, 0.55, 0.1, 0.9)
        path_radius: float = 0.008
        target_radius: float = 0.025
        ee_radius: float = 0.025
        arrow_length: float = 0.15
        num_path_samples: int = 64
        """Number of segments used to draw the trajectory polyline."""

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: "ManagerBasedRlEnv") -> "TrajectoryCommand":
        return TrajectoryCommand(self, env)

    def __post_init__(self) -> None:
        for s in self.shapes:
            if s not in SHAPE_TO_ID:
                raise ValueError(
                    f"Unknown trajectory shape '{s}'. Supported: {sorted(SHAPE_TO_ID)}"
                )


class TrajectoryCommand(CommandTerm):
    """End-effector trajectory command term."""

    cfg: TrajectoryCommandCfg

    def __init__(self, cfg: TrajectoryCommandCfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        # Resolve the EE site on the entity directly via find_sites.
        site_ids, _ = self.robot.find_sites([cfg.ee_site_name])
        self._ee_site_local_id = int(site_ids[0])

        # Resolve the wrist body (one link behind the EE site).
        wrist_ids, _ = self.robot.find_bodies([cfg.wrist_body_name])
        self._wrist_body_id = int(wrist_ids[0])

        N = self.num_envs

        # Per-env trajectory parameters.
        self._shape_id = torch.zeros(N, device=self.device, dtype=torch.long)
        self._center = torch.zeros(N, 3, device=self.device)
        self._rot = torch.eye(3, device=self.device).expand(N, 3, 3).clone()
        self._size = torch.zeros(N, device=self.device)
        self._period = torch.ones(N, device=self.device)
        self._phase_offset = torch.zeros(N, device=self.device)
        self._elapsed = torch.zeros(N, device=self.device)

        # Outputs.
        self._target_pos_w = torch.zeros(N, 3, device=self.device)
        self._target_vel_w = torch.zeros(N, 3, device=self.device)

        # Allowed shape codes as a tensor for indexed sampling.
        self._allowed_ids = torch.tensor(
            [SHAPE_TO_ID[s] for s in cfg.shapes], device=self.device, dtype=torch.long
        )

        self.metrics["ee_pos_error"] = torch.zeros(N, device=self.device)

    # CommandTerm API ----------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 6) — target position (3) + target linear velocity (3) in world frame."""
        return torch.cat([self._target_pos_w, self._target_vel_w], dim=-1)

    @property
    def target_pos_w(self) -> torch.Tensor:
        return self._target_pos_w

    @property
    def target_vel_w(self) -> torch.Tensor:
        return self._target_vel_w

    @property
    def ee_z_w(self) -> torch.Tensor:
        """EE approach direction (z-axis of the site frame) in world frame. Shape (num_envs, 3).

        Derived analytically from the site quaternion in (w, x, y, z) convention.
        """
        q = self.robot.data.site_quat_w[:, self._ee_site_local_id]  # (N, 4)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dim=-1,
        )  # (N, 3)

    @property
    def plane_normal_w(self) -> torch.Tensor:
        """Trajectory plane normal (z-axis of the trajectory frame) in world frame. Shape (num_envs, 3).

        Equal to the third column of the per-env rotation matrix ``_rot``.
        """
        return self._rot[:, :, 2]  # (N, 3)

    @property
    def wrist_target_pos_w(self) -> torch.Tensor:
        """Target position for the wrist body in world frame. Shape (num_envs, 3).

        Computed as ``target_pos_w - plane_normal_w * wrist_offset``, i.e. the
        trajectory target shifted back along the (negative) plane normal by the
        distance from the wrist body origin to the EE site.
        """
        return self._target_pos_w - self.plane_normal_w * self.cfg.wrist_offset

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = int(env_ids.numel())
        if n == 0:
            return

        idx = torch.randint(0, self._allowed_ids.numel(), (n,), device=self.device)
        self._shape_id[env_ids] = self._allowed_ids[idx]

        r = self.cfg.center_range
        lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
        upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
        local_center = lower + (upper - lower) * torch.rand(n, 3, device=self.device)
        # Add per-env world origin so trajectory follows env spacing.
        self._center[env_ids] = local_center + self._env.scene.env_origins[env_ids]

        if self.cfg.random_orientation:
            self._rot[env_ids] = _sample_random_rotation(n, self.device)
        else:
            self._rot[env_ids] = torch.eye(3, device=self.device).expand(n, 3, 3)

        # Ensure the plane normal (third column of rot) points away from the
        # robot base.  We compare against the vector from env_origin to the
        # trajectory center: if the dot-product is negative the normal faces
        # the wrong way.  Flipping columns 0 and 2 simultaneously keeps
        # det = +1 (proper rotation).
        base_to_center = (
            self._center[env_ids] - self._env.scene.env_origins[env_ids]
        )  # (n, 3)
        normal = self._rot[env_ids, :, 2]  # (n, 3)
        needs_flip = torch.sum(normal * base_to_center, dim=-1) < 0.0  # (n,)
        if needs_flip.any():
            rot = self._rot[env_ids].clone()
            rot[needs_flip, :, 0] = -rot[needs_flip, :, 0]
            rot[needs_flip, :, 2] = -rot[needs_flip, :, 2]
            self._rot[env_ids] = rot

        size_lo, size_hi = self.cfg.size_range
        self._size[env_ids] = size_lo + (size_hi - size_lo) * torch.rand(
            n, device=self.device
        )

        per_lo, per_hi = self.cfg.period_range
        self._period[env_ids] = per_lo + (per_hi - per_lo) * torch.rand(
            n, device=self.device
        )
        self._phase_offset[env_ids] = torch.rand(n, device=self.device)
        self._elapsed[env_ids] = 0.0

    def _update_command(self) -> None:
        # Advance time per env by the env step dt.
        self._elapsed = self._elapsed + self._env.step_dt
        phase = ((self._elapsed / self._period) + self._phase_offset) % 1.0

        xy = _shape_xy(self._shape_id, phase, self._size)  # (N, 2)
        local = torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=-1)  # (N, 3)
        # rot is (N, 3, 3); world = center + rot @ local
        world = self._center + torch.einsum("nij,nj->ni", self._rot, local)
        # Finite-difference velocity.
        self._target_vel_w = (world - self._target_pos_w) / max(self._env.step_dt, 1e-6)
        self._target_pos_w = world

    def trajectory_at(self, time_shift: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return trajectory position and velocity at ``current_time + time_shift``.

        Uses the same parametric equations as ``_update_command`` so the result
        is consistent with the trajectory the policy is tracking. Velocity is
        computed as a backward finite difference at the future time.

        Args:
            time_shift: Seconds ahead of the current step to evaluate.

        Returns:
            pos_w: ``(N, 3)`` world-frame position.
            vel_w: ``(N, 3)`` world-frame velocity.
        """
        dt = max(self._env.step_dt, 1e-6)

        elapsed_f = self._elapsed + time_shift
        phase_f = ((elapsed_f / self._period) + self._phase_offset) % 1.0
        xy_f = _shape_xy(self._shape_id, phase_f, self._size)
        local_f = torch.cat([xy_f, torch.zeros_like(xy_f[:, :1])], dim=-1)
        pos_w = self._center + torch.einsum("nij,nj->ni", self._rot, local_f)

        # Backward finite difference for velocity at the future time.
        phase_prev = (((elapsed_f - dt) / self._period) + self._phase_offset) % 1.0
        xy_prev = _shape_xy(self._shape_id, phase_prev, self._size)
        local_prev = torch.cat([xy_prev, torch.zeros_like(xy_prev[:, :1])], dim=-1)
        pos_prev = self._center + torch.einsum("nij,nj->ni", self._rot, local_prev)
        vel_w = (pos_w - pos_prev) / dt

        return pos_w, vel_w

    def _update_metrics(self) -> None:
        ee_pos = self.robot.data.site_pos_w[:, self._ee_site_local_id]
        self.metrics["ee_pos_error"] = torch.norm(self._target_pos_w - ee_pos, dim=-1)

    # Debug visualisation ------------------------------------------------

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        num_samples = self.cfg.viz.num_path_samples
        # Precompute phases (1, num_samples).
        phases = torch.linspace(0.0, 1.0, num_samples + 1, device=self.device)

        for batch in env_indices:
            shape_id_b = self._shape_id[batch : batch + 1].expand(num_samples + 1)
            size_b = self._size[batch : batch + 1].expand(num_samples + 1)
            xy = _shape_xy(shape_id_b, phases, size_b)  # (S+1, 2)
            local = torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=-1)
            rot_b = self._rot[batch]
            center_b = self._center[batch]
            world = center_b + (rot_b @ local.T).T  # (S+1, 3)
            world_np = world.cpu().numpy()

            for i in range(num_samples):
                visualizer.add_cylinder(
                    start=world_np[i],
                    end=world_np[i + 1],
                    radius=self.cfg.viz.path_radius,
                    color=self.cfg.viz.path_color,
                )

            visualizer.add_sphere(
                center=self._target_pos_w[batch].cpu().numpy(),
                radius=self.cfg.viz.target_radius,
                color=self.cfg.viz.target_color,
                label=f"trajectory_target_{batch}",
            )

            # End-effector position sphere.
            ee_pos = self.robot.data.site_pos_w[batch, self._ee_site_local_id]
            visualizer.add_sphere(
                center=ee_pos.cpu().numpy(),
                radius=self.cfg.viz.ee_radius,
                color=self.cfg.viz.ee_color,
                label=f"ee_pos_{batch}",
            )

            # EE z-axis (approach direction) — teal arrow.
            ee_quat = self.robot.data.site_quat_w[batch, self._ee_site_local_id]
            w, x, y, z = ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]
            ee_z = torch.stack(
                [
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ]
            )
            ee_pos_np = ee_pos.cpu().numpy()
            visualizer.add_arrow(
                start=ee_pos_np,
                end=(ee_pos_np + ee_z.cpu().numpy() * self.cfg.viz.arrow_length),
                color=self.cfg.viz.ee_z_color,
                label=f"ee_z_{batch}",
            )

            # Trajectory plane normal — orange arrow, anchored at the target position.
            plane_normal = self._rot[batch, :, 2]  # third column of rotation matrix
            target_np = self._target_pos_w[batch].cpu().numpy()
            visualizer.add_arrow(
                start=target_np,
                end=(
                    target_np + plane_normal.cpu().numpy() * self.cfg.viz.arrow_length
                ),
                color=self.cfg.viz.plane_normal_color,
                label=f"plane_normal_{batch}",
            )
