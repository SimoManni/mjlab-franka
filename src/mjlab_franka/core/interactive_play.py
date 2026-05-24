"""Interactive viser play app for the Franka trajectory-track task.

Extends :class:`mjlab_franka.core.visualizer.PlayApp` with a GUI that lets
the user select the trajectory shape, randomize its pose, control its period
and size, and record / plot per-step metrics (EE tracking error + joint
smoothness signals).

Assumes the environment has a ``"trajectory"`` command term implemented by
:class:`mjlab_franka.tasks.reaching.mdp.commands.TrajectoryCommand`, and that
the simulation runs with ``num_envs == 1``.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import viser

from mjlab.envs import ManagerBasedRlEnv

from mjlab_franka.core.visualizer import PlayApp, PlayAppConfig
from mjlab_franka.tasks.reaching.mdp.commands import (
    SHAPE_TO_ID,
    ID_TO_SHAPE,
    TrajectoryCommand,
    _sample_random_rotation,
)

log = logging.getLogger(__name__)


_METRIC_KEYS = (
    "ee_pos_error",
    "ee_speed_w",
    "joint_vel_l2",
    "joint_acc_l2",
    "joint_jerk_l2",
    "action_rate_l2",
)


@dataclass
class MetricsRecorder:
    """Per-step metric buffers for a single env."""

    t: list[float] = field(default_factory=list)
    ee_pos_error: list[float] = field(default_factory=list)
    ee_speed_w: list[float] = field(default_factory=list)
    joint_vel_l2: list[float] = field(default_factory=list)
    joint_acc_l2: list[float] = field(default_factory=list)
    joint_jerk_l2: list[float] = field(default_factory=list)
    action_rate_l2: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    def clear(self) -> None:
        self.t.clear()
        for key in _METRIC_KEYS:
            getattr(self, key).clear()

    def append(self, t: float, **values: float) -> None:
        self.t.append(float(t))
        for key in _METRIC_KEYS:
            getattr(self, key).append(float(values[key]))

    def save(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "metrics.csv"
        png_path = out_dir / "metrics.png"

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", *_METRIC_KEYS])
            for i in range(len(self.t)):
                writer.writerow(
                    [self.t[i], *(getattr(self, key)[i] for key in _METRIC_KEYS)]
                )

        # Lazy matplotlib import so headless / fast paths don't pay the cost.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(_METRIC_KEYS)
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 * n), sharex=True)
        for ax, key in zip(axes, _METRIC_KEYS, strict=True):
            ax.plot(self.t, getattr(self, key), linewidth=1.2)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle("Franka trajectory-track metrics")
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)

        return csv_path, png_path

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key in _METRIC_KEYS:
            arr = np.asarray(getattr(self, key), dtype=np.float64)
            if arr.size == 0:
                out[key] = {"mean": 0.0, "std": 0.0, "max": 0.0}
            else:
                out[key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "max": float(arr.max()),
                }
        return out


class InteractivePlayApp(PlayApp):
    """Viser play app with trajectory controls and per-step metric recording."""

    def __init__(
        self,
        cfg: PlayAppConfig,
        env: ManagerBasedRlEnv,
        policy: Any,
        *,
        log_root: Path | None = None,
    ) -> None:
        if env.num_envs != 1:
            log.warning(
                "InteractivePlayApp expects num_envs=1 for clean metrics; got %d.",
                env.num_envs,
            )

        self._cmd: TrajectoryCommand = env.command_manager.get_term("trajectory")
        self._robot = self._cmd.robot
        self._ee_site_local_id = self._cmd._ee_site_local_id  # noqa: SLF001

        # Metric state.
        self._recording = False
        self._metrics = MetricsRecorder()
        self._prev_joint_vel: torch.Tensor | None = None
        self._prev_joint_acc: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None
        self._last_action: torch.Tensor | None = None
        self._prev_ee_pos: torch.Tensor | None = None
        self._elapsed_record_time = 0.0

        # Output directory (created lazily; one subfolder per save).
        session_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = Path(log_root) if log_root is not None else Path.cwd() / "play_logs"
        self._log_dir = root / session_stamp
        self._save_counter = 0

        super().__init__(cfg, env, policy)

    # -- GUI ---------------------------------------------------------------

    def _setup_gui(self) -> None:
        # Read the current per-env-0 trajectory state so initial GUI values match.
        with torch.no_grad():
            init_shape_id = int(self._cmd._shape_id[0].item())  # noqa: SLF001
            init_period = float(self._cmd._period[0].item())  # noqa: SLF001
            init_size = float(self._cmd._size[0].item())  # noqa: SLF001

        init_shape = ID_TO_SHAPE.get(init_shape_id, next(iter(SHAPE_TO_ID)))

        with self.server.gui.add_folder("Trajectory"):
            self._shape_dropdown = self.server.gui.add_dropdown(
                "Shape",
                options=tuple(SHAPE_TO_ID.keys()),
                initial_value=init_shape,
            )
            self._randomize_button = self.server.gui.add_button("Randomize pose")
            per_lo, per_hi = self._cmd.cfg.period_range
            per_min = max(1.0, float(per_lo) * 0.25)
            per_max = max(30.0, float(per_hi) * 2.0)
            self._period_slider = self.server.gui.add_slider(
                "Period (s)",
                min=per_min,
                max=per_max,
                step=0.5,
                initial_value=max(per_min, min(per_max, max(1.0, init_period))),
            )
            size_lo, size_hi = self._cmd.cfg.size_range
            size_min = max(0.02, float(size_lo) * 0.5)
            size_max = max(0.35, float(size_hi) * 1.25)
            self._size_slider = self.server.gui.add_slider(
                "Size (m)",
                min=size_min,
                max=size_max,
                step=0.005,
                initial_value=max(size_min, min(size_max, init_size)),
            )

        with self.server.gui.add_folder("Metrics"):
            self._record_button = self.server.gui.add_button("Start recording")
            self._save_button = self.server.gui.add_button("Save & clear")
            self._samples_text = self.server.gui.add_text(
                "Samples recorded", initial_value="0"
            )
            self._last_save_text = self.server.gui.add_text(
                "Last save", initial_value="(none)"
            )

        with self.server.gui.add_folder("Live readout"):
            self._ee_err_text = self.server.gui.add_text(
                "EE error (m)", initial_value="0.000"
            )
            self._ee_speed_text = self.server.gui.add_text(
                "EE speed (m/s)", initial_value="0.000"
            )
            self._jvel_text = self.server.gui.add_text(
                "Joint vel L2", initial_value="0.000"
            )

        @self._shape_dropdown.on_update
        def _(_: viser.GuiEvent) -> None:
            self._apply_shape(self._shape_dropdown.value)

        @self._randomize_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._command_queue.put("randomize_pose")

        @self._period_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._apply_period(float(self._period_slider.value))

        @self._size_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._apply_size(float(self._size_slider.value))

        @self._record_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._toggle_recording()

        @self._save_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._command_queue.put("save_metrics")

    # -- Command queue dispatch -------------------------------------------

    def _handle_command(self, cmd: str) -> None:
        if cmd == "randomize_pose":
            self._randomize_pose()
        elif cmd == "save_metrics":
            self._save_and_clear()
        else:
            super()._handle_command(cmd)

    # -- Trajectory mutators ----------------------------------------------

    @torch.no_grad()
    def _apply_shape(self, shape_name: str) -> None:
        if shape_name not in SHAPE_TO_ID:
            log.warning("Unknown shape '%s'", shape_name)
            return
        self._cmd._shape_id[0] = SHAPE_TO_ID[shape_name]  # noqa: SLF001
        self._cmd._elapsed[0] = 0.0  # noqa: SLF001
        self._cmd._phase_offset[0] = 0.0  # noqa: SLF001

    @torch.no_grad()
    def _apply_period(self, period: float) -> None:
        period = max(float(period), 1e-2)
        self._cmd._period[0] = period  # noqa: SLF001

    @torch.no_grad()
    def _apply_size(self, size: float) -> None:
        self._cmd._size[0] = max(float(size), 1e-3)  # noqa: SLF001

    @torch.no_grad()
    def _randomize_pose(self) -> None:
        """Re-sample pose, size, period, phase offset for env 0 — keeping shape."""
        device = self._cmd.device
        # Center.
        r = self._cmd.cfg.center_range
        lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=device)
        upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=device)
        local_center = lower + (upper - lower) * torch.rand(1, 3, device=device)
        env_origin = self.env.scene.env_origins[0:1]
        self._cmd._center[0:1] = local_center + env_origin  # noqa: SLF001

        # Rotation (random SO(3), with normal flipped to face away from base).
        rot = _sample_random_rotation(1, device)
        base_to_center = self._cmd._center[0:1] - env_origin
        normal = rot[:, :, 2]
        needs_flip = (torch.sum(normal * base_to_center, dim=-1) < 0.0).item()
        if needs_flip:
            rot[:, :, 0] = -rot[:, :, 0]
            rot[:, :, 2] = -rot[:, :, 2]
        self._cmd._rot[0:1] = rot  # noqa: SLF001

        # Size, period, phase offset.
        size_lo, size_hi = self._cmd.cfg.size_range
        per_lo, per_hi = self._cmd.cfg.period_range
        new_size = float(
            size_lo + (size_hi - size_lo) * torch.rand(1, device=device).item()
        )
        new_period = float(
            per_lo + (per_hi - per_lo) * torch.rand(1, device=device).item()
        )
        self._cmd._size[0] = new_size  # noqa: SLF001
        self._cmd._period[0] = new_period  # noqa: SLF001
        self._cmd._phase_offset[0] = float(torch.rand(1, device=device).item())  # noqa: SLF001
        self._cmd._elapsed[0] = 0.0  # noqa: SLF001

        # Sync GUI sliders so values shown match what's actually in the command.
        self._size_slider.value = max(
            self._size_slider.min, min(self._size_slider.max, new_size)
        )
        self._period_slider.value = max(
            self._period_slider.min, min(self._period_slider.max, new_period)
        )

        log.info(
            "Randomized trajectory: center=%s size=%.3f period=%.2f",
            self._cmd._center[0].cpu().numpy().round(3).tolist(),  # noqa: SLF001
            new_size,
            new_period,
        )

    # -- PlayApp hooks ----------------------------------------------------

    @torch.no_grad()
    def _restore_user_state(self) -> None:
        """Re-apply GUI values to the command term after an auto-reset.

        mjlab calls ``_resample_command`` inside ``env.step()`` whenever an env
        terminates, overwriting shape, size, and period with random values.  This
        method re-applies whatever the user last set via the dropdown / sliders so
        that the trajectory is not silently hijacked by a collision reset.
        """
        self._apply_shape(self._shape_dropdown.value)
        self._apply_size(float(self._size_slider.value))
        self._apply_period(float(self._period_slider.value))

    def _on_reset(self, obs_dict: dict, info: dict) -> None:
        super()._on_reset(obs_dict, info)
        self._prev_joint_vel = None
        self._prev_joint_acc = None
        self._prev_action = None
        self._last_action = None

    def _get_actions(self) -> torch.Tensor:
        actions = super()._get_actions()
        # Cache for action-rate metric on the next step.
        self._last_action = actions.detach()
        return actions

    def _on_step(
        self, obs_dict: dict, reward: Any, terminated: Any, truncated: Any, info: dict
    ) -> None:
        super()._on_step(obs_dict, reward, terminated, truncated, info)
        # If env 0 was auto-reset inside env.step() (collision / time-out),
        # mjlab has already called _resample_command([0]) which overwrites shape,
        # size, and period with new random values.  Restore the user's settings.
        if bool(terminated[0].item()) or bool(truncated[0].item()):
            self._restore_user_state()
        if not self._recording:
            self._prev_joint_vel = None
            self._prev_joint_acc = None
            self._prev_action = None
            return

        dt = float(self.env.step_dt)
        env_idx = 0

        # EE error & speed.
        ee_pos = self._robot.data.site_pos_w[env_idx, self._ee_site_local_id]
        target_pos = self._cmd.target_pos_w[env_idx]
        target_vel = self._cmd.target_vel_w[env_idx]
        ee_pos_err = float(torch.norm(ee_pos - target_pos).item())

        # EE speed in world frame from the trajectory term's FD velocity. The
        # robot's site_pos_w lag is the cleanest stable signal; we instead
        # report the actual EE world speed by FD-ing site_pos_w across steps.
        if self._prev_ee_pos is None:
            ee_speed = 0.0
        else:
            ee_speed = float(
                torch.norm((ee_pos - self._prev_ee_pos) / max(dt, 1e-6)).item()
            )
        self._prev_ee_pos = ee_pos.detach().clone()
        # Silence unused-variable warning while keeping target_vel available
        # for future extensions (e.g. tracking-error metric).
        del target_vel

        # Joint smoothness.
        joint_vel = self._robot.data.joint_vel[env_idx].detach()
        joint_vel_l2 = float(torch.norm(joint_vel).item())

        if self._prev_joint_vel is not None:
            joint_acc = (joint_vel - self._prev_joint_vel) / max(dt, 1e-6)
        else:
            joint_acc = torch.zeros_like(joint_vel)
        joint_acc_l2 = float(torch.norm(joint_acc).item())

        if self._prev_joint_acc is not None:
            joint_jerk = (joint_acc - self._prev_joint_acc) / max(dt, 1e-6)
        else:
            joint_jerk = torch.zeros_like(joint_acc)
        joint_jerk_l2 = float(torch.norm(joint_jerk).item())

        # Action rate.
        if self._prev_action is not None and self._last_action is not None:
            action_rate = (
                self._last_action[env_idx] - self._prev_action[env_idx]
            ) / max(dt, 1e-6)
            action_rate_l2 = float(torch.norm(action_rate).item())
        else:
            action_rate_l2 = 0.0

        self._prev_joint_vel = joint_vel
        self._prev_joint_acc = joint_acc
        if self._last_action is not None:
            self._prev_action = self._last_action.detach().clone()

        self._elapsed_record_time += dt
        self._metrics.append(
            self._elapsed_record_time,
            ee_pos_error=ee_pos_err,
            ee_speed_w=ee_speed,
            joint_vel_l2=joint_vel_l2,
            joint_acc_l2=joint_acc_l2,
            joint_jerk_l2=joint_jerk_l2,
            action_rate_l2=action_rate_l2,
        )

        self._samples_text.value = str(len(self._metrics))
        self._ee_err_text.value = f"{ee_pos_err:.4f}"
        self._ee_speed_text.value = f"{ee_speed:.4f}"
        self._jvel_text.value = f"{joint_vel_l2:.4f}"

    # -- Metric record / save ---------------------------------------------

    def _toggle_recording(self) -> None:
        self._recording = not self._recording
        if self._recording:
            self._record_button.label = "Stop recording"
            self._elapsed_record_time = 0.0
            self._prev_ee_pos = None
            log.info("Metrics recording: ON")
        else:
            self._record_button.label = "Start recording"
            log.info("Metrics recording: OFF (%d samples buffered)", len(self._metrics))

    def _save_and_clear(self) -> None:
        if len(self._metrics) == 0:
            log.warning("No metrics recorded yet; press 'Start recording' first.")
            self._last_save_text.value = "(no samples)"
            return

        self._save_counter += 1
        out_dir = self._log_dir / f"save_{self._save_counter:03d}"
        csv_path, png_path = self._metrics.save(out_dir)

        summary = self._metrics.summary()
        log.info("Saved metrics to %s", out_dir)
        for key, stats in summary.items():
            log.info(
                "  %-16s  mean=%.4f  std=%.4f  max=%.4f",
                key,
                stats["mean"],
                stats["std"],
                stats["max"],
            )

        self._last_save_text.value = str(out_dir)
        self._metrics.clear()
        self._samples_text.value = "0"
        self._elapsed_record_time = 0.0
        log.info("Cleared metric buffers.")
        # Silence unused-variable lints.
        del csv_path, png_path
