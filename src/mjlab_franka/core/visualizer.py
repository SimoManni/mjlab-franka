"""Shared viser-based visualization (ported from aloy/core/visualizer.py).

Key design (mirrors aloy/starfish):

* The simulator still runs ``num_envs`` envs in parallel; *only the
  visualization* is downsampled. The viser ``MjlabViserScene`` is built with a
  small ``num_envs`` (``num_visible_envs`` argument) and only one env is
  rendered at a time.
* An ``Env #`` slider in the viser GUI switches which env is rendered.
* Per-env data is sliced before being pushed to the scene, so we never
  serialize all (e.g.) 256 envs every update.

Provides:

* Helper utilities (``apply_collision_visual_overrides``, ``update_all_groups``,
  ``rebuild_changed_geom_handles``, ``show_phase_legend``).
* ``ViserEnvApp``: base class with viser server, scene, and standard UI controls
  (Pause/Step/Reset, Env slider, Show All, Show Collision, Speed, RTF).
* ``PlayApp`` (+ ``PlayAppConfig``): runs a policy in a viser app loop.

The policy callable passed to :class:`PlayApp` receives the obs *tensor* for
``obs_group`` (defaults to ``"policy"``), matching the lightweight policies
constructed in :mod:`mjlab_franka.play`.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.viser.scene import MjlabViserScene as ViserMujocoScene
from mjviser.conversions import get_body_name, is_fixed_body, merge_geoms
from mujoco import mj_id2name, mjtObj
from viser import transforms as vtf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geom group helpers
# ---------------------------------------------------------------------------


def _is_collision_geom(mj_model: Any, geom_id: int) -> bool:
    return (
        mj_model.geom_contype[geom_id] != 0 or mj_model.geom_conaffinity[geom_id] != 0
    )


def _geom_name(mj_model: Any, geom_id: int) -> str:
    name = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, geom_id)
    return name or ""


def _is_visual_geom(mj_model: Any, geom_id: int) -> bool:
    if not _is_collision_geom(mj_model, geom_id):
        return True
    name = _geom_name(mj_model, geom_id).lower()
    return "visual" in name or name.endswith("_vis") or "viz" in name


def apply_collision_visual_overrides(
    mj_model: Any,
    *,
    collision_alpha: float | None,
    collision_rgb: tuple[float, float, float] | None = None,
) -> None:
    """Mutate ``mj_model.geom_rgba`` so collision geoms are visible at ``collision_alpha``."""
    if collision_alpha is None:
        return
    for geom_id in range(mj_model.ngeom):
        if not _is_collision_geom(mj_model, geom_id):
            continue
        if _is_visual_geom(mj_model, geom_id):
            continue
        rgba = mj_model.geom_rgba[geom_id]
        if collision_rgb is not None:
            rgba[:3] = np.array(collision_rgb, dtype=rgba.dtype)
        rgba[3] = float(collision_alpha)


def set_collision_geom_groups_visible(
    scene: ViserMujocoScene, mj_model: Any, visible: bool
) -> None:
    flags = list(scene.geom_groups_visible)
    group_has_visual: dict[int, bool] = {}
    group_has_collision_only: dict[int, bool] = {}
    changed = False
    for geom_id in range(mj_model.ngeom):
        group = int(mj_model.geom_group[geom_id])
        if _is_visual_geom(mj_model, geom_id):
            group_has_visual[group] = True
        if _is_collision_geom(mj_model, geom_id) and not _is_visual_geom(
            mj_model, geom_id
        ):
            group_has_collision_only[group] = True

    for group in group_has_collision_only:
        if group_has_visual.get(group, False):
            continue
        if 0 <= group < len(flags) and flags[group] != visible:
            flags[group] = visible
            changed = True

    for group in group_has_visual:
        if group > 4:
            continue
        if 0 <= group < len(flags) and not flags[group]:
            flags[group] = True
            changed = True

    if changed:
        scene.geom_groups_visible = flags
        scene._sync_visibilities()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Scene update wrapper
# ---------------------------------------------------------------------------


def update_all_groups(
    scene: ViserMujocoScene, *args: Any, use_mjdata: bool = False, **kwargs: Any
) -> None:
    """Wrap ``scene.update``/``update_from_mjdata`` so all groups are visible during the write.

    mjlab's internal update skips position writes for invisible handles. Force
    visibility during the update so every handle gets correct transforms, then
    restore the saved flags. The whole toggle is wrapped in an atomic block so
    no flicker is visible to the client.
    """
    saved_geom = list(scene.geom_groups_visible)
    saved_site = list(scene.site_groups_visible)
    scene.geom_groups_visible = [True] * len(saved_geom)
    scene.site_groups_visible = [True] * len(saved_site)
    with scene.server.atomic():
        scene._sync_visibilities()  # noqa: SLF001
        if use_mjdata:
            scene.update_from_mjdata(*args, **kwargs)
        else:
            scene.update(*args, **kwargs)
        scene.geom_groups_visible = [bool(v) for v in saved_geom]
        scene.site_groups_visible = [bool(v) for v in saved_site]
        scene._sync_visibilities()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Per-env geometry rebuild (selected env may have different geom sizes/pos)
# ---------------------------------------------------------------------------


def build_handle_geom_mapping(mj_model: Any) -> dict[tuple[int, int], list[int]]:
    """``(body_id, group_id) -> [geom_ids]`` for non-fixed, non-transparent geoms."""
    mapping: dict[tuple[int, int], list[int]] = {}
    for i in range(mj_model.ngeom):
        body_id = int(mj_model.geom_bodyid[i])
        if is_fixed_body(mj_model, body_id):
            continue
        if mj_model.geom_rgba[i, 3] == 0:
            continue
        key = (body_id, int(mj_model.geom_group[i]))
        mapping.setdefault(key, []).append(i)
    return mapping


def rebuild_changed_geom_handles(
    scene: ViserMujocoScene,
    handle_geom_mapping: dict[tuple[int, int], list[int]],
    original_geom_size: np.ndarray,
    original_geom_pos: np.ndarray,
    env_geom_size: np.ndarray,
    env_geom_pos: np.ndarray,
) -> None:
    """Recreate batched mesh handles whose geom sizes/positions have changed for the selected env."""
    size_changed = not np.allclose(original_geom_size, env_geom_size, atol=1e-8)
    pos_changed = not np.allclose(original_geom_pos, env_geom_pos, atol=1e-8)
    if not size_changed and not pos_changed:
        return
    mj_model = scene.mj_model
    saved_sizes = mj_model.geom_size.copy()
    saved_pos = mj_model.geom_pos.copy()
    try:
        mj_model.geom_size[:] = env_geom_size
        mj_model.geom_pos[:] = env_geom_pos
        for mg in scene._mesh_groups:  # noqa: SLF001
            prototype_body = int(mg.body_ids[0])
            group_id = int(mg.group_id)
            geom_ids = handle_geom_mapping.get((prototype_body, group_id))
            if not geom_ids:
                continue
            if np.allclose(
                original_geom_size[geom_ids], env_geom_size[geom_ids], atol=1e-8
            ) and np.allclose(
                original_geom_pos[geom_ids], env_geom_pos[geom_ids], atol=1e-8
            ):
                continue
            body_name = get_body_name(mj_model, prototype_body)
            mesh = merge_geoms(mj_model, geom_ids)
            lod_ratio = 1000.0 / mesh.vertices.shape[0]
            old_handle = mg.handle
            new_handle = scene.server.scene.add_batched_meshes_trimesh(
                f"/bodies/{body_name}/group{group_id}",
                mesh,
                batched_wxyzs=old_handle.batched_wxyzs,
                batched_positions=old_handle.batched_positions,
                lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
                visible=old_handle.visible,
            )
            old_handle.remove()
            mg.handle = new_handle
    finally:
        mj_model.geom_size[:] = saved_sizes
        mj_model.geom_pos[:] = saved_pos


# ---------------------------------------------------------------------------
# Command markers
# ---------------------------------------------------------------------------


def show_phase_legend(env: ManagerBasedRlEnv, server: viser.ViserServer) -> None:
    """Best-effort: render a phase legend if the env has a ``task_phase`` command."""
    try:
        phase_command = env.command_manager.get_term("task_phase")
        if phase_command is not None:
            with server.gui.add_folder("Phase Legend"):
                phase_legend = phase_command.get_phase_legend()
                for phase_idx in sorted(phase_legend.keys()):
                    name, emoji = phase_legend[phase_idx]
                    server.gui.add_markdown(f" {emoji} {name}")
    except (AttributeError, KeyError):
        log.debug("No task_phase command available for phase legend")


# ---------------------------------------------------------------------------
# ViserEnvApp base
# ---------------------------------------------------------------------------


class ViserEnvApp:
    """Base class for viser-based env visualization apps.

    Subclass hooks:
        _setup_gui():    Add extra GUI elements.
        _get_actions():  Return actions tensor (required).
        _on_step():      Called after each env.step().
        _on_reset():     Called after each env.reset().
        _tick():         Per-loop work (e.g. background polling).
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        port: int = 8080,
        collision_alpha: float = 0.3,
        num_visible_envs: int | None = None,
        server_label: str = "mjlab-play",
    ) -> None:
        self.env = env
        self.num_envs = (
            env.num_envs
            if num_visible_envs is None
            else min(num_visible_envs, env.num_envs)
        )
        self._step_interval = env.step_dt
        self._selected_env = 0
        self._show_all = False

        self.server = viser.ViserServer(port=port, label=server_label)
        self._paused = True
        self._last_step_time = time.time()
        self._step_times: list[float] = []

        self._command_queue: queue.Queue[str] = queue.Queue()

        apply_collision_visual_overrides(
            self.env.sim.mj_model, collision_alpha=collision_alpha
        )
        self._scene = ViserMujocoScene(
            server=self.server,
            mj_model=self.env.sim.mj_model,
            num_envs=self.num_envs,
        )
        self._scene.debug_visualization_enabled = True
        set_collision_geom_groups_visible(self._scene, self.env.sim.mj_model, False)

        self._original_geom_size = self.env.sim.mj_model.geom_size.copy()
        self._original_geom_pos = self.env.sim.mj_model.geom_pos.copy()
        self._handle_geom_mapping = build_handle_geom_mapping(self.env.sim.mj_model)

        self._frame_handles: dict[str, viser.FrameHandle] = {}
        self._frame_body_ids: dict[str, int] = {}

        self._setup_base_gui()
        self._setup_gui()
        self._create_body_frames()
        self._reset_env()

    # -- GUI ---------------------------------------------------------------

    def _setup_base_gui(self) -> None:
        # Floor fallback when no plane geom is in the model.
        has_plane_geom = any(
            self.env.sim.mj_model.geom_type[i] == 0  # mjGEOM_PLANE
            for i in range(self.env.sim.mj_model.ngeom)
        )
        if not has_plane_geom:
            self.server.scene.add_grid(
                "/floor",
                width=2000.0,
                height=2000.0,
                infinite_grid=True,
                fade_distance=50.0,
                shadow_opacity=0.2,
            )

        with self.server.gui.add_folder("Control"):
            self._pause_button = self.server.gui.add_button("Start / Stop")
            self._reset_button = self.server.gui.add_button("Reset")
            self._step_button = self.server.gui.add_button("Step")
            self._max_speed_checkbox = self.server.gui.add_checkbox(
                "Max Speed", initial_value=False
            )
            self._speed_slider = self.server.gui.add_slider(
                label="Speed", min=0.0, max=2.0, step=0.1, initial_value=1.0
            )

        with self.server.gui.add_folder("View"):
            self._show_all_checkbox = self.server.gui.add_checkbox(
                "Show All Envs", initial_value=False
            )
            self._collision_checkbox = self.server.gui.add_checkbox(
                "Show Collision", initial_value=False
            )
            self._env_slider = self.server.gui.add_slider(
                label="Env #",
                min=0,
                max=max(self.num_envs - 1, 0),
                step=1,
                initial_value=0,
            )

        with self.server.gui.add_folder("Info"):
            self._num_envs_text = self.server.gui.add_text(
                "Envs", initial_value=f"{self.num_envs} / {self.env.num_envs}"
            )
            self._step_counter_text = self.server.gui.add_text(
                "Steps", initial_value="0"
            )
            self._reward_text = self.server.gui.add_text(
                "Reward (mean)", initial_value="0.0"
            )
            self._rtf_text = self.server.gui.add_text("RTF", initial_value="--")

        @self._pause_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._paused = not self._paused
            if not self._paused:
                self._step_times.clear()

        @self._reset_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._command_queue.put("reset")

        @self._step_button.on_click
        def _(_: viser.GuiEvent) -> None:
            self._command_queue.put("step")

        @self._show_all_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            self._show_all = self._show_all_checkbox.value
            self._create_body_frames()
            self._sync_visualization()

        @self._collision_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            set_collision_geom_groups_visible(
                self._scene, self.env.sim.mj_model, self._collision_checkbox.value
            )
            self._sync_visualization()

        @self._env_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            self._selected_env = int(self._env_slider.value)
            if not self._show_all:
                self._create_body_frames()
                self._sync_visualization()

        @self._speed_slider.on_update
        def _(_: viser.GuiEvent) -> None:
            factor = self._speed_slider.value
            self._step_interval = self.env.step_dt / factor if factor > 0 else 1e9

        show_phase_legend(self.env, self.server)

    def _setup_gui(self) -> None:
        """Override to add additional GUI elements."""

    # -- Body frames -------------------------------------------------------

    def _create_body_frames(self) -> None:
        for frame in self._frame_handles.values():
            frame.remove()
        self._frame_handles.clear()
        self._frame_body_ids = {}
        if self._show_all:
            return
        mj_model = self.env.sim.mj_model
        body_ids = {int(bid) for mg in self._scene._mesh_groups for bid in mg.body_ids}  # noqa: SLF001
        for body_id in sorted(body_ids):
            if is_fixed_body(mj_model, body_id):
                continue
            body_name = get_body_name(mj_model, body_id)
            frame = self.server.scene.add_frame(
                f"/bodies/{body_name}/frame",
                axes_length=0.1,
                axes_radius=0.005,
                visible=False,
            )
            self._frame_handles[body_name] = frame
            self._frame_body_ids[body_name] = body_id

    def _update_body_frames(self) -> None:
        if not self._frame_handles or self._show_all:
            return
        data = self.env.sim.data
        xpos = data.xpos.cpu().numpy()
        xmat = data.xmat.cpu().numpy()
        xquat = vtf.SO3.from_matrix(xmat).wxyz
        env_idx = self._selected_env
        for body_name, frame in self._frame_handles.items():
            body_id = self._frame_body_ids.get(body_name)
            if body_id is not None:
                frame.position = tuple(xpos[env_idx, body_id, :])
                frame.wxyz = tuple(xquat[env_idx, body_id, :])

    # -- Sync --------------------------------------------------------------

    def _sync_visualization(self) -> None:
        self._scene.env_idx = self._selected_env
        self._scene.show_only_selected = not self._show_all
        self._scene.show_all_envs = self._show_all
        self._scene.clear()

        if hasattr(self.env, "command_manager") and hasattr(
            self.env.command_manager, "debug_vis"
        ):
            try:
                self.env.command_manager.debug_vis(self._scene)
            except Exception as exc:  # noqa: BLE001
                log.debug("command_manager.debug_vis failed: %s", exc)

        if hasattr(self.env, "event_manager") and hasattr(
            self.env.event_manager, "debug_vis"
        ):
            try:
                self.env.event_manager.debug_vis(self._scene)
            except Exception as exc:  # noqa: BLE001
                log.debug("event_manager.debug_vis failed: %s", exc)

        update_all_groups(self._scene, self.env.sim.data, env_idx=self._selected_env)
        self._update_body_frames()
        try:
            env_sizes = (
                self.env.sim.model.geom_size.cpu().numpy()[self._selected_env].copy()
            )
            env_pos = (
                self.env.sim.model.geom_pos.cpu().numpy()[self._selected_env].copy()
            )
            rebuild_changed_geom_handles(
                self._scene,
                self._handle_geom_mapping,
                self._original_geom_size,
                self._original_geom_pos,
                env_sizes,
                env_pos,
            )
        except (AttributeError, IndexError) as exc:
            log.debug("Skipping per-env geom rebuild: %s", exc)

    # -- Loop --------------------------------------------------------------

    def _reset_env(self) -> None:
        obs_dict, info = self.env.reset()
        self._on_reset(obs_dict, info)
        self._sync_visualization()
        self._step_counter_text.value = "0"
        self._reward_text.value = "0.0"
        self._rtf_text.value = "--"
        self._step_times.clear()

    def _on_reset(self, obs_dict: dict, info: dict) -> None:
        """Override hook."""

    def _get_actions(self) -> Any:
        raise NotImplementedError

    def _step_env(self) -> None:
        actions = self._get_actions()
        step_start = time.perf_counter()
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        self._on_step(obs_dict, reward, terminated, truncated, info)
        self._sync_visualization()
        step_end = time.perf_counter()

        self._step_times.append(step_end - step_start)
        if len(self._step_times) > 50:
            self._step_times.pop(0)
        if len(self._step_times) >= 5:
            avg_wall_dt = sum(self._step_times) / len(self._step_times)
            sim_dt = self.env.step_dt
            rtf = sim_dt / avg_wall_dt if avg_wall_dt > 0 else 0.0
            self._rtf_text.value = f"{rtf:.2f}x"

        try:
            self._step_counter_text.value = str(self.env.common_step_counter)
        except AttributeError:
            pass
        try:
            self._reward_text.value = f"{reward.mean().item():.4f}"
        except (AttributeError, RuntimeError):
            pass

    def _on_step(
        self, obs_dict: dict, reward: Any, terminated: Any, truncated: Any, info: dict
    ) -> None:
        """Override hook."""

    def _handle_command(self, cmd: str) -> None:
        if cmd == "reset":
            self._reset_env()
        elif cmd == "step":
            self._step_env()

    def _process_commands(self) -> None:
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                self._handle_command(cmd)
            except queue.Empty:
                break

    def _tick(self) -> None:
        """Override hook."""

    def run(self) -> None:
        log.info(
            "viser visualizer running at http://localhost:%d", self.server.get_port()
        )
        try:
            while True:
                self._tick()
                self._process_commands()
                if not self._paused:
                    if self._max_speed_checkbox.value:
                        self._step_env()
                    else:
                        current_time = time.time()
                        if current_time - self._last_step_time >= self._step_interval:
                            self._step_env()
                            self._last_step_time = current_time
                        else:
                            time.sleep(0.001)
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# PlayApp
# ---------------------------------------------------------------------------


@dataclass
class PlayAppConfig:
    """Configuration for :class:`PlayApp`."""

    port: int = 8080
    num_visible_envs: int | None = None
    """Max envs to expose in the viser scene. Sim still runs ``env.num_envs``."""
    obs_group: str = "policy"
    """Obs-dict key whose tensor is passed to the policy callable."""


# Policy callable signature used by PlayApp.
PolicyFn = Callable[[torch.Tensor], torch.Tensor]


class PlayApp(ViserEnvApp):
    """Run a policy in a viser visualization app.

    The policy is a plain callable ``policy(obs_tensor) -> action_tensor`` —
    matching :func:`mjlab_franka.play._random_policy`,
    :func:`mjlab_franka.play._zero_policy`, and the closure returned by
    :func:`mjlab_franka.play._skrl_policy_from_checkpoint`.
    """

    def __init__(
        self,
        cfg: PlayAppConfig,
        env: ManagerBasedRlEnv,
        policy: PolicyFn,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self._obs_dict: dict | None = None
        super().__init__(
            env,
            port=cfg.port,
            num_visible_envs=cfg.num_visible_envs,
            server_label="mjlab-play",
        )
        # Auto-start playback so the policy runs without a manual button press.
        self._paused = False

    def _on_reset(self, obs_dict: dict, info: dict) -> None:
        self._obs_dict = obs_dict

    def _on_step(
        self, obs_dict: dict, reward: Any, terminated: Any, truncated: Any, info: dict
    ) -> None:
        self._obs_dict = obs_dict
        # Auto-reset on episode end (mjlab envs reset automatically on step, but
        # we still refresh the cached obs dict here).

    def _get_actions(self) -> torch.Tensor:
        if self._obs_dict is None:
            self._obs_dict, _info = self.env.reset()
        obs_tensor = self._obs_dict[self.cfg.obs_group]
        with torch.inference_mode():
            actions = self.policy(obs_tensor)
            if isinstance(actions, tuple):
                actions = actions[0]
        return actions
