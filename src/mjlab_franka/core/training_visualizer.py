"""Training visualizer (ported from aloy/core/training_visualizer.py).

Replaces the old ``LiveTrainingVisualizer`` (which built the viser scene with
``num_envs=env.num_envs`` and pushed all envs every update — that's what was
freezing the UI).

Design (mirrors aloy/starfish):

* The simulator still runs all ``env.num_envs`` envs in parallel.
* The viser scene is built with ``num_envs=1`` — only one env is rendered.
* A small subset of envs (default 16, linearly spaced) is *recorded* into a
  bounded deque of :class:`_Frame`s after each ``env.step()``.
* A viser GUI exposes an "Env index" slider + a timeline scrubber + Play/Pause
  + "Follow live", letting the user replay any recorded env on demand.
* :class:`RecordingEnvWrapper` is a transparent wrapper around the env that
  calls :meth:`TrainingVisualizer.record` after each ``step()`` — synchronous,
  so there's no thread fighting for the GPU/server.

Usage from ``train.py``::

    env = make_env(...)
    if cfg.visualize:
        viz = TrainingVisualizer(env)
        env = RecordingEnvWrapper(env, viz)
    train_fn(cfg, env)
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any

import numpy as np
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.viser.scene import MjlabViserScene as ViserMujocoScene

from mjlab_franka.core.visualizer import (
    apply_collision_visual_overrides,
    build_handle_geom_mapping,
    rebuild_changed_geom_handles,
    set_collision_geom_groups_visible,
    show_phase_legend,
    update_all_groups,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame storage
# ---------------------------------------------------------------------------


class _Frame:
    """A single recorded frame of body poses (subset of envs)."""

    __slots__ = (
        "arrows",
        "cylinders",
        "geom_pos",
        "geom_size",
        "mocap_pos",
        "mocap_quat",
        "qpos",
        "qvel",
        "spheres",
        "xmat",
        "xpos",
    )

    def __init__(
        self,
        xpos: np.ndarray,
        xmat: np.ndarray,
        mocap_pos: np.ndarray,
        mocap_quat: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        arrows: list[
            list[
                tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]
            ]
        ],
        cylinders: list[
            list[
                tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float]]
            ]
        ],
        spheres: list[
            list[tuple[np.ndarray, float, tuple[float, float, float, float]]]
        ],
        geom_size: np.ndarray | None = None,
        geom_pos: np.ndarray | None = None,
    ) -> None:
        self.xpos = xpos
        self.xmat = xmat
        self.mocap_pos = mocap_pos
        self.mocap_quat = mocap_quat
        self.qpos = qpos
        self.qvel = qvel
        self.arrows = arrows
        self.cylinders = cylinders
        self.spheres = spheres
        self.geom_size = geom_size
        self.geom_pos = geom_pos


# ---------------------------------------------------------------------------
# Debug-vis collector (captures arrows/cylinders/spheres per env_idx)
# ---------------------------------------------------------------------------


class _Collector(DebugVisualizer):
    """Sink for ``command_manager.debug_vis(visualizer)`` calls."""

    def __init__(self, env_idx: int, meansize: float):
        self.env_idx = env_idx
        self.show_all_envs = False
        self._meansize = meansize
        self.arrows: list[
            tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]
        ] = []
        self.cylinders: list[
            tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float]]
        ] = []
        self.spheres: list[
            tuple[np.ndarray, float, tuple[float, float, float, float]]
        ] = []

    @property
    def meansize(self) -> float:
        return self._meansize

    def add_arrow(
        self,
        start: np.ndarray,
        end: np.ndarray,
        color: tuple[float, float, float, float] | tuple[float, float, float],
        width: float = 0.015,
        label: str | None = None,
    ) -> None:
        del label
        if hasattr(start, "cpu"):
            start = start.cpu().numpy()
        if hasattr(end, "cpu"):
            end = end.cpu().numpy()
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)
        self.arrows.append((start.copy(), end.copy(), color, width))

    def add_ghost_mesh(
        self, qpos: np.ndarray, model: Any, alpha: float = 0.5, label: str | None = None
    ) -> None:
        pass

    def add_frame(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        scale: float = 0.3,
        label: str | None = None,
        axis_radius: float = 0.01,
        alpha: float = 1.0,
        axis_colors: np.ndarray | None = None,
    ) -> None:
        pass

    def add_sphere(
        self,
        center: np.ndarray,
        radius: float,
        color: tuple[float, float, float, float] | tuple[float, float, float],
        label: str | None = None,
    ) -> None:
        del label
        if hasattr(center, "cpu"):
            center = center.cpu().numpy()
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)
        self.spheres.append((center.copy(), radius, color))

    def add_cylinder(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: tuple[float, float, float, float] | tuple[float, float, float],
        label: str | None = None,
    ) -> None:
        del label
        if hasattr(start, "cpu"):
            start = start.cpu().numpy()
        if hasattr(end, "cpu"):
            end = end.cpu().numpy()
        if len(color) == 3:
            color = (color[0], color[1], color[2], 1.0)
        self.cylinders.append((start.copy(), end.copy(), radius, color))

    def add_ellipsoid(
        self,
        center: np.ndarray,
        size: np.ndarray,
        mat: np.ndarray,
        color: np.ndarray,
        label: str | None = None,
    ) -> None:
        del center, size, mat, color, label

    def clear(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Adapter: present recorded numpy arrays with the mjwarp.Data interface
# ---------------------------------------------------------------------------


class _NumpyWrapper:
    """Mimic torch/warp tensor API used by ``MjlabViserScene.update()``."""

    __slots__ = ("_array",)

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "_NumpyWrapper":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _RecordedWpData:
    """Adapter to mimic ``mjwarp.Data`` for recorded numpy arrays."""

    __slots__ = ("mocap_pos", "mocap_quat", "qpos", "qvel", "xmat", "xpos")

    def __init__(
        self,
        *,
        xpos: np.ndarray,
        xmat: np.ndarray,
        mocap_pos: np.ndarray,
        mocap_quat: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        self.xpos = _NumpyWrapper(xpos)
        self.xmat = _NumpyWrapper(xmat)
        self.mocap_pos = _NumpyWrapper(mocap_pos)
        self.mocap_quat = _NumpyWrapper(mocap_quat)
        self.qpos = _NumpyWrapper(qpos)
        self.qvel = _NumpyWrapper(qvel)


# ---------------------------------------------------------------------------
# Training visualizer
# ---------------------------------------------------------------------------


class TrainingVisualizer:
    """Records frames during training; serves a viser UI for env selection and timeline scrubbing."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        max_frames: int = 2_000,
        record_envs: list[int] | None = None,
        record_every: int = 1,
        port: int = 8080,
        collision_alpha: float = 0.3,
    ) -> None:
        self.env = env
        self._record_every = record_every
        if record_envs is None:
            num_envs = min(16, env.num_envs)
            record_envs = np.linspace(
                0, env.num_envs - 1, num=num_envs, dtype=int
            ).tolist()
        self._record_env_ids = [eid for eid in record_envs if 0 <= eid < env.num_envs]
        self._step_count = 0
        self._frames: collections.deque[_Frame] = collections.deque(maxlen=max_frames)

        # Scene with num_envs=1: only one env's geometry is serialized to viser.
        # User picks which recorded env to show via the "Env index" slider.
        self._num_envs = len(self._record_env_ids)
        self._selected_env = 0
        self._server = viser.ViserServer(port=port, label="mjlab-train")
        apply_collision_visual_overrides(
            env.sim.mj_model, collision_alpha=collision_alpha
        )
        self._scene = ViserMujocoScene(
            server=self._server,
            mj_model=env.sim.mj_model,
            num_envs=1,
        )
        self._scene.debug_visualization_enabled = True
        set_collision_geom_groups_visible(self._scene, env.sim.mj_model, False)

        self._original_geom_size = env.sim.mj_model.geom_size.copy()
        self._original_geom_pos = env.sim.mj_model.geom_pos.copy()
        self._handle_geom_mapping = build_handle_geom_mapping(env.sim.mj_model)

        self._setup_gui()

        log.info(
            "Training visualizer at http://localhost:%d (recording %d/%d envs: %s, max %d frames)",
            port,
            self._num_envs,
            env.num_envs,
            self._record_env_ids,
            max_frames,
        )

    # -- Recording ---------------------------------------------------------

    def record(self) -> None:
        """Capture current body poses. Called after each ``env.step()``."""
        self._step_count += 1
        if self._step_count % self._record_every != 0:
            return
        if not self._record_env_ids:
            return

        wp = self.env.sim.wp_data
        env_ids = self._record_env_ids
        xpos = wp.xpos.numpy()[env_ids].copy()
        xmat = wp.xmat.numpy()[env_ids].copy()
        if self.env.sim.mj_model.nmocap > 0:
            mocap_pos = wp.mocap_pos.numpy()[env_ids].copy()
            mocap_quat = wp.mocap_quat.numpy()[env_ids].copy()
        else:
            mocap_pos = np.zeros((len(env_ids), 0, 3), dtype=np.float32)
            mocap_quat = np.zeros((len(env_ids), 0, 4), dtype=np.float32)
        qpos = wp.qpos.numpy()[env_ids].copy()
        qvel = wp.qvel.numpy()[env_ids].copy()
        try:
            geom_size = self.env.sim.model.geom_size.cpu().numpy()[env_ids].copy()
            geom_pos = self.env.sim.model.geom_pos.cpu().numpy()[env_ids].copy()
        except (AttributeError, IndexError):
            geom_size = None
            geom_pos = None

        arrows: list[
            list[
                tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]
            ]
        ] = []
        cylinders: list[
            list[
                tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float]]
            ]
        ] = []
        spheres: list[
            list[tuple[np.ndarray, float, tuple[float, float, float, float]]]
        ] = []
        meansize = float(self.env.sim.mj_model.stat.meansize)
        for env_id in env_ids:
            collector = _Collector(env_idx=env_id, meansize=meansize)
            try:
                self.env.command_manager.debug_vis(collector)
            except AttributeError:
                pass
            except Exception as exc:  # noqa: BLE001
                log.debug("debug_vis failed for env %d: %s", env_id, exc)
            arrows.append(collector.arrows)
            cylinders.append(collector.cylinders)
            spheres.append(collector.spheres)

        self._frames.append(
            _Frame(
                xpos=xpos,
                xmat=xmat,
                mocap_pos=mocap_pos,
                mocap_quat=mocap_quat,
                qpos=qpos,
                qvel=qvel,
                arrows=arrows,
                cylinders=cylinders,
                spheres=spheres,
                geom_size=geom_size,
                geom_pos=geom_pos,
            )
        )
        n = len(self._frames)
        self._frame_count_text.value = str(n)

        if self._follow_live.value:
            self._frame_slider.max = max(n - 1, 1)
            self._frame_slider.value = n - 1
            self._render_frame(n - 1)
        else:
            self._frame_slider.max = max(n - 1, 1)

    # -- GUI ---------------------------------------------------------------

    def _setup_gui(self) -> None:
        # Floor fallback when no plane geom exists in the model.
        has_plane_geom = any(
            self.env.sim.mj_model.geom_type[i] == 0
            for i in range(self.env.sim.mj_model.ngeom)
        )
        if not has_plane_geom:
            self._server.scene.add_grid(
                "/floor",
                width=2000.0,
                height=2000.0,
                infinite_grid=True,
                fade_distance=50.0,
                shadow_opacity=0.2,
            )

        with self._server.gui.add_folder("View"):
            self.env_slider = self._server.gui.add_slider(
                label="Env index",
                min=0,
                max=max(self._num_envs - 1, 0),
                step=1,
                initial_value=0,
            )
            self._collision_checkbox = self._server.gui.add_checkbox(
                "Show Collision", initial_value=False
            )
            self._env_id_text = self._server.gui.add_text(
                "Env ID",
                initial_value=str(self._record_env_ids[0])
                if self._record_env_ids
                else "--",
            )

        with self._server.gui.add_folder("Timeline"):
            self._frame_slider = self._server.gui.add_slider(
                label="Frame", min=0, max=1, step=1, initial_value=0
            )
            self._play_button = self._server.gui.add_button("Play")
            self._playback_speed = self._server.gui.add_slider(
                label="Speed (fps)", min=1, max=120, step=1, initial_value=30
            )
            self._follow_live = self._server.gui.add_checkbox(
                "Follow live", initial_value=True
            )
            self._frame_count_text = self._server.gui.add_text(
                "Recorded", initial_value="0"
            )

        self._playing = False
        self._playback_thread: threading.Thread | None = None

        @self._play_button.on_click
        def _on_play(_: viser.GuiEvent) -> None:
            if self._playing:
                self._playing = False
                self._play_button.name = "Play"
            else:
                self._follow_live.value = False
                self._playing = True
                self._play_button.name = "Pause"
                if (
                    self._playback_thread is None
                    or not self._playback_thread.is_alive()
                ):
                    self._playback_thread = threading.Thread(
                        target=self._playback_loop, daemon=True
                    )
                    self._playback_thread.start()

        @self.env_slider.on_update
        def _on_env(_: viser.GuiEvent) -> None:
            self._selected_env = int(self.env_slider.value)
            if self._record_env_ids:
                self._env_id_text.value = str(self._record_env_ids[self._selected_env])
            idx = int(self._frame_slider.value)
            if 0 <= idx < len(self._frames):
                self._render_frame(idx)

        @self._collision_checkbox.on_update
        def _on_collision(_: viser.GuiEvent) -> None:
            set_collision_geom_groups_visible(
                self._scene, self.env.sim.mj_model, self._collision_checkbox.value
            )
            if self._frames:
                idx = int(self._frame_slider.value)
                if 0 <= idx < len(self._frames):
                    self._render_frame(idx)
            else:
                self._server.flush()

        @self._frame_slider.on_update
        def _on_frame(_: viser.GuiEvent) -> None:
            idx = int(self._frame_slider.value)
            if 0 <= idx < len(self._frames):
                self._render_frame(idx)

        show_phase_legend(self.env, self._server)

    def _playback_loop(self) -> None:
        while self._playing:
            fps = self._playback_speed.value
            idx = int(self._frame_slider.value)
            next_idx = idx + 1
            if next_idx >= len(self._frames):
                self._playing = False
                self._play_button.name = "Play"
                break
            self._frame_slider.value = next_idx
            self._render_frame(next_idx)
            time.sleep(1.0 / fps)

    # -- Rendering ---------------------------------------------------------

    def _render_frame(self, idx: int) -> None:
        """Push a stored frame into the visualizer for the selected env."""
        frame = self._frames[idx]
        e = self._selected_env
        env_id = self._record_env_ids[e] if self._record_env_ids else e
        xpos = frame.xpos[e : e + 1]
        xmat = frame.xmat[e : e + 1]
        mocap_pos = frame.mocap_pos[e : e + 1]
        mocap_quat = frame.mocap_quat[e : e + 1]
        qpos = frame.qpos[e : e + 1]
        qvel = frame.qvel[e : e + 1]
        arrows = frame.arrows[e] if e < len(frame.arrows) else []
        cylinders = frame.cylinders[e] if e < len(frame.cylinders) else []
        spheres = frame.spheres[e] if e < len(frame.spheres) else []

        wp = _RecordedWpData(
            xpos=xpos,
            xmat=xmat,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            qpos=qpos,
            qvel=qvel,
        )

        self._scene.env_idx = env_id
        self._scene.show_all_envs = False
        self._scene.show_only_selected = False
        self._scene.clear()

        for start, end, color, width in arrows:
            self._scene.add_arrow(start, end, color=color, width=width)
        for start, end, radius, color in cylinders:
            self._scene.add_cylinder(start, end, radius=radius, color=color)
        for center, radius, color in spheres:
            self._scene.add_sphere(center, radius, color=color)

        update_all_groups(self._scene, wp, env_idx=0)
        if frame.geom_size is not None and frame.geom_pos is not None:
            try:
                rebuild_changed_geom_handles(
                    self._scene,
                    self._handle_geom_mapping,
                    self._original_geom_size,
                    self._original_geom_pos,
                    frame.geom_size[e],
                    frame.geom_pos[e],
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("rebuild_changed_geom_handles failed: %s", exc)

    def stop(self) -> None:
        """Best-effort shutdown of the viser server."""
        self._playing = False
        try:
            self._server.stop()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Recording wrapper
# ---------------------------------------------------------------------------


class RecordingEnvWrapper:
    """Transparent wrapper that calls ``viz.record()`` after each ``step()``.

    Both reads (``__getattr__``) and writes (``__setattr__``) for non-private
    attributes are forwarded to the inner env. This matters because
    :func:`mjlab_franka.algos.ppo_skrl._adapt_env_spaces_in_place` mutates
    ``single_observation_space`` / ``action_space`` etc., and skrl's IsaacLab
    wrapper later reads those *from the inner env* via ``self._unwrapped``.
    Forwarding writes keeps both views in sync.
    """

    _OWN_ATTRS = frozenset({"_env", "_viz"})

    def __init__(self, env: ManagerBasedRlEnv, visualizer: TrainingVisualizer) -> None:
        assert visualizer.env is env, (
            "TrainingVisualizer must be created with the same env"
        )
        # Bypass __setattr__ forwarding for our own attrs.
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_viz", visualizer)

    def step(self, actions: Any) -> Any:
        result = self._env.step(actions)
        try:
            self._viz.record()
        except Exception as exc:  # noqa: BLE001
            log.warning("TrainingVisualizer.record failed: %s", exc)
        return result

    def __getattr__(self, name: str) -> Any:
        # Only called when name is not on the wrapper itself.
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._OWN_ATTRS or name.startswith("__"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)
