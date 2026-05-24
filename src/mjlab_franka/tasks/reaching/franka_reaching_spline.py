"""Franka trajectory-tracking task using quintic-spline joint position actions."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.mdp import illegal_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_franka.robots.franka import (
    EE_SITE_NAME,
    FRANKA_ENTITY_NAME,
    get_franka_robot_cfg,
)
from mjlab_franka.tasks.reaching.mdp import (
    QuinticSplineJointPositionActionCfg,
    TrajectoryCommandCfg,
    ee_plane_perpendicular,
    ee_position_l2_metric,
    ee_position_tracking,
    ee_target_offset_w,
    ee_target_velocity_w,
    ee_velocity_tracking,
    ee_velocity_w,
    ee_z_axis_w,
    trajectory_plane_normal_w,
    wrist_position_tracking,
)
from mjlab_franka.tasks.registry import register_task


@register_task("franka_trajectory_track_spline")
def make_franka_reach_spline_env_cfg(
    num_envs: int = 4096, play: bool = False
) -> ManagerBasedRlEnvCfg:
    """Build the Franka trajectory-tracking env with quintic spline actions."""
    robot_site_cfg = SceneEntityCfg(FRANKA_ENTITY_NAME, site_names=(EE_SITE_NAME,))
    robot_joints_cfg = SceneEntityCfg(FRANKA_ENTITY_NAME, joint_names=(".*",))
    robot_wrist_cfg = SceneEntityCfg(FRANKA_ENTITY_NAME, body_names=("link7",))

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "ee_target_offset": ObservationTermCfg(
            func=ee_target_offset_w,
            params={"asset_cfg": robot_site_cfg, "command_name": "trajectory"},
        ),
        "ee_target_vel": ObservationTermCfg(
            func=ee_target_velocity_w,
            params={"command_name": "trajectory"},
        ),
        "traj_plane_normal": ObservationTermCfg(
            func=trajectory_plane_normal_w,
            params={"command_name": "trajectory"},
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    critic_terms = {
        **actor_terms,
        "ee_vel": ObservationTermCfg(
            func=ee_velocity_w,
            params={"asset_cfg": robot_site_cfg},
        ),
        "ee_z_axis": ObservationTermCfg(
            func=ee_z_axis_w,
            params={"command_name": "trajectory"},
        ),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=actor_terms, concatenate_terms=True, enable_corruption=True
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False
        ),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": QuinticSplineJointPositionActionCfg(
            entity_name=FRANKA_ENTITY_NAME,
            actuator_names=(".*",),
            include_velocity=False,
            pos_scale=0.1,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "trajectory": TrajectoryCommandCfg(
            entity_name=FRANKA_ENTITY_NAME,
            ee_site_name=EE_SITE_NAME,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
        )
    }

    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.5, 0.5),
                "velocity_range": (-0.5, 0.5),
                "asset_cfg": robot_joints_cfg,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg(FRANKA_ENTITY_NAME),
                "bias_range": (-0.015, 0.015),
            },
        ),
    }

    rewards = {
        "track_ee_pos": RewardTermCfg(
            func=ee_position_tracking,
            weight=5.0,
            params={
                "command_name": "trajectory",
                "asset_cfg": robot_site_cfg,
                "std": 0.05,
            },
        ),
        "track_ee_pos_coarse": RewardTermCfg(
            func=ee_position_tracking,
            weight=1.0,
            params={
                "command_name": "trajectory",
                "asset_cfg": robot_site_cfg,
                "std": 0.2,
            },
        ),
        "track_ee_vel": RewardTermCfg(
            func=ee_velocity_tracking,
            weight=0.5,
            params={
                "command_name": "trajectory",
                "asset_cfg": robot_site_cfg,
                "std": 0.5,
            },
        ),
        "ee_plane_perpendicular": RewardTermCfg(
            func=ee_plane_perpendicular,
            weight=1.0,
            params={"command_name": "trajectory"},
        ),
        "wrist_position": RewardTermCfg(
            func=wrist_position_tracking,
            weight=2.0,
            params={
                "command_name": "trajectory",
                "asset_cfg": robot_wrist_cfg,
                "std": 0.05,
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "arm_ground_contact": TerminationTermCfg(
            func=illegal_contact,
            params={"sensor_name": "arm_ground_contact"},
        ),
        "self_collision": TerminationTermCfg(
            func=illegal_contact,
            params={"sensor_name": "self_collision"},
        ),
    }

    metrics: dict[str, MetricsTermCfg] = {
        "ee_traj_distance_m": MetricsTermCfg(
            func=ee_position_l2_metric,
            params={"command_name": "trajectory", "asset_cfg": robot_site_cfg},
        ),
    }

    arm_ground_cfg = ContactSensorCfg(
        name="arm_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"link1",
            entity=FRANKA_ENTITY_NAME,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree", pattern="link0", entity=FRANKA_ENTITY_NAME
        ),
        secondary=ContactMatch(
            mode="subtree", pattern="link0", entity=FRANKA_ENTITY_NAME
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={FRANKA_ENTITY_NAME: get_franka_robot_cfg()},
        sensors=(arm_ground_cfg, self_collision_cfg),
        num_envs=num_envs,
        env_spacing=2.0,
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=FRANKA_ENTITY_NAME,
            body_name="link0",
            distance=1.8,
            elevation=-15.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=20,
            njmax=200,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=15.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False
        # Disable timed resampling; play-interactive drives it from the GUI.
        cfg.commands["trajectory"].resampling_time_range = (1.0e9, 1.0e9)

    return cfg
