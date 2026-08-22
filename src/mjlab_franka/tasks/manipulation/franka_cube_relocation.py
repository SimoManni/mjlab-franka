"""Franka trajectory-tracking task using quintic-spline joint position actions."""

from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.entity import EntityCfg
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
from mjlab_franka.tasks.manipulation.mdp import (
    QuinticSplineArmAndGripperActionCfg, 
    object_pos_local, 
    object_rotation_matrix_local, 
    object_vel_local, 
    target_2D_location, 
    Target2DGroundCommandCfg, 
    reset_robot_obj_scene, 
    obj_at_goal, 
    EntityEntityDistanceImprovement, 
    EntityCommandDistanceImprovement,
    finger_object_contact_reward,
    termination_triggered_reward,
    obj_to_target_distance,
    gripper_to_obj_distance,
)

from mjlab_franka.tasks.manipulation.phase import (
    FrankaCubePhaseCommandCfg, 
    phase_masked_reward,
    phase_masked_termination,
)
from mjlab_franka.tasks.registry import register_task


def create_cube_spec() -> mujoco.MjSpec:
    """Procedurally generates an MjSpec for a movable 4cm cube."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],  # Half-sizes: 4cm x 4cm x 4cm total dimensions
        rgba=[0.8, 0.2, 0.2, 1.0],
        mass=0.05,
    )
    return spec


# Configure the cube entity with a free floating initial state on the ground
cube_cfg = EntityCfg(
    spec_fn=create_cube_spec,
    init_state=EntityCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.02),  # Placed 0.5m forward, centered, resting flat on z=0.02
    ),
)


@register_task("franka_cube_relocation")
def make_franka_cube_relocation_env_cfg(
    num_envs: int = 4096, play: bool = False
) -> ManagerBasedRlEnvCfg:
    """Build the Franka trajectory-tracking env with quintic spline actions."""
    robot_site_cfg = SceneEntityCfg(FRANKA_ENTITY_NAME, site_names=(EE_SITE_NAME,))
    robot_joints_cfg = SceneEntityCfg(FRANKA_ENTITY_NAME, joint_names=(".*",))
    robot_wrist_cfg = SceneEntityCfg(FRANKA_ENTITY_NAME, body_names=("link7",))

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=None,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=None,
        ),
        "obj_goal_local": ObservationTermCfg(
            func=object_pos_local,
            params={"robot_cfg": robot_site_cfg, "object_cfg": SceneEntityCfg("cube")},
            noise=None, 
        ),
        "obj_rot_local": ObservationTermCfg(
            func=object_rotation_matrix_local,
            params={"robot_cfg": robot_site_cfg, "object_cfg": SceneEntityCfg("cube")},
            noise=None,
        ),
        "obj_vel_local": ObservationTermCfg(
            func=object_vel_local,
            params={"robot_cfg": robot_site_cfg, "object_cfg": SceneEntityCfg("cube")},
            noise=None,
        ),
        "target_2D_location": ObservationTermCfg(
            func=target_2D_location,
            params={"command_name": "target_2D_location"},
            noise=None,
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
    }
    critic_terms = {
        **actor_terms,
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
        "joint_pos": QuinticSplineArmAndGripperActionCfg(
            entity_name=FRANKA_ENTITY_NAME,
            actuator_names=(".*",),
            include_velocity=False,
            pos_scale=0.1,
        )
    }

    commands: dict[str, CommandTermCfg] = {
        "target_2D_location": Target2DGroundCommandCfg(),
        "phase": FrankaCubePhaseCommandCfg(),
    }

    events = {
        "reset_scene": EventTermCfg(
            func=reset_robot_obj_scene,
            mode="reset",
            params={
                "robot_cfg": SceneEntityCfg(FRANKA_ENTITY_NAME),
                "object_cfg": SceneEntityCfg("cube"),
                "object_pos_range": {
                    "x": (0.35, 0.6),
                    "y": (-0.2, 0.2),
                    "z": (0.0, 0.0),
                },
                "robot_joint_pos_range": (-0.3, 0.3),
                "robot_joint_vel_range": (-0.1, 0.1),
            },
        ),
    }

    APPROACHING = 0
    GRIPPING = 1
    MOVING = 2
    OFFLOADING = 3

    rewards = {
        # Phase 0: Approaching - Reward end-effector for moving closer to the cube
        "approach_cube": RewardTermCfg(
            func=phase_masked_reward,
            weight=3.0,
            params={
                "reward_fn_or_class": EntityEntityDistanceImprovement,
                "active_phases": [APPROACHING],
                "command_name": "phase",
                "source_cfg": robot_site_cfg,
                "target_cfg": SceneEntityCfg("cube"),
                "distance_type": "l2",
            },
        ),

        "maintain_grip": RewardTermCfg(
            func=phase_masked_reward,
            weight=4.0,
            params={
                "reward_fn_or_class": finger_object_contact_reward,
                "active_phases": [GRIPPING, MOVING],
                "command_name": "phase",
                "sensor_name": "finger_cube_contact",
            },
        ),


        # Phase 2: Moving - Reward the cube for moving closer to the 2D target location
        "move_cube_to_target": RewardTermCfg(
            func=phase_masked_reward,
            weight=5.0,
            params={
                "reward_fn_or_class": EntityCommandDistanceImprovement,
                "active_phases": [MOVING],
                "command_name": "phase",
                "asset_cfg": SceneEntityCfg("cube"),
                "command_name_target": "target_2D_location",  # maps internally depending on class setup
                "distance_type": "xy",
            },
        ),

        # Phase 3: Offloading - Keep the cube stable or close out final target accuracy
        "release_cube": RewardTermCfg(
            func=phase_masked_reward,
            weight=-4.0,  # Negative weight to penalize holding onto the object during offloading
            params={
                "reward_fn_or_class": finger_object_contact_reward,
                "active_phases": [OFFLOADING],
                "command_name": "phase",
                "sensor_name": "finger_cube_contact",
            },
        ),

        "success_reward": RewardTermCfg(
            func=termination_triggered_reward,
            weight=50.0,
            params={
                "termination_name": "success",
            },
        ),
        "self_collision_penalty": RewardTermCfg(
            func=termination_triggered_reward,
            weight=-20.0,
            params={
                "termination_name": "self_collision",
            },
        ),
        "arm_ground_contact_penalty": RewardTermCfg(
            func=termination_triggered_reward,
            weight=-20.0,
            params={
                "termination_name": "arm_ground_contact",
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
        "success": TerminationTermCfg(
            func=phase_masked_termination,
            params={
                "term_fn": obj_at_goal,
                "active_phases": [OFFLOADING],
                "command_name": "phase",
                "object_cfg": SceneEntityCfg("cube"),
                "command_name": "target_2D_location",
                "pos_threshold": 0.05,
            },
        ),  
    }

    metrics: dict[str, MetricsTermCfg] = {
        "obj_to_target_distance": MetricsTermCfg(
            func=obj_to_target_distance,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "command_name": "target_2D_location",
            },
        ),
        "gripper_to_obj_distance": MetricsTermCfg(
            func=gripper_to_obj_distance,
            params={
                "robot_cfg": robot_site_cfg,
                "object_cfg": SceneEntityCfg("cube"),
            },
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

    finger_cube_contact_cfg = ContactSensorCfg(
        name="finger_cube_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"right_finger|left_finger",  # Adjust pattern to match your gripper finger link names
            entity=FRANKA_ENTITY_NAME,
        ),
        secondary=ContactMatch(mode="body", pattern="cube", entity="cube"),
        fields=("found",),
        reduce="none",
        num_slots=2,
    )

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            FRANKA_ENTITY_NAME: get_franka_robot_cfg(with_gripper=True),
            "cube": cube_cfg,
        },
        sensors=(arm_ground_cfg, self_collision_cfg, finger_cube_contact_cfg),
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
