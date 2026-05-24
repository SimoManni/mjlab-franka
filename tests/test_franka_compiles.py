"""Smoke-test the Franka asset config."""

import mujoco

from mjlab.entity import Entity

from mjlab_franka.robots.franka import EE_SITE_NAME, get_franka_robot_cfg


def test_franka_compiles_with_mjlab_actuators() -> None:
    cfg = get_franka_robot_cfg()
    entity = Entity(cfg)
    model = entity.compile()

    assert isinstance(model, mujoco.MjModel)
    # 7 arm joints.
    assert model.nv == 7
    assert model.nu == 7
    # End-effector site must exist (used by the trajectory-tracking task).
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    assert site_id >= 0, f"Site '{EE_SITE_NAME}' missing from Franka model"
