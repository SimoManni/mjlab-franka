from mjlab_franka.tasks.manipulation.mdp.actions import QuinticSplineArmAndGripperActionCfg
from mjlab_franka.tasks.manipulation.mdp.observations import (
    object_pos_local,
    object_rotation_matrix_local,
    object_vel_local,
    target_2D_location,
)

from mjlab_franka.tasks.manipulation.mdp.commands import Target2DGroundCommandCfg, Target2DGroundCommand
from mjlab_franka.tasks.manipulation.mdp.events import reset_robot_obj_scene
from mjlab_franka.tasks.manipulation.mdp.terminations import obj_at_goal

from mjlab_franka.tasks.manipulation.mdp.rewards import (
    EntityEntityDistanceImprovement,
    EntityCommandDistanceImprovement,
    finger_object_contact_reward,
    termination_triggered_reward

)

from mjlab_franka.tasks.manipulation.mdp.metrics import (
    obj_to_target_distance,
    gripper_to_obj_distance,
)