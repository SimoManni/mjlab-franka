"""Verify the trajectory command is locked once per episode.

The TrajectoryCommand should sample its shape (and all other parameters) at
episode reset and keep them stable for the entire episode duration. This is
enforced by ``TrajectoryCommandCfg.resampling_time_range=(1e9, 1e9)`` so that
mjlab's CommandManager only resamples via ``CommandTerm.reset(env_ids)``.
"""

import torch

from mjlab_franka.envs import make_env


def test_trajectory_shape_constant_within_episode() -> None:
    env = make_env(
        task="franka_trajectory_track_spline",
        num_envs=4,
        device="cuda:0",
        seed=0,
    )
    try:
        env.reset()
        traj = env.command_manager.get_term("trajectory")
        initial_shape_id = traj._shape_id.clone()
        initial_size = traj._size.clone()
        initial_center = traj._center.clone()
        initial_period = traj._period.clone()

        action_shape = env.single_action_space.shape
        assert action_shape is not None
        actions = torch.zeros(4, *action_shape, device=env.device)

        # Track which envs have been reset (e.g. via a self-collision or
        # ground-contact termination): once reset, their trajectory params are
        # legitimately re-sampled, so they should be excluded from the lock check.
        ever_reset = torch.zeros(4, dtype=torch.bool, device=env.device)

        # Step well past any plausible per-step resampling window without
        # crossing the episode length (~15 s / dt=0.02 s = 750 steps).
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(actions)
            ever_reset |= terminated | truncated
            alive = ~ever_reset
            assert torch.equal(traj._shape_id[alive], initial_shape_id[alive])
            assert torch.equal(traj._size[alive], initial_size[alive])
            assert torch.equal(traj._center[alive], initial_center[alive])
            assert torch.equal(traj._period[alive], initial_period[alive])

        # Sanity: at least one env must have survived the whole window, else the
        # test isn't actually exercising the lock.
        assert (~ever_reset).any()
    finally:
        env.close()
