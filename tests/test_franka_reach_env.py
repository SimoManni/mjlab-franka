"""Smoke-test the Franka trajectory-tracking environment."""

import torch

from mjlab_franka.envs import make_env


def test_franka_reach_env_builds_and_steps() -> None:
    env = make_env(
        task="franka_trajectory_track_spline",
        num_envs=4,
        device="cuda:0",
        seed=0,
    )
    try:
        obs, _ = env.reset()
        assert "policy" in obs
        policy_obs = obs["policy"]
        assert policy_obs.ndim == 2
        assert policy_obs.shape[0] == 4
        assert torch.isfinite(policy_obs).all()

        action_shape = env.single_action_space.shape
        assert action_shape is not None
        actions = torch.zeros(4, *action_shape, device=env.device)
        for _ in range(50):
            obs, rewards, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(rewards).all()
            assert terminated.shape == (4,)
            assert truncated.shape == (4,)
    finally:
        env.close()
