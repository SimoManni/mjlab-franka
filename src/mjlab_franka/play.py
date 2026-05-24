"""Playback / visualization entry point.

Usage:
    # Visualize the task with a random action policy (no checkpoint needed):
    uv run play --task franka_trajectory_track_spline --agent random

    # Playback a trained policy from a skrl checkpoint:
    uv run play --task franka_trajectory_track_spline \\
        --checkpoint ~/.cache/mjlab_franka/runs/franka_trajectory_track_ppo/checkpoints/agent_15000.pt
"""

from __future__ import annotations

# Set PYTORCH_JIT=0 before importing torch/mjlab.
import mjlab_franka._compat  # noqa: F401

import argparse
import logging
from typing import Callable

import torch

# Imported for their registration decorators.
import mjlab_franka.algos  # noqa: F401
import mjlab_franka.tasks  # noqa: F401
from mjlab_franka.envs.factory import make_env
from mjlab_franka.envs.viewer_wrapper import ViewerEnvWrapper

log = logging.getLogger(__name__)


def _zero_policy(
    action_shape: tuple[int, ...], device: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    zeros = torch.zeros(action_shape, device=device)

    def policy(obs: torch.Tensor) -> torch.Tensor:
        del obs
        return zeros

    return policy


def _random_policy(
    action_shape: tuple[int, ...], device: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    def policy(obs: torch.Tensor) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=device) - 1

    return policy


def _skrl_policy_from_checkpoint(
    checkpoint_source: str,
    obs_space,
    action_space,
    device: str,
    wandb_filename: str = "best",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a deterministic policy callable from a skrl PPO checkpoint.

    ``checkpoint_source`` may be a local file/dir, a wandb run string
    (``entity/project/run_id``), or a full ``https://wandb.ai/...`` URL.
    """
    from mjlab_franka.algos.ppo_skrl import GaussianPolicy, PPOSkrlConfig
    from mjlab_franka.core.checkpoint import load_checkpoint
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    cfg = PPOSkrlConfig()
    state = load_checkpoint(
        checkpoint_source, device=device, checkpoint_file=wandb_filename
    )
    # Checkpoints written by our CheckpointSaver use {"step": .., "agent": {...}, ...}.
    # Legacy skrl checkpoints just store {"policy": state_dict, ...}.
    agent_state = state.get("agent", state)
    policy_state = agent_state["policy"] if "policy" in agent_state else agent_state

    net = GaussianPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
        hidden_dims=cfg.policy_hidden_dims,
        activation=cfg.activation,
    )
    net.load_state_dict(policy_state)
    net.to(device)
    net.eval()

    # Rebuild the observation preprocessor from the checkpoint. Training used
    # RunningStandardScaler on the obs, so the policy was always fed zero-mean /
    # unit-variance inputs. Skipping this gives the policy raw obs at inference
    # time, which collapses outputs to near zero — the arm appears frozen.
    obs_preprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None
    obs_preproc_state = agent_state.get("observation_preprocessor")
    if obs_preproc_state is not None and cfg.state_preprocessor:
        obs_preprocessor = RunningStandardScaler(size=obs_space, device=device)
        obs_preprocessor.load_state_dict(obs_preproc_state)
        obs_preprocessor.eval()

    @torch.no_grad()
    def policy(obs: torch.Tensor) -> torch.Tensor:
        if obs_preprocessor is not None:
            obs = obs_preprocessor(obs, train=False)
        # Use the mean for deterministic playback.
        return torch.tanh(net.mean_layer(net.net(obs)))

    return policy


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Playback / visualize a task.")
    parser.add_argument(
        "--task",
        default="franka_trajectory_track_spline",
        help="Task name from the registry.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs to simulate."
    )
    parser.add_argument("--device", default="cuda:0", help="Device.")
    parser.add_argument("--seed", type=int, default=0, help="Seed.")
    parser.add_argument(
        "--agent",
        choices=["zero", "random", "trained"],
        default="random",
        help="Policy source.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint source: local file or directory, wandb URL, or "
            "entity/project/run_id (downloaded into ~/.cache/mjlab_franka/checkpoints/)."
        ),
    )
    parser.add_argument(
        "--wandb-filename",
        type=str,
        default="best",
        help=(
            "Filename to pull from a wandb run or checkpoint directory: 'best', "
            "'latest', or a specific name like 'model_00030000.pt'."
        ),
    )
    parser.add_argument(
        "--viewer", choices=["viser", "native"], default="viser", help="Viewer backend."
    )
    parser.add_argument(
        "--viz-envs",
        type=int,
        default=4,
        help=(
            "Max number of envs to render in the viser scene (sim still runs --num-envs)."
            " The UI exposes an `Env #` slider to switch which one is shown."
        ),
    )
    args = parser.parse_args()

    if args.agent == "trained" and args.checkpoint is None:
        parser.error("--checkpoint is required when --agent=trained")

    env = make_env(
        task=args.task,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        play=True,
    )

    # Adapt spaces (gymnasium) — needed if we build a policy from checkpoint.
    from mjlab_franka.algos.ppo_skrl import _adapt_env_spaces_in_place

    _adapt_env_spaces_in_place(env)

    action_shape = (env.num_envs, *env.single_action_space.shape)

    if args.agent == "zero":
        policy = _zero_policy(action_shape, args.device)
    elif args.agent == "random":
        policy = _random_policy(action_shape, args.device)
    else:
        policy = _skrl_policy_from_checkpoint(
            args.checkpoint,
            obs_space=env.single_observation_space["policy"],
            action_space=env.single_action_space,
            device=args.device,
            wandb_filename=args.wandb_filename,
        )

    viewer_env = ViewerEnvWrapper(env, obs_group="policy")

    if args.viewer == "viser":
        from mjlab_franka.core.visualizer import PlayApp, PlayAppConfig

        log.info("Launching viser viewer at http://localhost:8080")
        # Only render up to `--viz-envs` envs in the scene (sim still runs all
        # `--num-envs`). Mirrors aloy: batched sim, single env rendered, with a
        # slider in the viser GUI to switch which one is shown.
        app_cfg = PlayAppConfig(
            port=8080,
            obs_group="policy",
            num_visible_envs=min(args.viz_envs, args.num_envs),
        )
        PlayApp(app_cfg, env, policy).run()
    else:
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(viewer_env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
