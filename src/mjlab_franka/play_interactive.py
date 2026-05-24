"""Interactive playback entry point for the Franka trajectory-track task.

Runs a trained policy and launches a viser GUI with shape selection, pose
randomization, period/size sliders, and per-step metric recording + plotting.

Usage:
    # Use the default bundled checkpoint at checkpoints/policy.pt
    uv run play-interactive

    # Override with another local file
    uv run play-interactive --checkpoint /path/to/agent.pt

    # Or pull from a wandb run (same syntax as `play`)
    uv run play-interactive --checkpoint entity/project/run_id
"""

from __future__ import annotations

# Set PYTORCH_JIT=0 before importing torch/mjlab.
import mjlab_franka._compat  # noqa: F401

import argparse
import logging
from pathlib import Path

# Imported for their registration decorators.
import mjlab_franka.algos  # noqa: F401
import mjlab_franka.tasks  # noqa: F401
from mjlab_franka.envs.factory import make_env

log = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "policy.pt"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Interactive playback of a trained Franka trajectory-track policy."
    )
    parser.add_argument(
        "--task",
        default="franka_trajectory_track_spline",
        help="Task name from the registry.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of envs to simulate. Forced to 1 for interactive metrics.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device.")
    parser.add_argument("--seed", type=int, default=0, help="Seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint source: local file or directory, wandb URL, or "
            "entity/project/run_id. Defaults to "
            f"{_DEFAULT_CHECKPOINT} if it exists."
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
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument(
        "--log-root",
        type=str,
        default=None,
        help=(
            "Directory under which per-session play_logs/<timestamp>/save_<N>/ "
            "folders are created. Defaults to ./play_logs."
        ),
    )
    args = parser.parse_args()

    if args.num_envs != 1:
        log.warning("Forcing --num-envs=1 for interactive metrics integrity.")
        args.num_envs = 1

    # Resolve checkpoint.
    checkpoint = args.checkpoint
    if checkpoint is None:
        if _DEFAULT_CHECKPOINT.exists():
            checkpoint = str(_DEFAULT_CHECKPOINT)
            log.info("Using default local checkpoint: %s", checkpoint)
        else:
            parser.error(
                "No --checkpoint provided and default does not exist at "
                f"{_DEFAULT_CHECKPOINT}. Drop a .pt there or pass --checkpoint."
            )

    env = make_env(
        task=args.task,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        play=True,
    )

    # Adapt spaces for skrl's policy / wrappers.
    from mjlab_franka.play import _skrl_policy_from_checkpoint
    from mjlab_franka.algos.ppo_skrl import _adapt_env_spaces_in_place

    _adapt_env_spaces_in_place(env)

    policy = _skrl_policy_from_checkpoint(
        checkpoint,
        obs_space=env.single_observation_space["policy"],
        action_space=env.single_action_space,
        device=args.device,
        wandb_filename=args.wandb_filename,
    )

    from mjlab_franka.core.interactive_play import InteractivePlayApp
    from mjlab_franka.core.visualizer import PlayAppConfig

    log.info("Launching viser viewer at http://localhost:%d", args.port)
    app_cfg = PlayAppConfig(
        port=args.port,
        obs_group="policy",
        num_visible_envs=1,
    )
    log_root = Path(args.log_root) if args.log_root else None
    InteractivePlayApp(app_cfg, env, policy, log_root=log_root).run()

    env.close()


if __name__ == "__main__":
    main()
