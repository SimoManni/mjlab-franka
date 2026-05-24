# mjlab_franka — Franka trajectory tracking with mjlab + viser

Train a Franka Panda arm to track parametric end-effector trajectories
(line, circle, square, figure-8, sinusoid) sampled in randomly-oriented planes
within the arm's reachable workspace. Built on:

- **[mjlab 1.3.0](https://pypi.org/project/mjlab/)** — Isaac-Lab-style RL framework on MuJoCo Warp.
- **[skrl](https://skrl.readthedocs.io/)** — PPO (asymmetric actor-critic).
- **[viser](https://viser.studio/)** — browser-based 3D viewer (live training + post-training playback).
- **[Hydra](https://hydra.cc/)** — structured config & CLI overrides.

The Franka asset is **not** in mjlab's bundled asset zoo (only G1/Go1/YAM ship);
we fetch it from MuJoCo Menagerie via a small script.

## Setup

```bash
# 1. Install dependencies (cu128 torch wheel + warp 1.12.x).
uv sync

# 2. Fetch Franka XML + meshes from MuJoCo Menagerie.
uv run python scripts/fetch_franka.py
```

> **Note** — `PYTORCH_JIT=0` is set automatically by `mjlab_franka._compat`,
> which is the first import in both `train.py` and `play.py`. You only need
> to set it manually if you invoke `python -c "import mjlab"` directly
> without going through the project entry points. This works around a
> `torch.jit.script` segfault in `mjlab/utils/lab_api/math.py:matrix_from_quat`.

## Train

```bash
# Full training (4096 envs, 30k timesteps, viser at http://localhost:8080):
uv run train experiment=franka_trajectory_track_ppo

# Quick smoke test:
uv run train experiment=franka_trajectory_track_ppo \
    env.num_envs=64 algo.timesteps=200 visualize=false

# Common overrides:
uv run train experiment=franka_trajectory_track_ppo \
    env.num_envs=2048 \
    algo.learning_rate=5e-4 \
    algo.timesteps=50000
```

Checkpoints go to `~/.cache/mjlab_franka/runs/<experiment>/checkpoints/`.

## Visualize / playback

```bash
# Visualize the task with a random action policy (no checkpoint needed):
uv run play --task franka_trajectory_track_spline --agent random

# Replay a trained policy:
uv run play --task franka_trajectory_track_spline --agent trained \
    --checkpoint ~/.cache/mjlab_franka/runs/franka_trajectory_track_ppo/checkpoints/agent_30000.pt
```

The viser server listens on **http://localhost:8080**. Open in any browser to
see the Franka, the sampled trajectory polyline (cylinders), and a sphere at
the current trajectory target.

### Interactive playback

`play-interactive` (note the **hyphen** — the entry point is `play-interactive`,
not `play_interactive`) launches a trained policy with extra GUI controls in
viser: shape selection, pose randomization, period/size sliders, and metric
recording + plotting. It defaults to the bundled checkpoint at
[checkpoints/policy.pt](checkpoints/policy.pt).

```bash
# Default: spline task, bundled checkpoint.
uv run play-interactive

# Use a custom local checkpoint:
uv run play-interactive --checkpoint /path/to/agent.pt

# Or pull from a wandb run (same syntax as `play`):
uv run play-interactive --checkpoint <entity>/<project>/<run_id> \
    --wandb-filename best
```

The only task currently registered is `franka_trajectory_track_spline`.

## Project layout

```
src/mjlab_franka/
├── _compat.py                       # sets PYTORCH_JIT=0
├── algos/
│   ├── ppo_skrl.py                  # skrl PPO entry point + Gaussian/Det networks
│   └── registry.py
├── config/
│   ├── base.py                      # TrainConfig / EnvConfig / TrackerConfig
│   └── experiments/
│       └── franka_trajectory_track_ppo.py
├── core/
│   ├── training_visualizer.py       # background viser scene for live training
│   ├── visualizer.py                # base viser PlayApp
│   └── interactive_play.py          # extends PlayApp with GUI controls + metrics
├── envs/
│   ├── factory.py                   # make_env(task, num_envs, ...)
│   └── viewer_wrapper.py            # adds get_observations() for the viewer
├── robots/franka/
│   ├── franka_constants.py          # EntityCfg + 7 BuiltinPositionActuatorCfg
│   └── xmls/                        # panda_nohand.xml + meshes (fetched)
├── tasks/
│   ├── registry.py                  # @register_task decorator
│   └── reaching/
│       ├── franka_reaching_spline.py    # task config factory
│       └── mdp/                         # commands / observations / rewards / actions
├── train.py
├── play.py
└── play_interactive.py
```

## Trajectory shapes

Per-episode, each parallel env independently samples:

- **shape** ∈ {`line`, `circle`, `square`, `figure_8`, `sinusoid`}
- **plane center** in a reachable cuboid (x ∈ [0.35, 0.6], y ∈ [-0.25, 0.25], z ∈ [0.25, 0.6])
- **plane orientation**: uniformly random `SO(3)` via QR-decomposition
- **size** ∈ [0.08, 0.18] m
- **period** ∈ [4, 10] s

The command tensor is `[target_pos_w (3), target_vel_w (3)]` per env.

## Known issues

- **torch.jit.script segfault**: workaround via `PYTORCH_JIT=0` (set in `_compat.py`).
- **warp-lang ≥ 1.13** removed `wp.context.runtime` → pinned `warp-lang>=1.12,<1.13`.
- Live training visualizer is a single-camera snapshot at ~30 Hz; no checkpoint scrubbing UI (use `play` on a saved checkpoint for that).

## Development

Formatting and linting are enforced by [ruff](https://docs.astral.sh/ruff/) via a
[pre-commit](https://pre-commit.com/) hook. After cloning:

```bash
uv sync                       # pulls pre-commit into the dev group
uv run pre-commit install     # registers .git/hooks/pre-commit
```

The hook runs `ruff` (lint with `--fix`) and `ruff-format` on every staged
file. To check the whole repo at once:

```bash
uv run pre-commit run --all-files
```

Tests live under `tests/` and run with `pytest`:

```bash
uv run pytest -q
```

## Logging with WandB (personal account, project-local)

> **Privacy** — your wandb API key, personal git identity, and any SSH key paths
> stay **local to this repo / machine** and are never pushed to GitHub. See
> [Privacy & secrets](#privacy--secrets) below for what is gitignored and what
> is not.

The training loop logs episode rewards, raw reward outputs (via the
`MetricsManager`), user-defined metric terms (e.g. `Metrics / ee_traj_distance_m`),
PPO loss stats, and uploads every checkpoint (`model_*.pt`, `best_agent.pt`)
to the active wandb run as soon as it is written.

To use a **personal** wandb account on this machine without touching the
existing global login, write the credentials to a project-local `.env`
(gitignored) at the repo root:

```bash
# Get a key from https://wandb.ai/authorize
cat > .env <<'EOF'
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_ENTITY=your-personal-username
EOF
```

`mjlab_franka._envfile` is imported from `_compat.py` and loads `.env` into
`os.environ` before any `import wandb`, so the global `~/.netrc` is left alone.

Enable wandb in the experiment / on the CLI:

```bash
uv run train experiment=franka_trajectory_track_ppo \
    tracker.wandb.project=mjlab_franka \
    tracker.checkpoint.save_interval=2000

# Disable wandb explicitly:
uv run train experiment=franka_trajectory_track_ppo tracker.wandb=null
```

### Resume from a wandb run

```bash
# Resume training, pulling the latest model_*.pt from the run:
uv run train experiment=franka_trajectory_track_ppo \
    resume=<entity>/<project>/<run_id> wandb_filename=latest

# Warm-start network weights only (no optimizer / step counter):
uv run train experiment=franka_trajectory_track_ppo \
    init_from=<entity>/<project>/<run_id> wandb_filename=best_agent.pt
```

### Play from a wandb checkpoint

```bash
uv run play --task franka_trajectory_track --agent trained \
    --checkpoint <entity>/<project>/<run_id> --wandb-filename best
```

Wandb files are cached under `~/.cache/mjlab_franka/checkpoints/<run_id>/`.

## Per-repo personal git identity

To use a **personal** git identity for this repo without touching your global
`~/.gitconfig`, run the helper script once:

```bash
scripts/setup_personal_git.sh "Your Name" you@personal.example ~/.ssh/id_ed25519_personal
```

This writes `user.name`, `user.email`, and `core.sshCommand` via
`git config --local`, so only this repo uses the personal identity / key.

Alternative — add a Host alias in `~/.ssh/config` and use it in the remote URL:

```sshconfig
# ~/.ssh/config
Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal
    IdentitiesOnly yes
```

```bash
git remote set-url origin git@github-personal:<user>/<repo>.git
```

`git config --local` writes to `.git/config`, which is **inside `.git/` and
therefore never tracked by git** — your personal name, email, and SSH key path
cannot be pushed to GitHub by accident.

## Privacy & secrets

This repo is set up so that nothing personal leaks to GitHub. What stays
strictly local:

| What | Where it lives | Why it can't be pushed |
| --- | --- | --- |
| Wandb API key, entity | `.env` (project root) | Listed in `.gitignore` (`.env`, `.env.*`, `.netrc`) |
| Per-repo git identity | `.git/config` | The whole `.git/` directory is never tracked by git |
| Per-repo SSH key path | `.git/config` (via `core.sshCommand`) | Same as above |
| SSH private keys | `~/.ssh/...` (outside the repo) | Not inside the repo; also gitignored if dropped here |
| Wandb run dirs / cached checkpoints | `wandb/`, `~/.cache/mjlab_franka/` | `wandb/` and `.cache/` are gitignored |

### Quick checks before you push

```bash
# 1. Confirm .env is ignored (should print "::: .env").
git check-ignore -v .env

# 2. Confirm no secrets are staged.
git diff --cached | grep -iE 'WANDB_API_KEY|api_key|password|BEGIN .* PRIVATE KEY' && \
    echo "!! secret-looking string is staged !!" || echo "OK"

# 3. Inspect the per-repo git config (never tracked, just for your eyes).
git config --local --list | grep -E 'user\.|core\.sshCommand'

# 4. Confirm the remote uses the right host alias.
git remote -v
```

### Setting up secrets (one-time, per clone)

```bash
# Wandb: get your key from https://wandb.ai/authorize
cp .env.example .env
$EDITOR .env            # fill in WANDB_API_KEY + WANDB_ENTITY

# Git identity + SSH key (writes to .git/config only):
scripts/setup_personal_git.sh "Your Name" you@personal.example ~/.ssh/id_ed25519_personal
```

If you accidentally commit `.env` or a key, **rotate the credential immediately**
(re-issue the wandb key at https://wandb.ai/authorize, regenerate the SSH key)
and rewrite history with `git filter-repo` before pushing.
