"""Project-local ``.env`` loader and wandb config directory setup.

Imported from :mod:`mjlab_franka._compat` before any wandb import so that:

1. ``WANDB_CONFIG_DIR`` points to ``<repo_root>/.wandb`` — this makes
   ``wandb login`` store credentials locally (in ``.wandb/``, which is
   gitignored) instead of the global ``~/.netrc``.

2. Any extra ``KEY=VALUE`` lines in a project-local ``.env`` file are loaded
   into ``os.environ`` (for ``WANDB_ENTITY`` overrides, etc.).

Only sets variables that are not already in ``os.environ`` (so explicit
``KEY=val uv run train ...`` invocations still win).
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path | None:
    """Walk up from this file looking for the project root (pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _setup_wandb_config_dir() -> None:
    """Point WANDB_CONFIG_DIR to a repo-local ``.wandb/`` directory.

    This makes ``wandb login`` write a local ``.netrc`` inside
    ``<repo>/.wandb/`` rather than ``~/.netrc``, keeping credentials
    per-project.  The ``.wandb/`` directory is gitignored.
    """
    if "WANDB_CONFIG_DIR" in os.environ:
        return
    root = _find_project_root()
    if root is None:
        return
    wandb_dir = root / ".wandb"
    wandb_dir.mkdir(exist_ok=True)
    os.environ["WANDB_CONFIG_DIR"] = str(wandb_dir)
    # Also redirect NETRC so wandb doesn't fall back to ~/.netrc
    local_netrc = wandb_dir / ".netrc"
    os.environ.setdefault("NETRC", str(local_netrc))


def load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from the project-local ``.env`` into ``os.environ``."""
    root = _find_project_root()
    if root is None:
        return
    path = root / ".env"
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_setup_wandb_config_dir()
load_dotenv()
