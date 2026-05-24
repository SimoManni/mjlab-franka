"""Checkpoint load/save/resume utilities.

Ported from aloy/core/checkpoint.py with the aloy.utils.wandb helper inlined.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from wandb.apis.public import Api
from wandb.apis.public import Run as ApiRun

log = logging.getLogger(__name__)

CHECKPOINT_PAD_WIDTH = 8


def checkpoint_filename(step: int) -> str:
    return f"model_{step:0{CHECKPOINT_PAD_WIDTH}d}.pt"


def save_training_state(state: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    log.debug(f"Saved training state to {path} (step {state.get('step')})")
    return path


def resume_training(
    source: str | Path,
    device: str,
    agent_modules: dict,
    wandb_filename: str = "latest",
) -> int:
    """Load training state and restore agent modules. Returns the step count."""
    ckpt = load_checkpoint(source, device=device, checkpoint_file=wandb_filename)
    if "agent" not in ckpt:
        raise ValueError(f"Checkpoint has no 'agent' key (keys: {list(ckpt.keys())}).")
    for name, module in agent_modules.items():
        if isinstance(module, torch.optim.Optimizer):
            module.load_state_dict(ckpt["agent"][name])
        else:
            module.load_state_dict(ckpt["agent"][name], strict=True)
    log.info(f"Resumed training from step {ckpt['step']}")
    return ckpt["step"]


def init_from_checkpoint(
    source: str | Path,
    device: str,
    agent_modules: dict,
    wandb_filename: str | None = None,
    *,
    strict: bool = False,
) -> None:
    """Load only network weights (no optimizer state, no step)."""
    ckpt = load_checkpoint(
        source, device=device, checkpoint_file=wandb_filename or "latest"
    )
    if "agent" not in ckpt:
        raise ValueError(f"Checkpoint has no 'agent' key (keys: {list(ckpt.keys())}).")
    loaded: list[str] = []
    for name, module in agent_modules.items():
        if isinstance(module, torch.optim.Optimizer):
            continue
        if name in ckpt["agent"]:
            result = module.load_state_dict(ckpt["agent"][name], strict=strict)
            loaded.append(name)
            if getattr(result, "missing_keys", None):
                log.warning(f"init_from '{name}': missing keys: {result.missing_keys}")
            if getattr(result, "unexpected_keys", None):
                log.warning(
                    f"init_from '{name}': unexpected keys: {result.unexpected_keys}"
                )
        else:
            log.warning(f"init_from: '{name}' not found in checkpoint, skipping.")
    log.info(f"Initialised weights from checkpoint (loaded: {loaded})")


def parse_wandb_source(source: str) -> tuple[str, str, str]:
    """Parse ``entity/project/run_id`` or a wandb URL into a 3-tuple."""
    if source.startswith("https://wandb.ai/"):
        parts = source.split("/")
        return parts[3], parts[4], parts[6]
    parts = source.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid wandb source: {source}. Expected entity/project/run_id"
        )
    return parts[0], parts[1], parts[2]


def _wandb_run_from_source(source: str) -> ApiRun:
    entity, project, run_id = parse_wandb_source(source)
    return Api().run(f"{entity}/{project}/{run_id}")


def _download_wandb_file(run: ApiRun, name: str, dest_dir: Path) -> Path:
    """Download ``name`` from ``run`` into ``dest_dir`` and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(name).name
    if target.exists():
        return target
    file = run.file(name)
    file.download(replace=True, root=str(dest_dir))
    # wandb preserves nested directories; flatten to dest_dir / basename if needed.
    nested = dest_dir / name
    if nested.exists() and nested != target:
        nested.replace(target)
    return target


def _resolve_local_checkpoint(directory: Path, checkpoint_file: str) -> Path:
    if checkpoint_file == "best":
        best = directory / "best_agent.pt"
        if best.exists():
            return best
        log.warning("best_agent.pt not found locally, falling back to latest.")
        checkpoint_file = "latest"

    if checkpoint_file == "latest":
        model_files = sorted(
            directory.glob("model_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True,
        )
        if not model_files:
            raise FileNotFoundError(f"No model_*.pt checkpoints found in {directory}")
        return model_files[0]

    resolved = directory / checkpoint_file
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def load_checkpoint(
    source: str | Path, device: str = "cpu", checkpoint_file: str = "latest"
) -> dict:
    """Load a checkpoint from a local path/dir or a wandb source."""
    source_str = str(source)
    path = Path(source_str)

    if path.is_file():
        log.info(f"Loading checkpoint: {path}")
        return torch.load(path, map_location=device, weights_only=False)

    if path.is_dir():
        resolved = _resolve_local_checkpoint(path, checkpoint_file)
        log.info(f"Loading checkpoint: {resolved}")
        return torch.load(resolved, map_location=device, weights_only=False)

    if source_str.startswith("https://wandb.ai/") or "/" in source_str:
        path = download_from_wandb(source_str, checkpoint_file=checkpoint_file)
        log.info(f"Loading checkpoint: {path}")
        return torch.load(path, map_location=device, weights_only=False)

    raise FileNotFoundError(f"Checkpoint not found: {source_str}")


def download_from_wandb(source: str, checkpoint_file: str = "latest") -> Path:
    """Download a checkpoint from a wandb run; cached under ``~/.cache/mjlab_franka/checkpoints/<run_id>/``."""
    entity, project, run_id = parse_wandb_source(source)
    run = _wandb_run_from_source(source)

    cache_dir = Path.home() / ".cache" / "mjlab_franka" / "checkpoints" / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_file not in ("best", "latest"):
        return _download_wandb_file(run, checkpoint_file, cache_dir)

    best_file: Any = None
    model_files: list[tuple[int, Any]] = []

    for file in run.files():
        if file.name.endswith("best_agent.pt"):
            best_file = file
        elif "model_" in file.name and file.name.endswith(".pt"):
            try:
                step = int(file.name.split("model_")[-1].replace(".pt", ""))
                model_files.append((step, file))
            except ValueError as e:
                raise ValueError(f"Invalid checkpoint filename: {file.name}") from e

    if checkpoint_file == "best":
        target_file = best_file
        if target_file is None and model_files:
            log.warning("best_agent.pt not found, falling back to latest model.")
            model_files.sort(key=lambda x: x[0], reverse=True)
            target_file = model_files[0][1]
    else:  # latest
        target_file = None
        if model_files:
            model_files.sort(key=lambda x: x[0], reverse=True)
            target_file = model_files[0][1]
        if target_file is None:
            target_file = best_file

    if target_file is None:
        raise FileNotFoundError(
            f"No checkpoint in wandb run {entity}/{project}/{run_id}"
        )

    log.info(
        f"Selected checkpoint file: {target_file.name} from {entity}/{project}/{run_id}"
    )
    return _download_wandb_file(run, target_file.name, cache_dir)
