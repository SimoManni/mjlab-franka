"""Algorithm registry — maps algo names to entry points.

Each registered algo class must expose:
    - ``train(train_cfg, env) -> None``
    - ``build_inference_policy(algo_cfg, obs_space, action_space, agent_state) -> nn.Module``
"""

from __future__ import annotations

from typing import Any, Callable

algo_registry: dict[str, Any] = {}


def register_algo(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        algo_registry[name] = cls
        return cls

    return decorator
