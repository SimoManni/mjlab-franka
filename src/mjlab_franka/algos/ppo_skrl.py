"""SKRL PPO algorithm with an aloy-style Tracker-driven training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch
import tqdm
from hydra.core.config_store import ConfigStore
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch import nn

from mjlab_franka.algos.registry import register_algo
from mjlab_franka.config.base import TrainConfig
from mjlab_franka.core.stats import skrl_agent_metrics
from mjlab_franka.core.tracker import Tracker

log = logging.getLogger(__name__)


def _to_gym_space(space: Any) -> gym.Space:
    """Convert mjlab's lightweight spaces (mjlab.utils.spaces) to gymnasium spaces."""
    # Already a gymnasium space.
    if isinstance(space, gym.Space):
        return space
    # mjlab Dict.
    inner = getattr(space, "spaces", None)
    if isinstance(inner, dict):
        return gym.spaces.Dict({k: _to_gym_space(v) for k, v in inner.items()})
    # mjlab Box.
    if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
        return gym.spaces.Box(
            low=space.low, high=space.high, shape=space.shape, dtype=space.dtype
        )
    raise TypeError(f"Cannot convert space of type {type(space)} to gymnasium space")


def _adapt_env_spaces_in_place(env: Any) -> None:
    """Replace mjlab spaces with gymnasium spaces so skrl wrappers can subscript them.

    Also clamps the action space to [-1, 1] (skrl/PPO expects bounded actions).
    """
    env.single_observation_space = _to_gym_space(env.single_observation_space)
    env.observation_space = _to_gym_space(env.observation_space)
    act = _to_gym_space(env.single_action_space)
    env.single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=act.shape, dtype=act.dtype
    )
    env.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(env.num_envs, *act.shape), dtype=act.dtype
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOSkrlConfig:
    """PPO config (Hydra-friendly)."""

    algo_type: str = "ppo_skrl"

    # Training schedule.
    timesteps: int = 15000
    rollouts: int = 24

    # Hyperparameters.
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    entropy_loss_scale: float = 0.0
    value_loss_scale: float = 1.0
    grad_norm_clip: float = 1.0
    learning_epochs: int = 5
    mini_batches: int = 4
    kl_threshold: float = 0.0
    time_limit_bootstrap: bool = True
    mixed_precision: bool = False

    # Networks.
    policy_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    value_hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    out_activation: str | None = "tanh"

    # Preprocessors.
    state_preprocessor: bool = True
    value_preprocessor: bool = True


cs = ConfigStore.instance()
cs.store(group="algo", name="ppo_skrl", node=PPOSkrlConfig)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_mlp(in_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(_activation(activation))
        prev = h
    return nn.Sequential(*layers), prev


class GaussianPolicy(GaussianMixin, Model):
    """Gaussian policy with state-independent log std."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device | str,
        hidden_dims: list[int],
        activation: str,
        out_activation: str | None = None,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-5.0,
            max_log_std=2.0,
        )

        in_dim = int(self.num_observations)
        self.net, last = _make_mlp(in_dim, hidden_dims, activation)
        self.mean_layer = nn.Linear(last, int(self.num_actions))
        self.out_activation = _activation(out_activation) if out_activation else None
        self.log_std_parameter = nn.Parameter(torch.zeros(int(self.num_actions)))

    def compute(
        self, inputs: dict[str, torch.Tensor], role: str
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x = self.net(inputs["observations"])
        mean = self.mean_layer(x)
        if self.out_activation is not None:
            mean = self.out_activation(mean)
        return mean, {"log_std": self.log_std_parameter}


class DeterministicValue(DeterministicMixin, Model):
    """State-value critic."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device | str,
        hidden_dims: list[int],
        activation: str,
    ) -> None:
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        in_dim = int(self.num_observations)
        self.net, last = _make_mlp(in_dim, hidden_dims, activation)
        self.value_layer = nn.Linear(last, 1)

    def compute(
        self, inputs: dict[str, torch.Tensor], role: str
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.value_layer(self.net(inputs["states"])), {}


# ---------------------------------------------------------------------------
# Algo entry point
# ---------------------------------------------------------------------------


@register_algo("ppo_skrl")
class PPOSkrl:
    """Thin SKRL PPO training entry point."""

    @staticmethod
    def train(train_cfg: TrainConfig, env: Any) -> None:
        algo: PPOSkrlConfig = train_cfg.algo  # type: ignore[assignment]
        device = train_cfg.env.device
        num_envs = env.num_envs

        # Wrap mjlab env for skrl (uses single_observation_space["policy"] / ["critic"]).
        _adapt_env_spaces_in_place(env)
        wrapped = wrap_env(env, wrapper="isaaclab")

        obs_space = wrapped.observation_space
        state_space = wrapped.state_space  # None when env has no 'critic' group
        action_space = wrapped.action_space

        # Models — asymmetric actor-critic when env exposes a 'critic' group.
        models = {
            "policy": GaussianPolicy(
                obs_space,
                action_space,
                device,
                algo.policy_hidden_dims,
                algo.activation,
                out_activation=algo.out_activation,
            ),
            "value": DeterministicValue(
                state_space if state_space is not None else obs_space,
                action_space,
                device,
                algo.value_hidden_dims,
                algo.activation,
            ),
        }

        memory = RandomMemory(
            memory_size=algo.rollouts, num_envs=wrapped.num_envs, device=device
        )

        cfg = PPO_CFG(
            rollouts=algo.rollouts,
            learning_epochs=algo.learning_epochs,
            mini_batches=algo.mini_batches,
            discount_factor=algo.discount_factor,
            gae_lambda=algo.gae_lambda,
            learning_rate=algo.learning_rate,
            grad_norm_clip=algo.grad_norm_clip,
            ratio_clip=algo.ratio_clip,
            value_clip=algo.value_clip,
            entropy_loss_scale=algo.entropy_loss_scale,
            value_loss_scale=algo.value_loss_scale,
            kl_threshold=algo.kl_threshold,
            time_limit_bootstrap=algo.time_limit_bootstrap,
            mixed_precision=algo.mixed_precision,
            state_preprocessor=RunningStandardScaler
            if algo.state_preprocessor
            else None,
            state_preprocessor_kwargs=(
                {
                    "size": state_space if state_space is not None else obs_space,
                    "device": device,
                }
                if algo.state_preprocessor
                else {}
            ),
            observation_preprocessor=RunningStandardScaler
            if algo.state_preprocessor
            else None,
            observation_preprocessor_kwargs=(
                {"size": obs_space, "device": device} if algo.state_preprocessor else {}
            ),
            value_preprocessor=RunningStandardScaler
            if algo.value_preprocessor
            else None,
            value_preprocessor_kwargs=(
                {"size": 1, "device": device} if algo.value_preprocessor else {}
            ),
        )
        # Disable skrl's built-in tracking — Tracker takes over.
        cfg.experiment.write_interval = 0
        cfg.experiment.checkpoint_interval = 0
        cfg.experiment.wandb = False

        agent = PPO(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=obs_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        agent.init()

        # Tracker: composes WandbLogger + CheckpointSaver + EpisodeStats.
        tracker = Tracker(
            train_cfg,
            str(device),
            agent_modules=agent.checkpoint_modules,
            env=env,
        )
        end_step = tracker.start_step + algo.timesteps

        log.info(
            f"Starting training for {algo.timesteps} timesteps "
            f"(steps {tracker.start_step}->{end_step})"
        )
        agent.enable_training_mode(True)
        observations, _infos = wrapped.reset()
        states = wrapped.state()

        progress_bar = tqdm.tqdm(
            range(tracker.start_step, end_step),
            desc="Training",
            initial=tracker.start_step,
            total=end_step,
        )
        for timestep in progress_bar:
            agent.pre_interaction(timestep=timestep, timesteps=end_step)

            with torch.no_grad():
                actions, _outputs = agent.act(
                    observations, states, timestep=timestep, timesteps=end_step
                )
                next_observations, rewards, terminated, truncated, infos = wrapped.step(
                    actions
                )
                next_states = wrapped.state()

                agent.record_transition(
                    observations=observations,
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=end_step,
                )

                tracker.record(
                    infos.get("log", {}) if isinstance(infos, dict) else None
                )

            agent.post_interaction(timestep=timestep, timesteps=end_step)
            tracker.flush(
                timestep + 1,
                progress_bar,
                algo_stats=skrl_agent_metrics(agent),
            )

            observations = next_observations
            states = next_states
            _ = num_envs  # silence unused-warning when build_inference_policy is removed

        tracker.finish()

    @staticmethod
    def build_inference_policy(
        algo_cfg: PPOSkrlConfig,
        obs_space: gym.Space,
        action_space: gym.Space,
        agent_state: dict[str, Any],
        device: str = "cuda:0",
    ) -> nn.Module:
        """Build a deterministic policy from a saved skrl checkpoint dict."""
        policy = GaussianPolicy(
            obs_space,
            action_space,
            device,
            algo_cfg.policy_hidden_dims,
            algo_cfg.activation,
            out_activation=algo_cfg.out_activation,
        )
        policy.load_state_dict(agent_state["policy"])
        policy.eval()
        return policy
