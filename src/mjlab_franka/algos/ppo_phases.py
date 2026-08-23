"""SKRL PPO algorithm with an aloy-style Tracker-driven training loop."""

from __future__ import annotations

import logging
import itertools
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import tqdm
from hydra.core.config_store import ConfigStore
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl import config
from skrl.agents.torch.ppo.ppo import compute_gae
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR


from mjlab_franka.algos.registry import register_algo
from mjlab_franka.config.base import TrainConfig
from mjlab_franka.core.stats import skrl_agent_metrics
from mjlab_franka.core.tracker import Tracker
from mjlab_franka.algos.ppo_skrl import (
    PPOSkrlConfig, 
    GaussianPolicy, 
    DeterministicValue, 
    _activation, 
    _make_mlp, 
    _adapt_env_spaces_in_place
)

log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MLPCfg:
    """Configuration for a simple MLP model."""

    value_hidden_dims : list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "elu"

@dataclass
class ResidualCfg:
    """Configuration for a residual network model."""

    value_hidden_dims : list[int] = field(default_factory=lambda: [256, 256])
    base_hidden_dims : list[int] = field(default_factory=lambda: [256, 256])
    head_hidden_dims : list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "elu"
    residual_l2_weight: float = 1e-4

@dataclass
class MultiheadCfg:
    """Configuration for a multi-head network model."""

    value_hidden_dims : list[int] = field(default_factory=lambda: [256, 256])
    head_hidden_dims : list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "elu"

@dataclass
class PPOPhasesConfig(PPOSkrlConfig):
    """Configuration for the PPO algorithm with phase-based training."""

    num_phases: int = 0
    actor_phase_prediction: bool = False
    actor_phase_head_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    value_cfg: Any = field(default_factory=MLPCfg)


cs = ConfigStore.instance()
cs.store(group="algo", name="ppo_phases", node=PPOPhasesConfig)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class GaussianPolicyWithPhasePrediction(GaussianPolicy):
    """Gaussian policy model that includes an auxiliary head to predict the current task phase

    and a dedicated helper method to query phase predictions.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device | str,
        hidden_dims: list[int],
        activation: str,
        num_phases: int,
        phase_head_hidden_dims: list[int] = [128, 128],
        out_activation: str | None = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            hidden_dims=hidden_dims,
            activation=activation,
            out_activation=out_activation,
        )

        last_base_dim = hidden_dims[-1] if hidden_dims else int(self.num_observations)
        
        phase_layers = []
        curr_dim = last_base_dim
        for h_dim in phase_head_hidden_dims:
            phase_layers.append(nn.Linear(curr_dim, h_dim))
            phase_layers.append(_activation(activation))
            curr_dim = h_dim
        
        phase_layers.append(nn.Linear(curr_dim, num_phases))
        self.phase_head = nn.Sequential(*phase_layers)

    def compute(
        self, inputs: dict[str, torch.Tensor], role: str
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x = self.net(inputs["observations"])
        
        mean = self.mean_layer(x)
        if self.out_activation is not None:
            mean = self.out_activation(mean)

        phase_logits = self.phase_head(x)

        return mean, {
            "log_std": self.log_std_parameter,
            "phase_logits": phase_logits,
        }

    def predict_phase(self, observations: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """Helper method to predict the current phase from raw observations.

        Args:
            observations: Tensor of observations, shape (batch_size, num_observations)
            return_probs: If True, returns softmax probabilities instead of discrete class indices.
            
        Returns:
            Tensor of predicted phase indices (shape [batch_size]) or probabilities (shape [batch_size, num_phases]).
        """
        # Ensure inputs are on the correct device and formatted correctly
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, device=self.device, dtype=torch.float32)
        else:
            observations = observations.to(self.device)

        # Forward pass through base network and phase head
        x = self.net(observations)
        phase_logits = self.phase_head(x)

        if return_probs:
            return torch.softmax(phase_logits, dim=-1)
        else:
            return torch.argmax(phase_logits, dim=-1)




class MultiHeadsValueNetwork(DeterministicMixin, Model):
    """Multi-head value network for 'multihead' architecture using MultiheadCfg dimensions:

    Shared trunk + one fully independent linear head per phase.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        cfg: MultiheadCfg,
        num_phases: int,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.num_phases = num_phases

        obs_dim = observation_space.shape[0]
        # Build base trunk using value_hidden_dims
        self.trunk, trunk_out_dim = _make_mlp(obs_dim, cfg.value_hidden_dims, activation=cfg.activation, return_last_dim=True)
        
        # Dedicated heads per phase using head_hidden_dims or direct linear mapping
        self.heads = nn.ModuleList([
            nn.Sequential(
                _make_mlp(trunk_out_dim, cfg.head_hidden_dims, activation=cfg.activation),
                nn.Linear(cfg.head_hidden_dims[-1], 1)
            ) for _ in range(num_phases)
        ])

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, dict]:
        phase_encoding = inputs.get("phase") # or parsed from state
        if phase_encoding is None:
            # Fallback assumption if phase is embedded in states
            obs = inputs.get("states")
            phase_encoding = obs[:, -self.num_phases:]
            
        active_phase_idx = torch.argmax(phase_encoding, dim=-1)
        obs = inputs.get("states")
        
        trunk_out = self.trunk(obs)
        all_vals = torch.stack(
            [head(trunk_out).squeeze(-1) for head in self.heads], dim=1
        )
        v = all_vals.gather(1, active_phase_idx.unsqueeze(1)).squeeze(1)
        return v.unsqueeze(1), {}   

class ResidualHeadsValueNetwork(DeterministicMixin, Model):
    """Residual value network for 'residual' architecture using ResidualCfg dimensions:

    V(s, phi) = V_base(s) + V_residual^phi(s)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        cfg: ResidualCfg,
        num_phases: int,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.num_phases = num_phases
        self.residual_l2_weight = cfg.residual_l2_weight

        obs_dim = observation_space.shape[0]
        
        # Base trunk matching base_hidden_dims
        self.trunk, trunk_out_dim = _make_mlp(obs_dim, cfg.base_hidden_dims, activation=cfg.activation, return_last_dim=True)

        # Global base head (value_hidden_dims mapping)
        self.base_head = _make_mlp(trunk_out_dim, 1, cfg.value_hidden_dims, activation=cfg.activation)

        # Phase-specific residual heads matching head_hidden_dims
        self.residual_heads = nn.ModuleList([
            _make_mlp(trunk_out_dim, 1, cfg.head_hidden_dims, activation=cfg.activation)
            for _ in range(num_phases)
        ])
        self._residual_params: list[nn.Parameter] = []

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, dict]:
        obs = inputs.get("states")
        is_1d = obs.ndim == 1
        if is_1d:
            obs = obs.unsqueeze(0)

        phase_encoding = inputs.get("phase")
        if phase_encoding is None:
            phase_encoding = obs[:, -self.num_phases:]
        active_phase_idx = torch.argmax(phase_encoding, dim=-1)

        trunk_out = self.trunk(obs)
        base_value = self.base_head(trunk_out).squeeze(-1)
        all_residuals = torch.stack(
            [head(trunk_out).squeeze(-1) for head in self.residual_heads], dim=1
        )

        residual = all_residuals.gather(1, active_phase_idx.unsqueeze(1)).squeeze(1)
        total_value = base_value + residual
        
        if is_1d:
            total_value = total_value.squeeze(0)

        return total_value.unsqueeze(1), {}

    def residual_l2_loss(self) -> torch.Tensor:
        if not self._residual_params:
            self._residual_params = list(self.residual_heads.parameters())
        if not self._residual_params:
            return torch.tensor(0.0, device=self.device)
        flat = torch.cat([p.reshape(-1) for p in self._residual_params])
        return self.residual_l2_weight * flat.pow(2).sum()


# ---------------------------------------------------------------------------
# Extended PPO Agent
# ---------------------------------------------------------------------------

class _PPOPhasesAgent(PPO):
    def __init__(self, cfg: PPOPhasesConfig, **kwargs) -> None:
        super().__init__(cfg=cfg, **kwargs)
        self.phases_cfg = cfg
        self._has_residual_l2 = hasattr(self.value, "residual_l2_loss")
        self._has_phase_prediction = hasattr(self.policy, "predict_phase")

    def init(self, *, trainer_cfg = None):
        super().init(trainer_cfg=trainer_cfg)

        if self.memory is not None:
            self.memory.create_tensor(name="phases", size=1, dtype=torch.long)
        self._tensors_names.append("phases")

    def record_transition(self, *, observations, states, actions, rewards, next_observations, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(observations=observations, states=states, actions=actions, rewards=rewards, next_observations=next_observations, next_states=next_states, terminated=terminated, truncated=truncated, infos=infos, timestep=timestep, timesteps=timesteps)
        current_phases = observations["phase"]
        self.memory.set_tensor_by_name("phases", current_phases)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            truncated=self.memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
            time_limit_bootstrap=self.cfg.time_limit_bootstrap,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_extra_loss = 0
        if self._has_phase_prediction:
            total_phase_correct = 0
            total_phase_samples = 0
            cumulative_phase_loss = 0

        # learning epochs
        for epoch in range(self.cfg.learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_phases,
            ) in self.memory.sample(
                names=self._tensors_names, batch_size=len(self.memory), mini_batches=self.cfg.mini_batches
            ):

                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                        "states": self._state_preprocessor(sampled_states, train=not epoch),
                        "phase": sampled_phases,
                    }

                    _, outputs = self.policy.act({**inputs, "taken_actions": sampled_actions}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self.cfg.kl_threshold and kl_divergence > self.cfg.kl_threshold:
                        break

                    # compute entropy loss
                    if self.cfg.entropy_loss_scale:
                        entropy_loss = -self.cfg.entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip, 1.0 + self.cfg.ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    predicted_values, _ = self.value.act(inputs, role="value")

                    if self.cfg.value_clip > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip, max=self.cfg.value_clip
                        )
                    value_loss = self.cfg.value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # Compute extra loss if any
                    extra_loss = 0.0
                    if self._has_residual_l2:
                        extra_loss = self.value.residual_l2_loss()

                    # Compute Auxiliary Actor Phase Prediction Loss
                    phase_loss = 0.0
                    if self._has_phase_prediction and "phase_logits" in outputs:
                        phase_logits = outputs["phase_logits"]
                        
                        # Convert one-hot to class indices for cross_entropy
                        if sampled_phases.ndim > 1 and sampled_phases.shape[-1] > 1:
                            phase_targets = torch.argmax(sampled_phases, dim=-1)
                        else:
                            phase_targets = sampled_phases.long().squeeze(-1)

                        phase_loss = F.cross_entropy(phase_logits, phase_targets)
                        
                        with torch.no_grad():
                            preds = torch.argmax(phase_logits, dim=-1)
                            total_phase_correct += (preds == phase_targets).sum().item()
                            total_phase_samples += phase_targets.numel()

                
                # optimization step
                total_loss = policy_loss + entropy_loss + value_loss + extra_loss + phase_loss
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self.cfg.grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                if self._has_residual_l2:
                    cumulative_extra_loss += extra_loss.item()
                if self._has_phase_prediction:
                    cumulative_phase_loss += phase_loss.item()

            # update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        n_steps = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data(
            "Loss / Policy loss", cumulative_policy_loss / n_steps
        )
        self.track_data("Loss / Value loss", cumulative_value_loss / n_steps)
        if self.cfg.entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / n_steps
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._has_residual_l2:
            self.track_data("Loss / Residual L2 loss", cumulative_extra_loss / n_steps)

        if self.scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
    
        if self._has_phase_prediction and total_phase_samples > 0:
            self.track_data("Loss / Phase prediction loss", cumulative_phase_loss / n_steps)
            self.track_data("Accuracy / Actor phase prediction", total_phase_correct / total_phase_samples)

# ---------------------------------------------------------------------------
# Algo entry point
# ---------------------------------------------------------------------------


value_cfg_to_class_map = {
    "mlp": DeterministicValue,
    "residual": ResidualHeadsValueNetwork,
    "multihead": MultiHeadsValueNetwork,
}

@register_algo("ppo_skrl")
class PPOSkrl:
    """Thin SKRL PPO training entry point."""

    @staticmethod
    def train(train_cfg: TrainConfig, env: Any) -> None:
        algo: PPOPhasesConfig = train_cfg.algo  # type: ignore[assignment]
        device = train_cfg.env.device
        num_envs = env.num_envs

        if algo.num_phases <= 0:
            raise ValueError(f"Invalid num_phases: {algo.num_phases}. Must be > 0.")

        # Wrap mjlab env for skrl (uses single_observation_space["policy"] / ["critic"]).
        _adapt_env_spaces_in_place(env)
        wrapped = wrap_env(env, wrapper="isaaclab")

        obs_space = wrapped.observation_space
        state_space = wrapped.state_space  # None when env has no 'critic' group
        action_space = wrapped.action_space

        # Models — asymmetric actor-critic when env exposes a 'critic' group.
        if algo.actor_phase_prediction:
            policy = GaussianPolicyWithPhasePrediction(
                observation_space=obs_space,
                action_space=action_space,
                device=device,
                hidden_dims=algo.actor_hidden_dims,
                activation=algo.actor_activation,
                num_phases=algo.num_phases,
                phase_head_hidden_dims=algo.actor_phase_head_hidden_dims,
            )
        else:
            policy = GaussianPolicy(
                observation_space=obs_space,
                action_space=action_space,
                device=device,
                hidden_dims=algo.actor_hidden_dims,
                activation=algo.actor_activation,
            )
        value_class = value_cfg_to_class_map.get(algo.value_cfg.__class__.__name__.lower())
        if value_class is None:
            raise ValueError(f"Unsupported value_cfg type: {type(algo.value_cfg)}")
        value = value_class(
            observation_space=state_space if state_space is not None else obs_space,
            action_space=action_space,
            device=device,
            cfg=algo.value_cfg,
            **algo.value_cfg.__dict__
        )

        models = {"policy": policy, "value": value}

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
