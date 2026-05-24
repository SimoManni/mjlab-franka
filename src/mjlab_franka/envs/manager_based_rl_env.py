"""Custom ManagerBasedRlEnv that swaps in aloy-style reward + metrics managers."""

from __future__ import annotations

from mjlab.envs.manager_based_rl_env import (
    ManagerBasedRlEnv as MjlabManagerBasedRlEnv,
)
from mjlab.envs.mdp.curriculums import (
    reward_curriculum,
    termination_curriculum,
)

from mjlab_franka.managers import MetricsManager, RewardManager


class ManagerBasedRlEnv(MjlabManagerBasedRlEnv):
    """Env subclass that uses our aloy-style reward + metrics managers.

    The custom RewardManager always computes every term (even weight=0.0) and
    caches raw per-step values in ``reward_manager.step_raw``. The MetricsManager
    consumes those raw values plus any user-defined ``cfg.metrics`` term and
    returns ``_episode_metrics`` from ``reset(env_ids)`` for the Tracker.
    """

    def load_managers(self) -> None:
        super().load_managers()

        # Rebuild reward manager with the raw-tracking subclass.
        self.reward_manager = RewardManager(
            self.cfg.rewards, self, scale_by_dt=self.cfg.scale_rewards_by_dt
        )

        # Rebuild metrics manager so it can read raw reward outputs.
        self.metrics_manager = MetricsManager(
            self.cfg.metrics, self, reward_manager=self.reward_manager
        )

        # ``super().load_managers()`` built the curriculum manager BEFORE we
        # rebuilt the reward manager above. Curriculum terms that target a
        # reward term (e.g. ``reward_curriculum``) cached a reference to the
        # old reward term_cfg in their ``__init__``; those references are now
        # stale and ``setattr`` updates would silently no-op. Rebind them to
        # the new reward_manager's term_cfgs.
        self._rebind_curriculum_to_new_managers()

    def _rebind_curriculum_to_new_managers(self) -> None:
        cm = getattr(self, "curriculum_manager", None)
        term_cfgs = getattr(cm, "_term_cfgs", None)
        if not term_cfgs:
            return
        for term_cfg in term_cfgs:
            func = term_cfg.func
            if isinstance(func, reward_curriculum):
                reward_name = term_cfg.params["reward_name"]
                func._term_cfg = self.reward_manager.get_term_cfg(reward_name)
            elif isinstance(func, termination_curriculum):
                termination_name = term_cfg.params["termination_name"]
                func._term_cfg = self.termination_manager.get_term_cfg(termination_name)
