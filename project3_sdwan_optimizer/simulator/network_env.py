"""A Gymnasium environment for SD-WAN path selection.

An SD-WAN edge device has several uplinks - a cheap broadband circuit, a more
reliable business line, an LTE/5G backup - and has to pick one per flow. The
naive policy is "always take the lowest latency", which sends everything over
the expensive circuit and ignores loss. This environment lets an agent learn a
policy that trades latency, loss, utilisation and cost against each other.

Link profiles are generic on purpose. Substitute your own carriers' measured
characteristics before reading anything into a trained policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

FEATURES_PER_PATH = 5
MAX_LATENCY_MS = 200.0
DEFAULT_EPISODE_LENGTH = 1000


@dataclass(frozen=True)
class LinkProfile:
    """A carrier circuit.

    ``reliability`` is the probability that a step sees no packet loss at all;
    ``cost`` is relative, with 1.0 as the cheapest broadband circuit.
    """

    name: str
    base_latency_ms: float
    latency_jitter_ms: float
    reliability: float
    cost: float


DEFAULT_PROFILES: tuple[LinkProfile, ...] = (
    LinkProfile("broadband-primary", base_latency_ms=25, latency_jitter_ms=10, reliability=0.95, cost=1.0),
    LinkProfile("broadband-secondary", base_latency_ms=30, latency_jitter_ms=12, reliability=0.93, cost=1.2),
    LinkProfile("business-fibre", base_latency_ms=18, latency_jitter_ms=5, reliability=0.99, cost=2.5),
    LinkProfile("lte-5g-backup", base_latency_ms=40, latency_jitter_ms=25, reliability=0.85, cost=2.0),
)


class SDWANPathEnv(gym.Env):
    """Pick one uplink per step; be rewarded for a good one.

    Observation: ``FEATURES_PER_PATH`` normalised values per path - latency,
    bandwidth utilisation, packet loss, cost, jitter - concatenated.
    Action: the index of the path to use.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_paths: int = 4,
        profiles: tuple[LinkProfile, ...] = DEFAULT_PROFILES,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
    ):
        super().__init__()

        if not 1 <= num_paths <= len(profiles):
            raise ValueError(f"num_paths must be between 1 and {len(profiles)}, got {num_paths}")

        self.num_paths = num_paths
        self.profiles = profiles[:num_paths]
        self.episode_length = episode_length

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_paths * FEATURES_PER_PATH,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_paths)

        self.path_states: list[dict[str, float]] = []
        self.current_step = 0

    # -- gym API ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # super().reset(seed=...) seeds self.np_random. Every random draw below
        # goes through it rather than the global numpy RNG, which is what makes
        # reset(seed=n) actually reproducible. The previous implementation used
        # np.random directly, so seeding silently did nothing.
        super().reset(seed=seed)

        self.path_states = [self._sample_path_state(p) for p in self.profiles]
        self.current_step = 0
        return self._observation(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} is outside {self.action_space}")

        self.current_step += 1
        reward = self._reward(self.path_states[action])

        # Conditions change every step, so a policy cannot memorise one good path.
        self.path_states = [self._sample_path_state(p) for p in self.profiles]

        terminated = False
        truncated = self.current_step >= self.episode_length
        return self._observation(), reward, terminated, truncated, {}

    # -- internals ----------------------------------------------------------

    def _sample_path_state(self, profile: LinkProfile) -> dict[str, float]:
        rng = self.np_random
        latency = float(
            np.clip(rng.normal(profile.base_latency_ms, profile.latency_jitter_ms), 5, MAX_LATENCY_MS)
        )
        degraded = rng.random() > profile.reliability
        return {
            "latency": latency / MAX_LATENCY_MS,
            "bandwidth_util": float(rng.uniform(0.3, 0.9)),
            "packet_loss": float(min(rng.exponential(0.01), 1.0)) if degraded else 0.0,
            "cost": profile.cost / 2.5,
            "jitter": float(min(rng.exponential(5) / 50, 1.0)),
        }

    @staticmethod
    def _reward(path: dict[str, float]) -> float:
        """Loss is weighted hardest: a lossy path breaks applications that a
        merely slow path only annoys."""
        return (
            -path["latency"] * 2.0
            - path["packet_loss"] * 5.0
            - path["cost"] * 0.5
            + (1.0 - path["bandwidth_util"]) * 0.5
        )

    def _observation(self) -> np.ndarray:
        values: list[float] = []
        for path in self.path_states:
            values.extend(
                [path["latency"], path["bandwidth_util"], path["packet_loss"], path["cost"], path["jitter"]]
            )
        return np.array(values, dtype=np.float32)
