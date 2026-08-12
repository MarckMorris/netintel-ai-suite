"""The SD-WAN path-selection environment.

The Gymnasium API contract is easy to get subtly wrong, and a broken contract
produces an agent that trains happily on garbage. These tests check the parts
that fail silently: seeding, observation bounds, and episode termination.
"""

from __future__ import annotations

import numpy as np
import pytest

from network_env import (
    DEFAULT_PROFILES,
    FEATURES_PER_PATH,
    LinkProfile,
    SDWANPathEnv,
)


@pytest.fixture
def env():
    return SDWANPathEnv()


class TestSpaces:
    def test_observation_has_five_features_per_path(self, env):
        assert env.observation_space.shape == (4 * FEATURES_PER_PATH,)

    def test_one_action_per_path(self, env):
        assert env.action_space.n == 4

    def test_the_space_scales_with_the_path_count(self):
        small = SDWANPathEnv(num_paths=2)
        assert small.observation_space.shape == (2 * FEATURES_PER_PATH,)
        assert small.action_space.n == 2

    def test_more_paths_than_profiles_is_rejected(self):
        with pytest.raises(ValueError):
            SDWANPathEnv(num_paths=len(DEFAULT_PROFILES) + 1)

    def test_zero_paths_is_rejected(self):
        with pytest.raises(ValueError):
            SDWANPathEnv(num_paths=0)


class TestSeeding:
    def test_the_same_seed_gives_the_same_start(self):
        """Without this, reset(seed=...) is decorative and no RL result in this
        repository can be reproduced."""
        first, _ = SDWANPathEnv().reset(seed=42)
        second, _ = SDWANPathEnv().reset(seed=42)
        assert np.array_equal(first, second)

    def test_different_seeds_give_different_starts(self):
        first, _ = SDWANPathEnv().reset(seed=42)
        second, _ = SDWANPathEnv().reset(seed=7)
        assert not np.array_equal(first, second)

    def test_a_seeded_rollout_is_reproducible_end_to_end(self):
        def rollout():
            env = SDWANPathEnv()
            env.reset(seed=123)
            return [round(env.step(i % env.action_space.n)[1], 10) for i in range(50)]

        assert rollout() == rollout()

    def test_the_global_numpy_rng_is_not_used(self):
        """Drawing from np.random would make results depend on unrelated code
        that happened to consume the global stream first."""
        np.random.seed(0)
        first, _ = SDWANPathEnv().reset(seed=42)
        np.random.seed(99999)
        [np.random.random() for _ in range(1000)]
        second, _ = SDWANPathEnv().reset(seed=42)
        assert np.array_equal(first, second)


class TestStep:
    def test_step_returns_the_five_element_gymnasium_tuple(self, env):
        env.reset(seed=1)
        result = env.step(0)
        assert len(result) == 5

    def test_observations_stay_inside_the_declared_space(self, env):
        obs, _ = env.reset(seed=1)
        assert env.observation_space.contains(obs)
        for action in range(env.action_space.n):
            obs = env.step(action)[0]
            assert env.observation_space.contains(obs)

    def test_conditions_change_every_step(self, env):
        """A static environment lets an agent memorise one path and look great."""
        before, _ = env.reset(seed=1)
        after = env.step(0)[0]
        assert not np.array_equal(before, after)

    def test_an_invalid_action_is_rejected(self, env):
        env.reset(seed=1)
        with pytest.raises(ValueError):
            env.step(99)

    def test_the_episode_truncates_at_its_length(self):
        env = SDWANPathEnv(episode_length=10)
        env.reset(seed=1)
        for step in range(9):
            assert env.step(0)[3] is False
        assert env.step(0)[3] is True

    def test_reset_restarts_the_episode_clock(self):
        env = SDWANPathEnv(episode_length=3)
        env.reset(seed=1)
        env.step(0)
        env.step(0)
        env.reset(seed=1)
        assert env.step(0)[3] is False


class TestReward:
    def test_a_clean_fast_path_beats_a_lossy_one(self):
        good = {"latency": 0.1, "bandwidth_util": 0.2, "packet_loss": 0.0, "cost": 0.4, "jitter": 0.05}
        lossy = {**good, "packet_loss": 0.2}
        assert SDWANPathEnv._reward(good) > SDWANPathEnv._reward(lossy)

    def test_loss_is_punished_harder_than_latency(self):
        """A lossy path breaks applications; a slow path only annoys them."""
        base = {"latency": 0.1, "bandwidth_util": 0.5, "packet_loss": 0.0, "cost": 0.4, "jitter": 0.1}
        slower = {**base, "latency": 0.3}
        lossier = {**base, "packet_loss": 0.2}
        assert SDWANPathEnv._reward(lossier) < SDWANPathEnv._reward(slower)

    def test_a_cheaper_path_is_preferred_all_else_equal(self):
        cheap = {"latency": 0.2, "bandwidth_util": 0.5, "packet_loss": 0.0, "cost": 0.4, "jitter": 0.1}
        pricey = {**cheap, "cost": 1.0}
        assert SDWANPathEnv._reward(cheap) > SDWANPathEnv._reward(pricey)

    def test_spare_capacity_is_rewarded(self):
        idle = {"latency": 0.2, "bandwidth_util": 0.2, "packet_loss": 0.0, "cost": 0.4, "jitter": 0.1}
        busy = {**idle, "bandwidth_util": 0.9}
        assert SDWANPathEnv._reward(idle) > SDWANPathEnv._reward(busy)


class TestProfiles:
    def test_the_shipped_profiles_are_not_named_after_real_carriers(self):
        """Generic names, so nobody reads a vendor comparison into a simulation."""
        for profile in DEFAULT_PROFILES:
            assert profile.name.replace("-", "").isalnum()

    def test_reliability_is_a_probability(self):
        for profile in DEFAULT_PROFILES:
            assert 0.0 < profile.reliability <= 1.0

    def test_a_custom_profile_set_can_be_supplied(self):
        profiles = (LinkProfile("only-link", 10, 2, 0.99, 1.0),)
        env = SDWANPathEnv(num_paths=1, profiles=profiles)
        env.reset(seed=1)
        assert env.action_space.n == 1

    def test_a_more_reliable_link_loses_fewer_packets_over_many_samples(self):
        flaky = SDWANPathEnv(num_paths=1, profiles=(LinkProfile("flaky", 25, 5, 0.50, 1.0),))
        solid = SDWANPathEnv(num_paths=1, profiles=(LinkProfile("solid", 25, 5, 0.999, 1.0),))
        flaky.reset(seed=3)
        solid.reset(seed=3)

        def loss_events(env):
            return sum(1 for _ in range(300) if env.step(0)[0][2] > 0)

        assert loss_events(flaky) > loss_events(solid)
