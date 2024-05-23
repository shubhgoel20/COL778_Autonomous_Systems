"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)

register(
    id="HybridCar-v1",
    entry_point="car_gym.envs.hybrid_car.car:CarEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=600,
    reward_threshold=200,  # optimum = 0.91
)


# Hook to load plugins from entry points
load_plugin_envs()
