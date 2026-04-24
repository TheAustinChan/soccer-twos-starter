import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from utils_mod import create_rllib_env  # Ensure this imports correctly

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO_selfplay_rec_reg_mod_new/PPO_Soccer_0b4e9_00000_0_2026-04-24_04-17-33/checkpoint_000080/checkpoint-80",
)
POLICY_NAME = "default"  # Use this as the main policy


class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), "params.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError(f"Could not find params.pkl at {config_path}")

        # No need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # Register the real environment
        tune.registry.register_env("Soccer", lambda env_config: create_rllib_env(env_config))
        config["env"] = "Soccer"

        # Create the Trainer from the config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)

        # Load state from checkpoint
        agent.restore(CHECKPOINT_PATH)

        # Get policies for multi-agent setup
        self.policies = {
            "default": agent.get_policy("default"),
            "opponent_1": agent.get_policy("opponent_1"),
            "opponent_2": agent.get_policy("opponent_2"),
            "opponent_3": agent.get_policy("opponent_3"),
        }

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: A dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: A dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        for player_id, obs in observation.items():
            # Choose the policy based on the agent ID
            if player_id == 0:
                policy = self.policies["default"]
            elif player_id == 1:
                policy = self.policies["opponent_1"]
            elif player_id == 2:
                policy = self.policies["opponent_2"]
            elif player_id == 3:
                policy = self.policies["opponent_3"]
            else:
                policy = self.policies["default"]  # Default fallback

            # Compute action using the selected policy
            action, _ = policy.compute_single_action(obs)
            actions[player_id] = action
            
        return actions
