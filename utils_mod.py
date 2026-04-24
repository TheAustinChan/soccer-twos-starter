from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper that allows for custom observation and reward shaping.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # You can define constants here, e.g., reward scaling factors
        self.reward_scale = 1.0

    def reset(self):
        """
        Modify observations at the start of an episode.
        """
        obs = self.env.reset()
        return self._modify_obs(obs)

    def step(self, action_dict):
        """
        Modify rewards, observations, or check for custom termination logic.
        """
        # 1. Take the step in the underlying environment
        obs, rewards, dones, infos = self.env.step(action_dict)

        # 2. Modify observations
        obs = self._modify_obs(obs)

        # 3. Modify rewards
        rewards = self._modify_rewards(rewards)

        return obs, rewards, dones, infos

    def _modify_obs(self, obs_dict):
        """
        Apply custom logic to the observation dictionary.
        """
        for agent_id, agent_obs in obs_dict.items():
            # Example: Normalize or add noise
            # obs_dict[agent_id] = agent_obs * 1.0 
            pass
        return obs_dict

    def _modify_rewards(self, reward_dict):
        """
        Apply custom logic to the reward dictionary.
        """
        for agent_id, reward in reward_dict.items():
            # Example: Dense reward for moving toward the ball
            # reward_dict[agent_id] = reward + custom_shaping_logic
            pass
        return reward_dict


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
