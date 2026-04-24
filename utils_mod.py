from random import uniform as randfloat

import gym
from gym.spaces import Box
from ray.rllib import MultiAgentEnv
import soccer_twos
import numpy as np


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper that allows for custom observation and reward shaping.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # You can define constants here, e.g., reward scaling factors
        self.reward_scale = 1.0

        #NEW section: observation
        base_shape = env.observation_space.shape[0]
        assert base_shape in [336, 345], (f"[RLLibWrapper ERROR] Unexpected observation size: {base_shape}. "
                                          f"Expected 336 or 345 depending on env config.")
        
        self.base_obs_size = base_shape     #EDIT: SOMETIMES 345 ssometimes 336, unsure? from soccer-twos-env (ray + player + ball info)
        self.extra_obs_size = 4      # teammate (x,y,vx,vy)
        self.final_obs_size = self.base_obs_size + self.extra_obs_size  # 349

        print(f"[RLLibWrapper] base obs size = {self.base_obs_size}")

        # update RLlib observation space with new obs size
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.final_obs_size,),
            dtype=np.float32,
        )
        
        # fixed 2v2 structure
        self.teams = {
            0: [0, 1],
            1: [2, 3],
        }

        # store last infos safely (needed for obs augmentation)
        self._last_infos = {}
        #END NEW
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

        #NEW STMT: store infos for observation augmentation
        self._last_infos = infos
        
        # 2. Modify observations
        obs = self._modify_obs(obs)

        # 3. Modify rewards
        rewards = self._modify_rewards(rewards, infos) #NEW PARAM: infos

        return obs, rewards, dones, infos

    #Augment observation from 345 -> 345 + 4 for x, y, vx, vy of teammate
    def _modify_obs(self, obs_dict):
        """
        Apply custom logic to the observation dictionary.
        """
        """for agent_id, agent_obs in obs_dict.items():
            # Example: Normalize or add noise
            # obs_dict[agent_id] = agent_obs * 1.0 
            pass
        return obs_dict
        """
        new_obs = {}

        for team in self.teams.values():
            for agent_id in team:

                base_obs = np.array(obs_dict[agent_id], dtype=np.float32)

                # FIND TEAMMATE
                teammate_id = [i for i in team if i != agent_id][0]

                # default zeros if missing the extra 336:340 OR 345:349 obs
                teammate_pos = np.zeros(2, dtype=np.float32)
                teammate_vel = np.zeros(2, dtype=np.float32)

                # GET TEAMMATE INFO
                info = self._last_infos.get(teammate_id, None)

                if info and "player_info" in info:
                    p = info["player_info"]
                    teammate_pos = np.array(p.get("position", [0, 0]), dtype=np.float32)
                    teammate_vel = np.array(p.get("velocity", [0, 0]), dtype=np.float32)

                # BUILD EXTRA FEATURES (4 dims)
                extra = np.concatenate([teammate_pos, teammate_vel])

                # FINAL OBS (345:349) or (336:340)
                new_obs[agent_id] = np.concatenate([base_obs, extra]).astype(np.float32)

        return new_obs

    def _modify_rewards(self, reward_dict, infos):
        """
        Apply custom logic to the reward dictionary.
        """
        """
        for agent_id, reward in reward_dict.items():
            # Example: Dense reward for moving toward the ball
            # reward_dict[agent_id] = reward + custom_shaping_logic
            pass
        return reward_dict
        """
        shaped_rewards = {}

        for agent_id, reward in reward_dict.items():
            shaped = reward

            info = infos.get(agent_id, None)

            if not info or "ball_info" not in info or "player_info" not in info:
                shaped_rewards[agent_id] = shaped
                continue

            # EXTRACT STATE
            ball_pos = np.array(info["ball_info"]["position"], dtype=np.float32)
            player_pos = np.array(info["player_info"]["position"], dtype=np.float32)

            dist = np.linalg.norm(ball_pos - player_pos)

            # smooth control signal
            possession = 1.0 / (1.0 + dist)

            # SHAPING
            shaped += 0.02 * possession
            shaped -= 0.01 * dist * (1.0 - possession)

            shaped_rewards[agent_id] = shaped

        return shaped_rewards

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
