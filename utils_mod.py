from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos

import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import soccer_twos

class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(self, env):
        super().__init__(env)
        self.reward_scale = 1.0
        self.touch_radius = 0.5
        
        # Tracking states
        self.prev_infos = {}
        self.was_touching = {}
        self.in_possession = {}

        # Update observation space: Raycasts (336) + Rel Vectors (6) = 342
        orig_shape = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape + 6,), dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        self.prev_infos = {} # Reset tracking on new episode
        self.was_touching = {}
        self.in_possession = {}
        
        # Get initial positions if available from base wrapper
        infos = getattr(self.env, "prev_infos", {})
        return self._modify_obs(obs, infos)

    def step(self, action_dict):
        # Unity safety: Don't step if RLlib sends an empty dict
        if not action_dict:
            return {}, {}, {"__all__": False}, {}

        obs, rewards, dones, infos = self.env.step(action_dict)

        # Order matters: Modify rewards BEFORE updating prev_infos
        rewards = self._modify_rewards(rewards, infos)
        obs = self._modify_obs(obs, infos)
        
        self.prev_infos = infos
        return obs, rewards, dones, infos

    def _modify_obs(self, obs_dict, infos):
        new_obs = {}
        for agent_id, agent_obs in obs_dict.items():
            if agent_id not in infos:
                # Fallback for frame 0
                new_obs[agent_id] = np.concatenate([agent_obs, np.zeros(6, dtype=np.float32)])
                continue

            info = infos[agent_id]
            p_pos = np.array(info["player_info"]["position"], dtype=np.float32)
            b_pos = np.array(info["ball_info"]["position"], dtype=np.float32)
            
            if agent_id < 2:
                opp_goal_pos = np.array([14.0, 0.0], dtype=np.float32)
                teammate_id = 1 - agent_id
            else:
                opp_goal_pos = np.array([-14.0, 0.0], dtype=np.float32)
                teammate_id = 5 - agent_id # 2->3, 3->2

            t_pos = np.array(infos.get(teammate_id, {}).get("player_info", {}).get("position", p_pos), dtype=np.float32)

            rel_ball = b_pos - p_pos
            rel_goal = opp_goal_pos - p_pos
            rel_team = t_pos - p_pos
            
            new_obs[agent_id] = np.concatenate([agent_obs, rel_ball, rel_goal, rel_team]).astype(np.float32)
        return new_obs

    def _modify_rewards(self, reward_dict, infos):
        shaped_rewards = {}

        for agent_id, base_reward in reward_dict.items():
            info = infos.get(agent_id)
            # Use current info as prev_info fallback for the first step
            prev_info = self.prev_infos.get(agent_id, info)

            if not info or not prev_info:
                shaped_rewards[agent_id] = base_reward
                continue

            # Positions
            ball_curr = np.array(info["ball_info"]["position"])
            player_curr = np.array(info["player_info"]["position"])
            ball_prev = np.array(prev_info["ball_info"]["position"])
            player_prev = np.array(prev_info["player_info"]["position"])
            
            opp_goal = np.array([14.0, 0.0]) if agent_id < 2 else np.array([-14.0, 0.0])

            # Distances
            d_pb_curr = np.linalg.norm(ball_curr - player_curr)
            d_pb_prev = np.linalg.norm(ball_prev - player_prev)
            d_bg_curr = np.linalg.norm(ball_curr - opp_goal)
            d_bg_prev = np.linalg.norm(ball_prev - opp_goal)

            # Touch Logic
            touching = d_pb_curr < self.touch_radius
            touch_bonus = 1.0 if touching and not self.was_touching.get(agent_id, False) else 0.0
            self.was_touching[agent_id] = touching

            # Possession Logic
            opp_ids = [2, 3] if agent_id < 2 else [0, 1]
            opp_dists = [np.linalg.norm(np.array(infos[oid]["player_info"]["position"]) - ball_curr) 
                         for oid in opp_ids if oid in infos]
            
            d_closest_opp = min(opp_dists) if opp_dists else float('inf')
            in_pos = touching and (d_pb_curr <= d_closest_opp)
            
            possession_reward = 0.0
            if in_pos:
                possession_reward += 0.05
                if not self.in_possession.get(agent_id, False):
                    possession_reward += 0.2
            self.in_possession[agent_id] = in_pos

            # Combine
            shaped_rewards[agent_id] = (
                base_reward +
                (d_bg_prev - d_bg_curr) * 1.0 + # Ball toward goal
                (d_pb_prev - d_pb_curr) * 0.2 + # Player toward ball
                touch_bonus +
                possession_reward
            )
            
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
