from random import uniform as randfloat
import gym
from gym.spaces import Box
from ray.rllib import MultiAgentEnv
import soccer_twos
import numpy as np

class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(self, env):
        super().__init__(env)
        
        # --- MATCHING mod_agent.py SPECS ---
        self.base_obs_size = 336
        self.extra_obs_size = 6  # rel_ball(2), rel_goal(2), rel_team(2)
        self.final_obs_size = self.base_obs_size + self.extra_obs_size # 342
        
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self.final_obs_size,),
            dtype=np.float32,
        )
        
        # Tracking for rewards
        self.prev_infos = {} 
        self._last_infos = {}
        self.was_touching = {}
        self.touch_radius = 0.5

    def reset(self):
        obs = self.env.reset()
        self.prev_infos = {}
        self.was_touching = {}
        self._last_infos = {}
        return self._modify_obs(obs, {})

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        
        # 1. Update info tracking
        self._last_infos = infos
        
        # 2. Reward Shaping (Uses current and previous info for deltas)
        rewards = self._modify_rewards(rewards, infos)

        # 3. Observation Augmentation (Matches mod_agent.py)
        obs = self._modify_obs(obs, infos)

        # 4. Cycle state
        self.prev_infos = infos
        return obs, rewards, dones, infos

    def _modify_obs(self, obs_dict, infos):
        """
        Mirroring mod_agent.py logic exactly for 336 + 6 structure.
        """
        new_obs = {}
        for agent_id, agent_obs in obs_dict.items():
            # Truncate to match mod_agent.py's BASE_OBS_SIZE
            truncated_obs = agent_obs[:self.base_obs_size]
            
            if agent_id not in infos:
                # Fallback for reset or missing keys
                new_obs[agent_id] = np.concatenate([truncated_obs, np.zeros(6, dtype=np.float32)])
                continue

            info = infos[agent_id]
            p_pos = np.array(info["player_info"]["position"], dtype=np.float32)
            b_pos = np.array(info["ball_info"]["position"], dtype=np.float32)
            
            # Determine Goal and Teammate IDs
            if agent_id < 2:
                opp_goal_pos = np.array([16.0, 0.0], dtype=np.float32)
                teammate_id = 1 - agent_id
            else:
                opp_goal_pos = np.array([-16.0, 0.0], dtype=np.float32)
                teammate_id = 5 - agent_id

            # Get teammate pos (or self if teammate is missing)
            t_info = infos.get(teammate_id, {}).get("player_info", {})
            t_pos = np.array(t_info.get("position", p_pos), dtype=np.float32)

            # Calculate Relative Vectors
            rel_ball = b_pos - p_pos
            rel_goal = opp_goal_pos - p_pos
            rel_team = t_pos - p_pos
            
            extra = np.concatenate([rel_ball, rel_goal, rel_team])
            new_obs[agent_id] = np.concatenate([truncated_obs, extra]).astype(np.float32)
            
        return new_obs

    def _modify_rewards(self, reward_dict, infos):
        shaped_rewards = {}

        for agent_id, base_reward in reward_dict.items():
            info = infos.get(agent_id)
            prev_info = self.prev_infos.get(agent_id, info)

            if not info or "ball_info" not in info:
                shaped_rewards[agent_id] = base_reward
                continue

            # Extract Positions
            ball_curr = np.array(info["ball_info"]["position"])
            ball_prev = np.array(prev_info["ball_info"]["position"])
            player_curr = np.array(info["player_info"]["position"])
            player_prev = np.array(prev_info["player_info"]["position"])
            player_vel = np.array(info["player_info"]["velocity"])

            # Goal configuration
            if agent_id < 2:
                opp_goal, home_goal, side_dir = np.array([16.0, 0.0]), np.array([-16.0, 0.0]), 1
            else:
                opp_goal, home_goal, side_dir = np.array([-16.0, 0.0]), np.array([16.0, 0.0]), -1

            # --- MATH REWARDS ---
            d_bg_curr = np.linalg.norm(ball_curr - opp_goal)
            d_bg_prev = np.linalg.norm(ball_prev - opp_goal)
            goal_progress = d_bg_prev - d_bg_curr

            d_pb_curr = np.linalg.norm(player_curr - ball_curr)
            d_pb_prev = np.linalg.norm(player_prev - ball_prev)
            
            # Alignment check
            to_ball_norm = (ball_curr - player_curr) / (d_pb_curr + 1e-6)
            to_goal_norm = (opp_goal - ball_curr) / (d_bg_curr + 1e-6)
            alignment = max(0.0, np.dot(to_ball_norm, to_goal_norm))
            approach_reward = (d_pb_prev - d_pb_curr) * alignment

            # Tangential Penalty (anti-orbit)
            app_speed = np.dot(player_vel, to_ball_norm)
            tang_penalty = -0.01 * np.linalg.norm(player_vel - (app_speed * to_ball_norm)) if d_pb_curr < 1.5 else 0.0

            # Defensive Cone (Staying on line to home goal)
            defensive_reward = 0.0
            is_behind = (ball_curr[0] - player_curr[0]) * side_dir > 0
            if is_behind:
                b_to_h = home_goal - ball_curr
                b_to_p = player_curr - ball_curr
                dev = np.abs(np.cross(b_to_h, b_to_p)) / (np.linalg.norm(b_to_h) + 1e-6)
                defensive_reward = 0.05 * (1.0 - min(1.0, dev))

            # Touch bonus
            touching = d_pb_curr < self.touch_radius
            touch_bonus = 0.2 if touching and not self.was_touching.get(agent_id, False) else 0.0
            self.was_touching[agent_id] = touching

            shaped_rewards[agent_id] = (
                base_reward + 
                (goal_progress * 1.2) + 
                (approach_reward * 0.15) + 
                touch_bonus + 
                defensive_reward + 
                tang_penalty
            )

        return shaped_rewards

# --- Helpers ---
def create_rllib_env(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (env_config.worker_index * env_config.get("num_envs_per_worker", 1) + env_index)
    env = soccer_twos.make(**env_config)
    return RLLibWrapper(env)
""" INCASE UNITY ISSUES
# Unity requires a unique worker_id/port for every concurrent instance.
    # Ray provides worker_index and vector_index in the config.
    if "worker_index" in env_config:
        # Calculate a unique ID: (WorkerID * EnvsPerWorker) + LocalEnvIndex
        env_config["worker_id"] = (
            env_config["worker_index"] * env_config.get("num_envs_per_worker", 1) + 
            env_config.get("vector_index", 0)
        )
    
    # Clean up keys that soccer_twos.make doesn't recognize
    spawn_config = {k: v for k, v in env_config.items() 
                    if k not in ["worker_index", "vector_index", "num_envs_per_worker"]}
    
    env = soccer_twos.make(**spawn_config)
    return RLLibWrapper(env)

"""

def sample_vec(range_dict): return [randfloat(range_dict["x"][0], range_dict["x"][1]), randfloat(range_dict["y"][0], range_dict["y"][1])]
def sample_val(range_tpl): return randfloat(range_tpl[0], range_tpl[1])
def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict: _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict: _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s
def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict: _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s