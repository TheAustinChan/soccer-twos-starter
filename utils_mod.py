import gym
import numpy as np
from gym.spaces import Box
from ray.rllib import MultiAgentEnv
import soccer_twos
from random import uniform as randfloat

class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(self, env):
        super().__init__(env)
        # 336 (Sensors) + 6 (Rel Ball, Rel Goal, Rel Teammate)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(342,), dtype=np.float32
        )
        self.env_channel = getattr(self.env, "env_channel", None)
        self.prev_infos = {}
        self.current_task_id = 0

    def set_task(self, task_id):
        """Standard method used by CombinedCallback to sync curriculum state."""
        self.current_task_id = task_id

    def reset(self):
        obs = self.env.reset()
        self.prev_infos = {}
        return self._modify_obs(obs, {})

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        rewards = self._modify_rewards(rewards, infos)
        obs = self._modify_obs(obs, infos)
        self.prev_infos = infos
        return obs, rewards, dones, infos

    def _modify_obs(self, obs_dict, infos):
        new_obs = {}
        for agent_id, agent_obs in obs_dict.items():
            truncated = agent_obs[:336]
            if agent_id not in infos:
                new_obs[agent_id] = np.concatenate([truncated, np.zeros(6)])
                continue
            
            info = infos[agent_id]
            p_pos = np.array(info["player_info"]["position"])
            b_pos = np.array(info["ball_info"]["position"])
            
            # Global goal positions
            opp_goal = np.array([16.0, 0.0]) if agent_id < 2 else np.array([-16.0, 0.0])
            tid = 1 - agent_id if agent_id < 2 else 5 - agent_id
            t_info = infos.get(tid, {}).get("player_info", {})
            t_pos = np.array(t_info.get("position", p_pos))

            extra = np.concatenate([b_pos - p_pos, opp_goal - p_pos, t_pos - p_pos])
            new_obs[agent_id] = np.concatenate([truncated, extra]).astype(np.float32)
        return new_obs

    def _modify_rewards(self, reward_dict, infos):
        shaped = {}
        # Normalization constant: Field is ~32 units wide (-16 to 16)
        # All distance-based rewards are divided by D_MAX to ensure they scale
        # as a 'percentage of field progress' rather than raw meters.

        # introduced some arbitrary small weights to account for max timesteps we fix here as 5000
        # more sophistication would include alignment rewards on shooting in path to goal and obs
        # that would take into account shooting or passing towards an opponent who intercepts
        D_MAX = 32.0 

        for agent_id, base_reward in reward_dict.items():
            info = infos.get(agent_id)
            prev = self.prev_infos.get(agent_id)
            
            if not info or not prev:
                shaped[agent_id] = base_reward
                continue

            # --- 1. COORDINATE FETCHING ---
            b_pos, b_prev = np.array(info["ball_info"]["position"]), np.array(prev["ball_info"]["position"])
            p_pos, p_prev = np.array(info["player_info"]["position"]), np.array(prev["player_info"]["position"])
            opp_goal = np.array([16.0, 0.0]) if agent_id < 2 else np.array([-16.0, 0.0])
            own_goal = np.array([-16.0, 0.0]) if agent_id < 2 else np.array([16.0, 0.0])
            
            tid = 1 - agent_id if agent_id < 2 else 5 - agent_id
            t_info = infos.get(tid)
            tp_pos = np.array(t_info["player_info"]["position"]) if t_info else p_pos
            tp_prev = np.array(self.prev_infos.get(tid, {}).get("player_info", {}).get("position", tp_pos))

            # --- 2. THE APPROACH SUM (D_BA) ---
            # Ego weight (1.0) vs Altruism weight (0.2)
            # This encourages the agent to find the ball, but also rewards them
            # slightly for their teammate's progress to prevent redundant 'clumping'.
            # Scaled by 0.05 to ensure this is a minor nudge, not a primary objective.
            d_ba_ego = (np.linalg.norm(p_prev - b_prev) - np.linalg.norm(p_pos - b_pos)) / D_MAX
            d_ba_alt = ((np.linalg.norm(tp_prev - b_prev) - np.linalg.norm(tp_pos - b_pos)) / D_MAX) * 0.5
            approach_sum = (d_ba_ego + d_ba_alt) * 0.05

            # --- 3. THE BEHAVIORAL MAX SWITCH ---
            # We select the highest value behavior to prevent 'farming' multiple roles.
            # Magnitudes are ~0.1 to 0.2, ensuring the Goal (1.0) remains the dominant signal.
            
            # A) SCORING (D_GB): Ball progress toward opponent's goal
            reward_scoring = (np.linalg.norm(b_prev - opp_goal) - np.linalg.norm(b_pos - opp_goal)) / D_MAX

            # B) PASSING/COOPERATION (D_AiB): Ball progress toward teammate
            # This makes passing a mathematically viable alternative to solo dribbling.
            reward_passing = (np.linalg.norm(b_prev - tp_prev) - np.linalg.norm(b_pos - tp_pos)) / D_MAX

            # C) DEFENSE (Clearance): Penalty if near own goal, reward for clearing
            # This activates when ball is within 8 units (25% of field) of own goal.
            dist_to_own = np.linalg.norm(b_pos - own_goal)
            reward_defense = 0.0
            if dist_to_own < 8.0:
                # Defense has slightly higher weight (0.15 vs 0.1) to prioritize safety
                reward_defense = ((dist_to_own - np.linalg.norm(b_prev - own_goal)) / D_MAX) * 1.5

            # The Switch: Chooses the most contextually relevant strategy.
            # This logic mimics 'regime changes' (e.g., flipping from Striker to Defender).
            strategy_reward = max(reward_scoring, reward_passing, reward_defense) * 0.15

            # --- 4. FINAL REWARD ASSEMBLY ---
            # base_reward (G) = (1 - steps/5000) upon scoring.
            # Shaping per step totals ~0.005 to 0.02, preventing it from outweighing G.
            shaped[agent_id] = base_reward + approach_sum + strategy_reward

        return shaped

# ---------------------------
# Curriculum Helper Functions
# ---------------------------

def sample_vec(r):
    """Samples a 2D vector within specified x and y ranges."""
    return [randfloat(r["x"][0], r["x"][1]), randfloat(r["y"][0], r["y"][1])]

def sample_pos_vel(r):
    """Samples position and velocity from curriculum ranges."""
    res = {}
    if "position" in r:
        res["position"] = sample_vec(r["position"])
    if "velocity" in r:
        res["velocity"] = sample_vec(r["velocity"])
    return res

def sample_player(r):
    """Samples a full player state including rotation."""
    res = sample_pos_vel(r)
    if "rotation_y" in r:
        res["rotation_y"] = randfloat(r["rotation_y"][0], r["rotation_y"][1])
    return res

def create_rllib_env(env_config={}):
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

