from random import uniform as randfloat
 
import numpy as np
import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import soccer_twos
 
# ---------------------------
# Field constants
# ---------------------------
# Field: x=[-14, 14], y=[-5, 5]
# Goal posts sit outside the field boundary at x=±16
GOAL_POS_TEAM_A = np.array([16.0, 0.0], dtype=np.float32)   # Team A attacks toward +x
GOAL_POS_TEAM_B = np.array([-16.0, 0.0], dtype=np.float32)  # Team B attacks toward -x
 
# Team layout (2v2)
TEAM_A = [0, 1]
TEAM_B = [2, 3]
 
# ---------------------------
# Reward shaping weights
# ---------------------------
GOAL_REWARD_SCALE     = 10.0  # Scale sparse goal reward so it dominates shaping
BALL_TO_GOAL_WEIGHT   = 1.0   # Ball moving toward opponent goal
PLAYER_TO_BALL_WEIGHT = 0.1   # Player moving toward ball (small — avoid hovering)
TOUCH_BONUS           = 0.3   # One-time bonus for making contact with ball
POSSESSION_BONUS      = 0.3   # One-time bonus for gaining possession (not per-step)
TOUCH_RADIUS          = 0.5   # Distance threshold to count as touching ball
 
# Clip shaping per step so large negative values don't swamp the goal signal
BALL_PROGRESS_CLIP   = 0.5    # max abs shaping from ball-to-goal per step
PLAYER_PROGRESS_CLIP = 0.2    # max abs shaping from player-to-ball per step
 
 
class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(self, env):
        super().__init__(env)
 
        # Curriculum task index — set by train.py callback via set_task()
        self.current_task = 0
 
        # Touch / possession tracking (reset each episode)
        self.prev_infos    = {}
        self.was_touching  = {}
        self.in_possession = {}
 
        # Observation space: original raycasts + 6 relative features
        #   rel_ball (2) + rel_goal (2) + rel_teammate (2)
        orig_shape = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(orig_shape + 6,),
            dtype=np.float32,
        )
 
    # ------------------------------------------------------------------
    # Curriculum task setter — called by train.py foreach_worker callback
    # ------------------------------------------------------------------
    def set_task(self, task_id: int):
        self.current_task = task_id
 
    # ------------------------------------------------------------------
    # Team helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_teammate_id(agent_id: int) -> int:
        team = TEAM_A if agent_id in TEAM_A else TEAM_B
        return [i for i in team if i != agent_id][0]
 
    @staticmethod
    def _get_opp_goal(agent_id: int) -> np.ndarray:
        """Return the goal the agent is attacking toward."""
        return GOAL_POS_TEAM_A if agent_id in TEAM_A else GOAL_POS_TEAM_B
 
    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        obs = self.env.reset()
        self.prev_infos    = {}
        self.was_touching  = {}
        self.in_possession = {}
        infos = getattr(self.env, "prev_infos", {})
        return self._modify_obs(obs, infos)
 
    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action_dict):
        if not action_dict:
            return {}, {}, {"__all__": False}, {}
 
        obs, rewards, dones, infos = self.env.step(action_dict)
 
        # Modify rewards BEFORE updating prev_infos (needs previous frame)
        rewards = self._modify_rewards(rewards, infos)
        obs     = self._modify_obs(obs, infos)
 
        self.prev_infos = infos
        return obs, rewards, dones, infos
 
    # ------------------------------------------------------------------
    # Observation augmentation
    # Appends 6 relative features: rel_ball(2) + rel_goal(2) + rel_teammate(2)
    # ------------------------------------------------------------------
    def _modify_obs(self, obs_dict: dict, infos: dict) -> dict:
        new_obs = {}
        for agent_id, agent_obs in obs_dict.items():
            if agent_id not in infos:
                # Frame 0 fallback — zeros are safe for one step
                new_obs[agent_id] = np.concatenate(
                    [agent_obs, np.zeros(6, dtype=np.float32)]
                )
                continue
 
            info        = infos[agent_id]
            p_pos       = np.array(info["player_info"]["position"],  dtype=np.float32)
            b_pos       = np.array(info["ball_info"]["position"],    dtype=np.float32)
            opp_goal    = self._get_opp_goal(agent_id)
            teammate_id = self._get_teammate_id(agent_id)
 
            # Fallback to own position if teammate info is missing
            t_pos = np.array(
                infos.get(teammate_id, {})
                     .get("player_info", {})
                     .get("position", p_pos),
                dtype=np.float32,
            )
 
            rel_ball     = b_pos    - p_pos
            rel_goal     = opp_goal - p_pos
            rel_teammate = t_pos    - p_pos
 
            new_obs[agent_id] = np.concatenate(
                [agent_obs, rel_ball, rel_goal, rel_teammate]
            ).astype(np.float32)
 
        return new_obs
 
    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------
    def _modify_rewards(self, reward_dict: dict, infos: dict) -> dict:
        shaped = {}
 
        for agent_id, base_reward in reward_dict.items():
            info      = infos.get(agent_id)
            prev_info = self.prev_infos.get(agent_id, info)  # first-step fallback
 
            if not info or not prev_info:
                shaped[agent_id] = base_reward
                continue
 
            # --- Positions ---
            ball_curr   = np.array(info["ball_info"]["position"],       dtype=np.float32)
            player_curr = np.array(info["player_info"]["position"],     dtype=np.float32)
            ball_prev   = np.array(prev_info["ball_info"]["position"],  dtype=np.float32)
            player_prev = np.array(prev_info["player_info"]["position"],dtype=np.float32)
            opp_goal    = self._get_opp_goal(agent_id)
 
            # --- Distances ---
            d_pb_curr = np.linalg.norm(ball_curr - player_curr)
            d_pb_prev = np.linalg.norm(ball_prev - player_prev)
            d_bg_curr = np.linalg.norm(ball_curr - opp_goal)
            d_bg_prev = np.linalg.norm(ball_prev - opp_goal)
 
            # --- Ball progress toward goal (clipped to avoid large negatives) ---
            ball_progress = np.clip(
                (d_bg_prev - d_bg_curr) * BALL_TO_GOAL_WEIGHT,
                -BALL_PROGRESS_CLIP,
                BALL_PROGRESS_CLIP,
            )
 
            # --- Player progress toward ball (clipped) ---
            player_progress = np.clip(
                (d_pb_prev - d_pb_curr) * PLAYER_TO_BALL_WEIGHT,
                -PLAYER_PROGRESS_CLIP,
                PLAYER_PROGRESS_CLIP,
            )
 
            # --- Touch bonus (one-time, fires only on contact start) ---
            touching    = d_pb_curr < TOUCH_RADIUS
            touch_bonus = TOUCH_BONUS if (touching and not self.was_touching.get(agent_id, False)) else 0.0
            self.was_touching[agent_id] = touching
 
            # --- Possession bonus (one-time, fires only when possession is gained) ---
            # Possession = touching AND closer to ball than any opponent
            opp_ids   = TEAM_B if agent_id in TEAM_A else TEAM_A
            opp_dists = [
                np.linalg.norm(
                    np.array(infos[oid]["player_info"]["position"], dtype=np.float32)
                    - ball_curr
                )
                for oid in opp_ids if oid in infos
            ]
            d_closest_opp = min(opp_dists) if opp_dists else float("inf")
            in_pos        = touching and (d_pb_curr <= d_closest_opp)
 
            # Only reward gaining possession, NOT holding it per-step
            possession_bonus = (
                POSSESSION_BONUS
                if (in_pos and not self.in_possession.get(agent_id, False))
                else 0.0
            )
            self.in_possession[agent_id] = in_pos
 
            # --- Combine ---
            # base_reward scaled up so sparse goal signal dominates shaping
            shaped[agent_id] = (
                base_reward * GOAL_REWARD_SCALE
                + ball_progress
                + player_progress
                + touch_bonus
                + possession_bonus
            )
 
        return shaped
 
 
# ---------------------------
# Env factory for RLLib
# ---------------------------
def create_rllib_env(env_config: dict = {}):
    """
    Creates and wraps a soccer_twos environment for RLLib.
    env_config keys:
        worker_index        : set by RLLib automatically
        num_envs_per_worker : used to compute unique worker_id
        base_port           : base Unity port
        multiagent          : set False to disable multiagent wrapper
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
 
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
 
    return RLLibWrapper(env)
 
 
# ---------------------------
# Curriculum sampling helpers
# ---------------------------
def sample_vec(range_dict: dict) -> list:
    """Sample a 2D [x, y] vector from a range dict with 'x' and 'y' keys."""
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]
 
 
def sample_val(range_tpl: list) -> float:
    """Sample a scalar from a [min, max] list."""
    return randfloat(range_tpl[0], range_tpl[1])
 
 
def sample_pos_vel(range_dict: dict) -> dict:
    """Sample position and/or velocity from a range dict."""
    result = {}
    if "position" in range_dict:
        result["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        result["velocity"] = sample_vec(range_dict["velocity"])
    return result
 
 
def sample_player(range_dict: dict) -> dict:
    """Sample position, velocity, and/or rotation_y for a player."""
    result = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        result["rotation_y"] = sample_val(range_dict["rotation_y"])
    return result