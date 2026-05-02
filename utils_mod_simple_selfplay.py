from random import uniform as randfloat

import numpy as np
import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import soccer_twos

# ---------------------------
# Field constants
# ---------------------------
# Field: x=[-16, 16], y=[-5, 5]
# Goal centers at x=±16, y=0
GOAL_POS_TEAM_A = np.array([ 16.0, 0.0], dtype=np.float32)  # Team A attacks +x
GOAL_POS_TEAM_B = np.array([-16.0, 0.0], dtype=np.float32)  # Team B attacks -x

# Team layout (2v2)
TEAM_A = [0, 1]
TEAM_B = [2, 3]

# ---------------------------
# Reward weights
#
# Philosophy: keep shaping minimal so it never dominates the goal signal.
# Only two shaping terms to solve the cold-start problem:
#   1. ball toward opponent goal  — directional push
#   2. agent approaching ball     — get agents moving early
#
# DO NOT add: possession, touch bonus, passing, time penalty.
# These cause reward hacking in zero-sum self-play.
# ---------------------------
GOAL_REWARD         = 10.0  # scales base_reward (±1) — always dominant
BALL_TO_GOAL_WEIGHT =  1.0  # ball moving toward opponent goal
BALL_TO_GOAL_CLIP   =  0.3  # hard cap — shaping never dominates goal signal
APPROACH_WEIGHT     =  0.1  # agent approaching ball
APPROACH_CLIP       =  0.1  # hard cap


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(self, env):
        super().__init__(env)

        # Per-agent episode state
        self.prev_infos = {}

        # Observation space unchanged — raycasts already encode
        # ball, teammate, opponent, and goal positions spatially.

    # ------------------------------------------------------------------
    # Team helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_opp_goal(agent_id: int) -> np.ndarray:
        return GOAL_POS_TEAM_A if agent_id in TEAM_A else GOAL_POS_TEAM_B

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        obs = self.env.reset()
        self.prev_infos = {}
        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action_dict):
        if not action_dict:
            return {}, {}, {"__all__": False}, {}

        obs, rewards, dones, infos = self.env.step(action_dict)

        # Modify rewards BEFORE updating prev_infos
        rewards = self._modify_rewards(rewards, infos)

        self.prev_infos = infos
        return obs, rewards, dones, infos

    # ------------------------------------------------------------------
    # Reward — minimal self-play shaping
    #
    # R = G                                  (goal signal — dominant)
    #   + clip(ball_toward_opp_goal, ±0.3)   (directional push)
    #   + clip(approach_ball,        ±0.1)   (cold start)
    #
    # Zero-sum by design:
    #   - G is +1/-1 scaled — already opposite for each team
    #   - ball_progress uses each team's own goal position so it's
    #     naturally opposite: good for Team A = bad for Team B
    # ------------------------------------------------------------------
    def _modify_rewards(self, reward_dict: dict, infos: dict) -> dict:
        shaped = {}

        # --- Compute global ball progress (zero-sum) ---
        # Use Team B goal as reference direction
        any_agent = next(iter(infos))
        info_any = infos[any_agent]
        prev_any = self.prev_infos.get(any_agent, info_any)

        ball_curr = np.array(info_any["ball_info"]["position"], dtype=np.float32)
        ball_prev = np.array(prev_any["ball_info"]["position"], dtype=np.float32)

        # Distance to Team B goal (x = -16)
        d_prev = np.linalg.norm(ball_prev - GOAL_POS_TEAM_B)
        d_curr = np.linalg.norm(ball_curr - GOAL_POS_TEAM_B)

        # Positive if moving toward Team B goal
        ball_progress_global = np.clip(
            (d_prev - d_curr) * BALL_TO_GOAL_WEIGHT,
            -BALL_TO_GOAL_CLIP,
            BALL_TO_GOAL_CLIP,
        )
        # In utils_mod.py _modify_rewards, temporarily add:
        if infos and any(infos.values()):
            print(f"SAMPLE INFO: {next(iter(infos.values()))}")


        for agent_id, base_reward in reward_dict.items():
            info      = infos.get(agent_id)
            prev_info = self.prev_infos.get(agent_id, info)

            if not info or not prev_info:
                shaped[agent_id] = base_reward * GOAL_REWARD
                continue

            # --- Positions ---
            player_curr = np.array(info["player_info"]["position"], dtype=np.float32)
            player_prev = np.array(prev_info["player_info"]["position"], dtype=np.float32)

            ball_curr = np.array(info["ball_info"]["position"], dtype=np.float32)
            ball_prev = np.array(prev_info["ball_info"]["position"], dtype=np.float32)

            # --- Goal reward ---
            G = base_reward * GOAL_REWARD

            # --- Zero-sum ball shaping ---
            if agent_id in TEAM_A:
                ball_progress = +ball_progress_global
            else:
                ball_progress = -ball_progress_global

            # --- Approach reward (fixed) ---
            d_to_ball_prev = np.linalg.norm(ball_prev - player_prev)
            d_to_ball_curr = np.linalg.norm(ball_curr - player_curr)

            approach = np.clip(
                (d_to_ball_prev - d_to_ball_curr) * APPROACH_WEIGHT,
                -APPROACH_CLIP,
                APPROACH_CLIP,
            )

            shaped[agent_id] = G + ball_progress + approach

        return shaped


# ---------------------------
# Env factory for RLLib
# ---------------------------
def create_rllib_env(env_config: dict = {}):
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
# Sampling helpers
# ---------------------------
def sample_vec(range_dict: dict) -> list:
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]

def sample_val(range_tpl: list) -> float:
    return randfloat(range_tpl[0], range_tpl[1])

def sample_pos_vel(range_dict: dict) -> dict:
    result = {}
    if "position" in range_dict:
        result["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        result["velocity"] = sample_vec(range_dict["velocity"])
    return result

def sample_player(range_dict: dict) -> dict:
    result = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        result["rotation_y"] = sample_val(range_dict["rotation_y"])
    return result