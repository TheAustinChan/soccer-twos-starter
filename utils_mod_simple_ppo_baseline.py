from random import uniform as randfloat

import numpy as np
import gym
from ray.rllib import MultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import soccer_twos

# ---------------------------
# Field constants
# ---------------------------
FIELD_DIAGONAL = np.linalg.norm([32.0, 10.0])  # ~33.5, used for normalization

# Goal positions (world coords)
# team l scores IN GOAL_POS[l]  →  D^l = dist(ball, GOAL_POS[l])
GOAL_POS = {
    0: np.array([ 16.0, 0.0], dtype=np.float32),  # team 0 attacks +x goal
    1: np.array([-16.0, 0.0], dtype=np.float32),  # team 1 attacks -x goal
}

AGENT_TEAM = {0: 0, 1: 0, 2: 1, 3: 1}

LAMBDA = 0.03  # normalised kickable-distance threshold


def _nd(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance normalised by field diagonal."""
    return float(np.linalg.norm(a[:2] - b[:2])) / FIELD_DIAGONAL


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    Dense reward — verbatim from arXiv:2103.05174, Eqs. (1)-(3).

    Variable names match the paper exactly:

        d^i_t      normalised dist of agent i to ball at step t
        D^l_t      normalised dist of ball to goalpost team l scores IN  (opponent goal)
        D^(1-l)_t  normalised dist of ball to team l's OWN goalpost
        Δξ_t       ξ_t − ξ_{t−1}
        b^i_t      True if ball was within λ of agent i at some t* < t
        k^i_t      True if b^i_t is False AND d^i_t ≤ λ

    Eq (1):
        r_t = r✗_t   if no goal scored
              r✓_t   if goal scored

    Eq (2):
        r✗_t = β − 0.1                                          if k^i_t
               1.2·(ΔD^(1−l)_t − ΔD^l_t) − Δd^i_t − 0.1      if b^i_t
              −Δd^i_t − 0.1                                     otherwise

    Eq (3):
        r✓_t = +α   goal scored in team (1−l)'s goalpost  [agent's team scored]
               −α   goal scored in team  l's  goalpost    [agent's team conceded]
    """

    def __init__(self, env, max_steps: int = 3000):
        super().__init__(env)

        # --- ADD THESE LINES ---
        # RLlib looks for these attributes specifically on the wrapper 
        # to define the Policy's neural network input/output shapes.
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # -----------------------

        self.alpha = max_steps / 10.0
        self.beta  = self.alpha / 10.0

        self._b         = {}   # b^i_t flag per agent
        self._prev_pos  = {}   # agent positions from previous step
        self._prev_ball = None # ball position from previous step

    def reset(self):
        obs = self.env.reset()
        self._b         = {i: False for i in range(4)}
        self._prev_pos  = {}
        self._prev_ball = None
        return obs

    def step(self, action_dict):
        if not action_dict:
            return {}, {}, {"__all__": False}, {}
        obs, base_rewards, dones, infos = self.env.step(action_dict)
        return obs, base_rewards, dones, infos

    def _shaped(self, base_rewards, infos):
        out = {}

        any_info  = infos[next(iter(infos))]
        ball      = np.array(any_info["ball_info"]["position"], dtype=np.float32)
        ball_prev = self._prev_ball if self._prev_ball is not None else ball.copy()

        for i, base in base_rewards.items():
            info = infos.get(i)
            if not info:
                out[i] = 0.0
                continue

            l         = AGENT_TEAM[i]
            agent     = np.array(info["player_info"]["position"], dtype=np.float32)
            agent_pre = self._prev_pos.get(i, agent.copy())

            # d^i_t  and  d^i_{t-1}
            d_t   = _nd(agent,     ball)
            d_tm1 = _nd(agent_pre, ball_prev)

            # D^l_t  (ball → opponent goal, i.e. the goal team l is trying to score in)
            D_l_t   = _nd(ball,      GOAL_POS[l])
            D_l_tm1 = _nd(ball_prev, GOAL_POS[l])

            # D^(1-l)_t  (ball → team l's own goal)
            D_1l_t   = _nd(ball,      GOAL_POS[1 - l])
            D_1l_tm1 = _nd(ball_prev, GOAL_POS[1 - l])

            # Δ values
            delta_d   = d_t    - d_tm1
            delta_Dl  = D_l_t  - D_l_tm1
            delta_D1l = D_1l_t - D_1l_tm1

            # --- Eq (1) ---
            if base != 0.0:
                # --- Eq (3) ---
                r = self.alpha if base > 0 else -self.alpha
            else:
                # --- Eq (2) ---
                k_t = (not self._b[i]) and (d_t <= LAMBDA)
                b_t = self._b[i]

                if k_t:
                    r = self.beta - 0.1
                    self._b[i] = True
                elif b_t:
                    r = 1.2 * (delta_D1l - delta_Dl) - delta_d - 0.1
                else:
                    r = -delta_d - 0.1

            out[i] = r
            self._prev_pos[i] = agent.copy()

        self._prev_ball = ball.copy()
        return out


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

    max_steps = (
        getattr(env, "_max_episode_steps", None)
        or getattr(env.spec, "max_episode_steps", None)
        or 3000
    )

    return RLLibWrapper(env, max_steps=max_steps)


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