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
GOAL_POS_TEAM_A  = np.array([ 16.0, 0.0], dtype=np.float32)  # Team A attacks +x
GOAL_POS_TEAM_B  = np.array([-16.0, 0.0], dtype=np.float32)  # Team B attacks -x
FIELD_DIAGONAL   = float(np.linalg.norm([32.0, 10.0]))        # for polar normalization

# Team layout (2v2)
TEAM_A = [0, 1]
TEAM_B = [2, 3]

# ---------------------------
# Reward shaping weights
# (inspired by Samtani et al. 2021 — arXiv:2103.05174)
# ---------------------------
GOAL_REWARD_SCALE    = 10.0   # scale sparse goal signal so it dominates shaping
BALL_DIR_WEIGHT      = 1.2    # ball moving toward opponent goal   (post-touch)
OWN_GOAL_WEIGHT      = 1.0    # ball moving toward own goal penalty (post-touch)
PLAYER_TO_BALL_WEIGHT= 0.1    # player approaching ball            (pre-touch)
FIRST_TOUCH_BONUS    = 0.5    # large one-time bonus on very first touch
POSSESSION_BONUS     = 0.3    # one-time bonus on gaining possession
TIME_PENALTY         = -0.05  # per-step penalty → encourages fast scoring
TOUCH_RADIUS         = 0.5    # distance threshold to count as touching

# Clip per-step shaping to prevent large negative accumulation
BALL_PROGRESS_CLIP   = 0.5
PLAYER_PROGRESS_CLIP = 0.2


# ---------------------------
# Polar normalization helper
# ---------------------------
def to_polar_norm(vec: np.ndarray, max_dist: float = FIELD_DIAGONAL) -> np.ndarray:
    """
    Convert a 2D Cartesian vector to a normalized polar representation:
        [distance / max_dist,  angle / (2π)]
    More robust than raw (x, y) — the network doesn't need to learn
    that (3,4) and (5,0) are the same distance from origin.
    """
    dist  = float(np.linalg.norm(vec)) / max_dist
    angle = float(np.arctan2(vec[1], vec[0])) / (2.0 * np.pi)
    return np.array([dist, angle], dtype=np.float32)


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(self, env):
        super().__init__(env)

        # Curriculum task index — set by train.py callback via set_task()
        self.current_task = 0

        # Per-agent episode state (reset on every reset())
        self.prev_infos      = {}   # infos from previous step
        self.was_touching    = {}   # bool: was agent touching ball last step
        self.ever_touched    = {}   # bool: has agent touched ball this episode
        self.in_possession   = {}   # bool: did agent have possession last step

        # Observation space: original raycasts + 6 polar features
        #   polar(rel_ball)(2) + polar(rel_goal)(2) + polar(rel_teammate)(2)
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
        """Goal the agent is attacking toward."""
        return GOAL_POS_TEAM_A if agent_id in TEAM_A else GOAL_POS_TEAM_B

    @staticmethod
    def _get_own_goal(agent_id: int) -> np.ndarray:
        """Goal the agent is defending."""
        return GOAL_POS_TEAM_B if agent_id in TEAM_A else GOAL_POS_TEAM_A

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        obs = self.env.reset()
        self.prev_infos    = {}
        self.was_touching  = {}
        self.ever_touched  = {}
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
    # Appends 6 polar-normalized features:
    #   polar(rel_ball)(2) + polar(rel_goal)(2) + polar(rel_teammate)(2)
    # ------------------------------------------------------------------
    def _modify_obs(self, obs_dict: dict, infos: dict) -> dict:
        new_obs = {}
        for agent_id, agent_obs in obs_dict.items():
            if agent_id not in infos:
                # Frame-0 fallback: zeros safe for one step
                new_obs[agent_id] = np.concatenate(
                    [agent_obs, np.zeros(6, dtype=np.float32)]
                )
                continue

            info        = infos[agent_id]
            p_pos       = np.array(info["player_info"]["position"], dtype=np.float32)
            b_pos       = np.array(info["ball_info"]["position"],   dtype=np.float32)
            opp_goal    = self._get_opp_goal(agent_id)
            teammate_id = self._get_teammate_id(agent_id)

            # Fallback to own position if teammate info is missing
            t_pos = np.array(
                infos.get(teammate_id, {})
                     .get("player_info", {})
                     .get("position", p_pos),
                dtype=np.float32,
            )

            polar_ball     = to_polar_norm(b_pos    - p_pos)
            polar_goal     = to_polar_norm(opp_goal - p_pos)
            polar_teammate = to_polar_norm(t_pos    - p_pos)

            new_obs[agent_id] = np.concatenate(
                [agent_obs, polar_ball, polar_goal, polar_teammate]
            ).astype(np.float32)

        return new_obs

    # ------------------------------------------------------------------
    # Reward shaping
    # Implements the three-phase state machine from Samtani et al. 2021:
    #
    #   Phase 0 — ball never touched this episode:
    #       reward = player_to_ball_progress + time_penalty
    #       (just get to the ball; ball direction is noise at this stage)
    #
    #   Phase 1 — ball touched for the very first time:
    #       reward = FIRST_TOUCH_BONUS + time_penalty
    #       (celebrate first contact; transition to Phase 2)
    #
    #   Phase 2 — ball has been touched before:
    #       reward = ball_toward_opp_goal - ball_toward_own_goal
    #                + possession_bonus + time_penalty
    #       (now care about ball direction; penalize own-goals)
    # ------------------------------------------------------------------
    def _modify_rewards(self, reward_dict: dict, infos: dict) -> dict:
        shaped = {}

        for agent_id, base_reward in reward_dict.items():
            info      = infos.get(agent_id)
            prev_info = self.prev_infos.get(agent_id, info)

            if not info or not prev_info:
                shaped[agent_id] = base_reward
                continue

            # --- Positions ---
            ball_curr   = np.array(info["ball_info"]["position"],       dtype=np.float32)
            player_curr = np.array(info["player_info"]["position"],     dtype=np.float32)
            ball_prev   = np.array(prev_info["ball_info"]["position"],  dtype=np.float32)
            player_prev = np.array(prev_info["player_info"]["position"],dtype=np.float32)
            opp_goal    = self._get_opp_goal(agent_id)
            own_goal    = self._get_own_goal(agent_id)

            # --- Distances ---
            d_pb_curr  = float(np.linalg.norm(ball_curr - player_curr))
            d_pb_prev  = float(np.linalg.norm(ball_prev - player_prev))
            d_bg_curr  = float(np.linalg.norm(ball_curr - opp_goal))
            d_bg_prev  = float(np.linalg.norm(ball_prev - opp_goal))
            d_og_curr  = float(np.linalg.norm(ball_curr - own_goal))
            d_og_prev  = float(np.linalg.norm(ball_prev - own_goal))

            # --- Touch state ---
            touching         = d_pb_curr < TOUCH_RADIUS
            was_touching     = self.was_touching.get(agent_id, False)
            ever_touched     = self.ever_touched.get(agent_id, False)
            first_touch_now  = touching and not ever_touched

            if touching:
                self.ever_touched[agent_id] = True
            self.was_touching[agent_id] = touching

            # --- Possession (touching AND closest player to ball) ---
            opp_ids   = TEAM_B if agent_id in TEAM_A else TEAM_A
            opp_dists = [
                float(np.linalg.norm(
                    np.array(infos[oid]["player_info"]["position"], dtype=np.float32)
                    - ball_curr
                ))
                for oid in opp_ids if oid in infos
            ]
            d_closest_opp = min(opp_dists) if opp_dists else float("inf")
            in_pos        = touching and (d_pb_curr <= d_closest_opp)
            had_pos       = self.in_possession.get(agent_id, False)

            possession_bonus = POSSESSION_BONUS if (in_pos and not had_pos) else 0.0
            self.in_possession[agent_id] = in_pos

            # ----------------------------------------------------------
            # Three-phase reward state machine (Samtani et al. 2021)
            # ----------------------------------------------------------
            if first_touch_now:
                # Phase 1: first contact this episode
                shaping = FIRST_TOUCH_BONUS

            elif not ever_touched:
                # Phase 0: ball never touched — only reward closing distance
                player_progress = np.clip(
                    (d_pb_prev - d_pb_curr) * PLAYER_TO_BALL_WEIGHT,
                    -PLAYER_PROGRESS_CLIP,
                    PLAYER_PROGRESS_CLIP,
                )
                shaping = player_progress

            else:
                # Phase 2: ball has been touched — reward ball direction
                # Penalize moving ball toward own goal (own-goal prevention)
                ball_progress = np.clip(
                    BALL_DIR_WEIGHT   * (d_bg_prev - d_bg_curr)
                    - OWN_GOAL_WEIGHT * (d_og_prev - d_og_curr),
                    -BALL_PROGRESS_CLIP,
                    BALL_PROGRESS_CLIP,
                )
                shaping = ball_progress + possession_bonus

            # Time pressure — discourages passive play
            shaping += TIME_PENALTY

            # Scale sparse goal reward so it dominates all shaping
            shaped[agent_id] = base_reward * GOAL_REWARD_SCALE + shaping

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