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
import sys
from . import utils_mod
 
sys.modules["utils_mod"] = utils_mod
 
import threading
_TLS = threading.local()
 
def _set_infos(infos):
    _TLS.last_infos = infos if isinstance(infos, dict) else {}
 
def _get_infos():
    return getattr(_TLS, "last_infos", {})
 
def _patch_env_chain(env):
    cur = env
    while cur is not None:
        cls = cur.__class__
        if not getattr(cls, "_team_infos_patch", False):
            orig_step = cls.step
            orig_reset = getattr(cls, "reset", None)
 
            def step_patched(self, action, _orig=orig_step):
                out = _orig(self, action)
                if isinstance(out, tuple) and len(out) == 4:
                    _set_infos(out[3])  # infos
                return out
 
            cls.step = step_patched
 
            if orig_reset is not None:
                def reset_patched(self, *args, _orig=orig_reset, **kwargs):
                    _set_infos({})
                    return _orig(self, *args, **kwargs)
                cls.reset = reset_patched
 
            cls._team_infos_patch = True
 
        cur = getattr(cur, "env", None)
 
 
 
ALGORITHM = "PPO"
# CHECKPOINT_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "ray_results\PPO_selfplay_rec_reg_mod_new\PPO_Soccer_0b4e9_00000_0_2026-04-24_04-17-33\checkpoint_000080\checkpoint_000080"
# )
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    #"ray_results\PPO_selfplay_full\PPO_Soccer_cfd20_00000_0_2026-04-26_05-25-06\PPO_Soccer_cfd20_00000_0_2026-04-26_05-25-06\checkpoint_001500\checkpoint-1500"
    "./ray_results/PPO_selfplay_full/PPO_Soccer_0e992_00000_0_2026-04-27_11-52-13/checkpoint_000600/checkpoint-600"
)
POLICY_NAME = "default"
 
 
 
 
class RayAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()
        #if not ray.is_initialized:
        ray.init(ignore_reinit_error=True)
        _patch_env_chain(env)
        # self.last_infos = {}
        # self.env = env
 
        # # Walk the wrapper chain and print everything for debugging
        # target = env
        # depth = 0
        # while True:
        #     print(f"[{depth}] type: {type(target)}")
        #     print(f"[{depth}] attrs with 'info': {[x for x in dir(target) if 'info' in x.lower()]}")
        #     print(f"[{depth}] attrs with 'step': {[x for x in dir(target) if 'step' in x.lower()]}")
        #     if hasattr(target, '__dict__'):
        #         print(f"[{depth}] __dict__ keys: {list(target.__dict__.keys())}")
        #     if hasattr(target, 'env'):
        #         target = target.env
        #         depth += 1
        #     else:
        #         break
 
        # # Keep reference to innermost env
        # env.step = patched_step
 
        # # Patch reset to clear stale infos
        # original_reset = env.reset
        # def patched_reset():
        #     self.last_infos = {}
        #     return original_reset()
        # env.reset = patched_reset
 
        # Load configuration from checkpoint
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")
 
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )
 
        config["num_workers"] = 0
        config["num_gpus"] = 0
 
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"
 
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        agent.restore(CHECKPOINT_PATH)
        self.policy = agent.get_policy(POLICY_NAME)
 
        self.teams = {
            0: [0,1],
            1: [2,3]
        }
 
 
    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        # # Print what we have for debugging
        # print(f"[act] last_infos empty: {self.last_infos == {}}")
        # print(f"[act] last_infos: {self.last_infos}")
 
        # # Try reading directly from inner env attributes
        # print(f"[act] inner_env __dict__: {self.inner_env.__dict__}")
        """
        infos = _get_infos()
        #print(infos)
        augmented_obs = self._augment_obs(observation, infos)
 
        actions = {}
        for player_id in augmented_obs:
            actions[player_id], *_ = self.policy.compute_single_action(
                augmented_obs[player_id]
            )
        return actions
        """
        infos = _get_infos()
        
        # Check if we are currently acting for the Orange team (IDs 2, 3)
        # by looking at the keys available in the global 'infos' dict.
        is_orange_team = 2 in infos or 3 in infos
        
        # Map the local watch.py IDs {0, 1} to their global reality {2, 3}
        global_obs = {}
        for local_id, obs in observation.items():
            global_id = local_id + 2 if is_orange_team else local_id
            global_obs[global_id] = obs

        # Now augment using the correct global IDs
        augmented_obs = self._augment_obs(global_obs, infos)

        # Return actions using the LOCAL IDs that watch.py expects
        actions = {}
        for global_id, full_obs in augmented_obs.items():
            local_id = global_id - 2 if is_orange_team else global_id
            actions[local_id], *_ = self.policy.compute_single_action(full_obs)
            
        return actions
    
    def _augment_obs(self, obs_dict: Dict, infos: Dict) -> Dict:
        """Mirrors RLLibWrapper._modify_obs exactly."""
        BASE_OBS_SIZE = 336
        new_obs = {}
        for agent_id, agent_obs in obs_dict.items():
            agent_obs = agent_obs[:BASE_OBS_SIZE]
            #print(agent_id)
            if agent_id not in infos:
                #print("Not in info, infos", infos)
                new_obs[agent_id] = np.concatenate(
                    [agent_obs, np.zeros(6, dtype=np.float32)]
                )
                continue
 
            info = infos[agent_id]
            p_pos = np.array(info["player_info"]["position"], dtype=np.float32)
            b_pos = np.array(info["ball_info"]["position"], dtype=np.float32)
 
            if agent_id < 2:
                opp_goal_pos = np.array([16.0, 0.0], dtype=np.float32)
                #teammate_id = 1 - agent_id
                #teammate_id = [i for i in self.team if i != agent_id][0]
                if agent_id == 0:
                    teammate_id = 1
                elif agent_id == 1:
                    teammate_id = 0
            else:
                opp_goal_pos = np.array([-16.0, 0.0], dtype=np.float32)
                #teammate_id = 5 - agent_id
                #teammate_id = [i for i in team if i != agent_id][0]
                if agent_id == 2:
                    teammate_id = 3
                elif agent_id == 3:
                    teammate_id = 2
 
            t_pos = np.array(
                infos.get(teammate_id, {})
                     .get("player_info", {})
                     .get("position", p_pos),
                dtype=np.float32
            )
 
            rel_ball = b_pos - p_pos
            rel_goal = opp_goal_pos - p_pos
            rel_team = t_pos - p_pos
 
            new_obs[agent_id] = np.concatenate(
                [agent_obs, rel_ball, rel_goal, rel_team]
            ).astype(np.float32)
 
        return new_obs