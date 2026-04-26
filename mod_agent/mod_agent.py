import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
#from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
#from utils_mod import create_rllib_env  # Ensure this imports correctly

"""
NOTICE: WATCH.PY IS THE ONE THAT DICTATES THE ENVIRONMENT AND IS WHAT WE CALL WITH python watch.py -m ... agents -m1, ...
agents, thereofre it determines/creates what environment by default is NOT utils_mod as we were able to dictate for 
training purposes as goal is agent in unfamiliar environment
"""

ALGORITHM = "PPO"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./ray_results/PPO_selfplay_rec_reg_mod_new/PPO_Soccer_0b4e9_00000_0_2026-04-24_04-17-33/checkpoint_000080/checkpoint-80",
)
POLICY_NAME = "default"  # Use this as the main policy

"""
class DummyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        self.agents = ["agent_0", "agent_1"]

    def reset(self):
        return {a: np.zeros(10, dtype=np.float32) for a in self.agents}

    def step(self, action_dict):
        obs = {a: np.zeros(10, dtype=np.float32) for a in self.agents}
        rewards = {a: 0.0 for a in self.agents}
        dones = {a: False for a in self.agents}
        dones["__all__"] = False
        infos = {a: {} for a in self.agents}
        return obs, rewards, dones, infos



#Dummy based on my wrapper
class DummyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        self.agents = [0, 1, 2, 3]

        # MUST match your wrapper output size
        #self.obs_size = 340  # or 340 depending on config Last test
        self._last_infos = {} 
        self.base_obs_size = 336
        self.extra_obs_size = 4
        self.final_obs_size = self.base_obs_size + self.extra_obs_size

        self.teams = {
            0: [0, 1],
            1: [2, 3],
        }

    def reset(self):
        #return {
        #    a: np.zeros(self.obs_size, dtype=np.float32)
        #    for a in self.agents
        #}
    
        base_obs = {
            a: np.zeros(self.base_obs_size, dtype=np.float32)
            for a in self.agents
        }

        #Fake infos so augmentation has something to read
        self._last_infos = self._fake_infos()

        #augment wrapper
        return self._augment_obs(base_obs)
    
        #obs = self.env.reset()
        #return self._modify_obs(obs)

    def step(self, action_dict):

        #obs = {
        #    a: np.zeros(self.obs_size, dtype=np.float32)
        #    for a in self.agents
        #}
        base_obs = {
            a: np.zeros(self.base_obs_size, dtype=np.float32)
            for a in self.agents
        }

        rewards = {a: 0.0 for a in self.agents}

        dones = {a: False for a in self.agents}
        dones["__all__"] = False

        # IMPORTANT: must include keys used in wrapper
        #infos = {
        #    a: {
        #        "ball_info": {
        #            "position": [0.0, 0.0]
        #        },
        #        "player_info": {
        #            "position": [0.0, 0.0],
        #            "velocity": [0.0, 0.0],
        #        }
        #    }
        #    for a in self.agents
        #}
        #newly
        infos = self._fake_infos()

        #Need to find way to access info dict in agent.py
        #Constructor should have access to env obj, so cache the infos dict in there. Then access the cached version from the act method
        #
        


        #NEW STMT: store infos for observation augmentation
        self._last_infos = infos
        
        # 2. Modify observations
        #obs = self._modify_obs(obs)
        obs = self._augment_obs(base_obs)

        return obs, rewards, dones, infos
    
    def _augment_obs(self, obs_dict):
        new_obs = {}

        for team in self.teams.values():
            for agent_id in team:
                base_obs = np.array(obs_dict[agent_id], dtype=np.float32)

                teammate_id = [i for i in team if i != agent_id][0]

                teammate_pos = np.zeros(2, dtype=np.float32)
                teammate_vel = np.zeros(2, dtype=np.float32)

                info = self._last_infos.get(teammate_id, None)

                if info and "player_info" in info:
                    p = info["player_info"]
                    teammate_pos = np.array(
                        p.get("position", [0, 0]), dtype=np.float32
                    )
                    teammate_vel = np.array(
                        p.get("velocity", [0, 0]), dtype=np.float32
                    )

                extra = np.concatenate([teammate_pos, teammate_vel])

                new_obs[agent_id] = np.concatenate(
                    [base_obs, extra]
                ).astype(np.float32)

        return new_obs
    
    def _fake_infos(self):
        return {
            a: {
                "player_info": {
                    "position": [0.0, 0.0],
                    "velocity": [0.0, 0.0],
                },
                "ball_info": {
                    "position": [0.0, 0.0],
                },
            }
            for a in self.agents
        }

    def _modify_obs(self, obs_dict):
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
    """

class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)
        
        #if ray.is_initialized():
        #    ray.shutdown()
        #ray.init(
        #address=None,          # tried different ports all couldn't open
        #include_dashboard=False,
        #ignore_reinit_error=True
        #)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl") #KEY HERE .. is BEFORE curr dir of checkpoint
        
        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # No need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # NOTICE: watch.py in soccer-twos-env is WHAT ACTUALLY SENDS US the environment to use
        # the original baseline did NOT explain that, so misunderstanding here that the DummyEnv mattered
        #tune.registry.register_env("Soccer", lambda env_config: create_rllib_env(env_config))
        #Dummy env only for RLlib contstruction (do not use IN inference)
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        #tune.registry.register_env("DummyEnv", lambda config: DummyMultiAgentEnv(config))
        #config["env"] = "Soccer"
        config["env"] = "DummyEnv"

        # Create the Trainer from the config
        cls = get_trainable_cls(ALGORITHM)
        #agent = cls(env=config["env"], config=config) #before TEST
        #TEST
        agent = cls(config=config)

        # Load state from checkpoint
        agent.restore(CHECKPOINT_PATH)

        # Get policies for multi-agent setup
        # BEFORE TEST 3
        """self.policies = {
            "default": agent.get_policy("default"),
            "opponent_1": agent.get_policy("opponent_1"),
            "opponent_2": agent.get_policy("opponent_2"),
            "opponent_3": agent.get_policy("opponent_3"),
        }
        #TEST 3
        
        self.policies = {
            pid: agent.get_policy(pid)
            for pid in agent.policy_ids
        }        """

        self.policy = agent.get_policy(POLICY_NAME)
        print("MODEL INPUT EXPECTED:", self.policy.model.obs_space)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: A dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: A dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        actions = {}
        #TEST: based on watch.py in env soccer-twos-env act is sent agent1_actions = agent1.act({0: obs[0], 1: obs[1]})
        #Therefore we for actions have set for agent 0 agent1_actions[0] and for agent player 1 a...1_actions[1]
        #then that means
        #HERE we get observation for player 0 and player 1 ONLY therefore
        #we get out the player id and obs which at each time are p0 and p1 xor p2 and p3
        #HOWEVER, player_id here is therefore being set ONLY 0, 1 as player_id relative TO TEAM: line56,57 in watch.py
        #Furthermore: watch.py sends 336 OR 345 NOT 340/349 as we have set
        print("OBS SHAPES:", {k: v.shape for k, v in observation.items()})
        for player_id, obs in observation.items():

            #PLAN: reconstruct our shared teammate obs from scratch here
            #get teammate's state
            teammate_id = 1 - player_id
            teammate_obs = observation[teammate_id]

            #Extract teammate features if available
            if len(teammate_obs) >= 345: #>=345 since if 336 or 340 then teammate only has its raycasts and maybe curren player state not their state
                teammate_pos = teammate_obs[336:338]
                teammate_vel = teammate_obs[339:341]
            else: #if not already available then and only has raycasts by being 336 or 340(336+playerid current teammate)
                teammate_pos = np.zeros(2, dtype=np.float32)
                teammate_vel = np.zeros(2, dtype=np.float32)

            extra = np.concatenate([teammate_pos, teammate_vel])
            #augment the observation
            obs_aug = np.concatenate([obs, extra]).astype(np.float32)
            print("AUG SHAPE:", obs_aug.shape)

            #TEST ATTEMPT AUgment
            #obs = np.array(obs, dtype=np.float32)
            #if obs.shape[0] == 336:
            #    obs = np.concatenate([obs, np.zeros(4, dtype=np.float32)])

            # Choose the policy based on the agent ID
            """
            if player_id == 0:
                policy = self.policies["default"]
            elif player_id == 1:
                policy = self.policies["opponent_1"]
            elif player_id == 2:
                policy = self.policies["opponent_2"]
            elif player_id == 3:
                policy = self.policies["opponent_3"]
            else:
                policy = self.policies["default"]  # Default fallback
            """
            #policy = self.policies["default"]
            # Compute action using the selected policy
            #action, _ = policy.compute_single_action(obs)
            action, *_ = self.policy.compute_single_action(obs_aug)
            actions[player_id] = action
            
        return actions
