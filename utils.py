from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """
    #we are first passed to us a wrapper to use from the caller training script like in the example scripts in gym.core.Wrapper

    #then passed a MultiAgentEnv object that is called by create_rllib_env at the end

    """
    MultiAgentEnv object is an env that hosts many independent agents id'd by agentId strs.
    simply allows us to operate with many agents, returning info based on dict passed?

    Convert gym,env to MultiAgentEnv from single agent by stacking n instances of given gym.Env class into MultiAgEnv and returns new class
    where each live in n parallel single agent envs.

    We are passed the multiagent env passed to us from the env from soccer_twos.make(**env_config) whose config is set a a doubl ptr of
    hasattr branch were for each worker we have set the global iindex among groups for that worker. env_config given to us by training script
    in the main where EnvType.multiagent_team passed to variation sets the variation to multiagent independent once again.
    

    To test simple rewards, inherit as much possible exceot the basic reward to test: penalize hoarding the ball while an enemy is closeby
    so if a raycast i detects opponent close to self it will pass the ball, or the reward generally will maximize over the distance from the opponent to the ball
    so need to include dims for the ball and for the opponent position (consider simple cases just positions not velocities), later can infer based on difference in
    estimated pos and true pos of enemy. So anyways, the reward should be less in value and bounded by the current value of scoring the goal so that that goal takes
    precedence so we don't just end up only passing the ball to another teammate.

    All agents share the same policy for the ray ma players example unlike in teams

    To get the extra info. In line 294 of single step func we set the obs for each. Warning though can mess with the observation space hardcoded to 336
    so we can implement single step here too

    For things we don't care about we might be able to just call super to extend the original class.

    Edit in the wrappers.step func the final reward returned, return higher return as additional,

    NOTICE: for this implementation we do not alter the observation space in any way reliant on the default 336 state and ray cast interceptions with object types
    which are sparse so hard to learn, or the simulator state which we in line 294 exist if 345.
    MAY NOT need to force to 345

    For more info on the environment itself, visit how the example_code.ipynb calls .make on soccer_twos presumably instantiated
    by the init.py in soccer-twos-env/soccer-twos/__init__.py


    Reward Scheme: defined following in a reward heirarchy of pressure as in about to be blocked off or lose ball, then we have general reward policies around hoarding
                #to occasionally sometimes pass/shoot even if not highly threatened at a given moment

    Potential Pitfalls: opponents maybe positioned between two teammates when a pass happens or directly in the way of the ball to the target goal even if relatively far 
    and can then intercept
    Possible reward conflict between closing in to take a ball in possession of an enemy when it may be critical to do so and the reward of passing/shooting
    Could also explore adding reward based on non linear functions, perhaps as some sinusoidal model of what a game that looks bad looks like

    Approaches: start testing the more important to strategy without any glaring fault reward schemes first as part of one coherent strategy, then add fine tuning
    rewards for noise, etc. Just having a balanced 2 rewards is alright to test first, and the positive reward towards shooting and scoring is fine to be say half or a 1/4th
    the weight of the reward of the penalty since we have an overarching default scoring reward
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        return self.env.reset()
    
    #now customly define new reward weights to add to the scalar reward assigned to the player that calls later
def step(self, action_dict):
    obs, rewards, dones, infos = self.env.step(action_dict)

    shaped_rewards = {}

    for agent_id, reward in rewards.items():
        shaped_reward = reward #original reward baseline

        if agent_id in infos: #agent info available this step
            info = infos[agent_id]

            if "ball_info" in info and "player_info" in info: #state exists (345-based info unpacked)
                ball_pos = info["ball_info"]["position"] #ball state from simulator
                player_pos = info["player_info"]["position"] #agent state from simulator

                #euclidean distance player to ball
                dist_to_ball = ((ball_pos[0] - player_pos[0]) ** 2 + (ball_pos[1] - player_pos[1]) ** 2) ** 0.5

                #possession as smooth function (closer = higher control), encourage closing in on the ball to take it
                possession = 1.0 / (1.0 + dist_to_ball)

                #ray pressure proxy (no explicit opponent state available since we didn't alter obs)
                pressure = 0.0

                if agent_id in obs: #use observation rays if available
                    agent_obs = obs[agent_id]

                    if isinstance(agent_obs, tuple): #flatten multi-input obs
                        flat_obs = np.concatenate([np.array(x).flatten() for x in agent_obs])
                    else:
                        flat_obs = np.array(agent_obs).flatten()

                    ray_signal = flat_obs[:200] #forward ray-heavy region
                    pressure = np.mean(ray_signal > 0.5) #density heuristic

                #defined following in a reward heirarchy of pressure as in about to be blocked off or lose ball, then we have general reward policies around hoarding
                #to occasionally sometimes pass/shoot even if not highly threatened at a given moment
                #possession shaping reward (encourage ball control without hard threshold)
                #The following are okay to define possession as 1/4th the weight of pressure to release since we have the overarching default reward to score anyways
                shaped_reward += 0.015 * possession

                #penalize holding ball under pressure (discourage hoarding when we don't have a clear shot to score since enemy blocking orr about to intercept)
                shaped_reward -= 0.06 * possession * pressure

                #Test these if the above two are somewhat ok, these below are lower priority
                #reward releasing pressure (implicit passing behavior)
                #if pressure > 0.3 and possession < 0.3:
                #    shaped_reward += 0.03 * (1.0 - pressure) * possession

                #prevent hoarding in general if possession way too high
                #if possession > 0.85:
                #    shaped_reward -= 0.005

        shaped_rewards[agent_id] = shaped_reward

    return obs, shaped_rewards, dones, infos

    #superclass wrapper should allow us to inherit the other funcs in MultiAgentUnityWrapper


    #pass


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
