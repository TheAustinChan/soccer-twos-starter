import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import yaml
from utils_mod import create_rllib_env, sample_pos_vel, sample_player

NUM_ENVS_PER_WORKER = 2
BASE_PORT = 4549
# ---------------------------
# Load curriculum
# ---------------------------
with open("curriculum.yaml") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)

tasks = curriculum["tasks"]

config_fns = {
    "none": lambda *_: None,
    "random_players": lambda env: env.set_policies(
        lambda *_: env.action_space.sample()
    ),
}

# ---------------------------
# Policy mapping
# ---------------------------
def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"
    else:
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            p=[0.4, 0.2, 0.2, 0.2],
        )

# ---------------------------
# Combined Callback
# ---------------------------
class CombinedCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.current_task = 0

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        for env in base_env.get_unwrapped():
            task_id = self.current_task
            config_fns[tasks[task_id]["config_fn"]](env)
            
            # Set Unity parameters via the side channel
            if hasattr(env, "env_channel"):
                env.env_channel.set_parameters(
                    ball_state=sample_pos_vel(tasks[task_id]["ranges"]["ball"]),
                    players_states={
                        player: sample_player(tasks[task_id]["ranges"]["players"][player])
                        for player in tasks[task_id]["ranges"]["players"]
                    },
                )

    def on_train_result(self, *, trainer, result, **kwargs):
        iteration = result["training_iteration"]

        if iteration % 50 == 0:
            if self.current_task < len(tasks) - 1:
                self.current_task += 1
                print(f"---- Curriculum -> Task {self.current_task}: {tasks[self.current_task]['name']} ----")

        if iteration > 250 and iteration % 20 == 0:
            print("---- Updating opponents ----")
            weights = trainer.get_weights()
            trainer.set_weights({
                "opponent_3": weights["opponent_2"],
                "opponent_2": weights["opponent_1"],
                "opponent_1": weights["default"],
            })

# --# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_full",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CombinedCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER,
                            "base_port": BASE_PORT,},
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 15000000, "time_total_s": 42000,},  # 2h
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_twos_2/PPO_Soccer_a8b44_00000_0_2021-09-18_11-13-55/checkpoint_000600/checkpoint-600",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
