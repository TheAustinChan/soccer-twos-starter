import os
import logging

os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_METRICS_EXPORT_PORT"] = "-1"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_GRAFANA_HOST"] = "127.0.0.1"
os.environ["RAY_PROMETHEUS_HOST"] = "127.0.0.1"

import ray

# Reduce logging noise
logging.getLogger("ray").setLevel(logging.ERROR)

from ray import tune
from soccer_twos import EnvType

from utils_mod_simple_ppo_baseline import create_rllib_env


NUM_ENVS_PER_WORKER = 2
BASE_PORT           = 4234


if __name__ == "__main__":
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _node_ip_address="127.0.0.1",
        _metrics_export_port=-1   # ⭐ THIS is the key
    )

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_simple_baseline",
        config={
            # system settings
            "num_gpus": 0,
            "num_workers": 16,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
           
            "log_level": "INFO",
            "framework": "torch",
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "opponent_policy": lambda *_: 0,
                "base_port": BASE_PORT,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512],
            },
            "rollout_fragment_length": 500,
            "train_batch_size": 12000,
        },
        stop={
            "timesteps_total": 20000000,  # 15M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        #restore = "./ray_results/PPO_simple/PPO_Soccer_8cfe8_00000_0_2026-04-28_22-10-10/checkpoint_000600/checkpoint-600"
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
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
