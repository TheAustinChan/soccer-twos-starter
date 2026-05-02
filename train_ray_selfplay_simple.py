import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils_mod_simple_selfplay import create_rllib_env

NUM_ENVS_PER_WORKER = 2
BASE_PORT           = 4332

# ---------------------------
# Self-play settings
# ---------------------------
SELFPLAY_WARMUP_ITERS        = 20  # no snapshots before this iteration
SELFPLAY_UPDATE_FREQ         = 10   # check for snapshot every N iters
IMPROVEMENT_THRESHOLD        = 0.02 # min reward improvement to snapshot
MAX_ITERS_WITHOUT_SNAPSHOT   = 100  # force snapshot if stuck this long

# ---------------------------
# Policy mapping
# ---------------------------
def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Agent 0 always uses the training policy (default).
    Agents 1, 2, 3 sample from frozen opponent snapshots.
    40% chance of current default so agent trains against
    its current self AND historical snapshots.
    """
    if agent_id == 0:
        return "default"
    return np.random.choice(
        ["default", "opponent_1", "opponent_2", "opponent_3"],
        p=[0.4, 0.2, 0.2, 0.2],
    )


# ---------------------------
# Self-play Callback
# Uses older RLLib **info signature — matches soccer_twos example
# ---------------------------
class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.best_snapshot_reward  = -np.inf
        self.last_snapshot_iter    = 0
        print("==== SelfPlayCallback initialized ====")

    def on_train_result(self, **info):
        result    = info["result"]
        trainer   = info["trainer"]

        iteration   = result["training_iteration"]
        mean_reward = result.get("episode_reward_mean", -np.inf)

        print(
            f"[Iter {iteration}] reward={mean_reward:.3f} | "
            f"best_snapshot={self.best_snapshot_reward:.3f} | "
            f"iters_since_snapshot={iteration - self.last_snapshot_iter}"
        )

        # Hard warmup gate — no snapshots while policy is still random
        if iteration < SELFPLAY_WARMUP_ITERS:
            print(f"[Self-play] Warming up ({iteration}/{SELFPLAY_WARMUP_ITERS})")
            return

        # Only check every SELFPLAY_UPDATE_FREQ iterations
        if iteration % SELFPLAY_UPDATE_FREQ != 0:
            return

        iters_since_snapshot = iteration - self.last_snapshot_iter
        reward_improved      = mean_reward > self.best_snapshot_reward + IMPROVEMENT_THRESHOLD
        stuck_too_long       = iters_since_snapshot >= MAX_ITERS_WITHOUT_SNAPSHOT

        if reward_improved or stuck_too_long:
            reason = "improvement" if reward_improved else "forced (stuck)"
            self.best_snapshot_reward = mean_reward
            self.last_snapshot_iter   = iteration

            print(
                f"[Self-play] Rotating snapshots — "
                f"reason={reason}, reward={mean_reward:.3f}"
            )
            trainer.set_weights({
                "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
            })
        else:
            print(
                f"[Self-play] No improvement — keeping current opponents "
                f"(reward={mean_reward:.3f} vs best={self.best_snapshot_reward:.3f})"
            )


# ---------------------------
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
        name="PPO_selfplay_simple",
        config={
            # System
            "num_gpus": 0,
            "num_workers": 16,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayCallback,

            # Multi-agent
            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },

            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
            },

            # Model
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },

            "rollout_fragment_length": 1000,
            "train_batch_size": 32000,  # 16 workers × 2 envs × 5000
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 30_000_000,
            "time_total_s": 84_000,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay/checkpoint_000600/checkpoint-600",
        #restore="../ray_results/PPO_simple/PPO_Soccer_51251_00000_0_2026-04-28_17-29-20/checkpoint_000500/checkpoint-500"
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")