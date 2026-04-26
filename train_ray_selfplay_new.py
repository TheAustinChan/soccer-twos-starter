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
with open("curriculum_new.yaml") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)

tasks = curriculum["tasks"]

config_fns = {
    "none": lambda *_: None,
    "random_players": lambda env: env.set_policies(
        lambda *_: env.action_space.sample()
    ),
}

# ---------------------------
# Curriculum settings
# ---------------------------
MIN_ITERS_PER_TASK   = 30    # Don't advance too quickly even if reward spikes early
MAX_ITERS_PER_TASK   = 200   # Don't get stuck forever on a task
REQUIRED_CONSECUTIVE = 3     # Consecutive evals above threshold before advancing

# ---------------------------
# Self-play settings
# ---------------------------
SELFPLAY_WARMUP_ITERS = 50   # Wait this many iters after curriculum ends before rotating
SELFPLAY_UPDATE_FREQ  = 20   # How often to snapshot opponents once warmed up

# ---------------------------
# Policy mapping
# ---------------------------
def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Agent 0 always uses the training policy.
    All other agents (teammates + opponents) sample from frozen snapshots,
    with a 40% chance of being the current default policy so the agent
    also learns to coordinate with a skilled teammate.
    """
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
        # Curriculum state
        self.current_task        = 0
        self.task_start_iter     = 0
        self.consecutive_above   = 0

        # Self-play state (None until curriculum finishes)
        self.curriculum_done     = False
        self.selfplay_start_iter = None

    # ------------------------------------------------------------------
    # Episode start — configure env for current task
    # NOTE: on_episode_start runs on workers. self.current_task is synced
    #       to workers via foreach_worker inside on_train_result.
    # ------------------------------------------------------------------
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        for env in base_env.get_unwrapped():
            task_id = self.current_task

            config_fn_key = tasks[task_id]["config_fn"]
            fn = config_fns.get(config_fn_key)
            if fn is None:
                raise ValueError(
                    f"Unknown config_fn '{config_fn_key}' "
                    f"in task '{tasks[task_id]['name']}'"
                )
            fn(env)

            if hasattr(env, "env_channel"):
                env.env_channel.set_parameters(
                    ball_state=sample_pos_vel(tasks[task_id]["ranges"]["ball"]),
                    players_states={
                        player: sample_player(
                            tasks[task_id]["ranges"]["players"][player]
                        )
                        for player in tasks[task_id]["ranges"]["players"]
                    },
                )

    # ------------------------------------------------------------------
    # Phase 1: Curriculum advancement (runs on driver)
    # ------------------------------------------------------------------
    def _try_advance_curriculum(self, trainer, iteration, mean_reward):
        if self.curriculum_done:
            return

        iters_on_task = iteration - self.task_start_iter
        threshold     = tasks[self.current_task].get("advance_threshold", 0.65)

        # Track consecutive successes; reset streak on any failure
        if mean_reward >= threshold:
            self.consecutive_above += 1
        else:
            self.consecutive_above = 0

        reward_ready = (
            self.consecutive_above >= REQUIRED_CONSECUTIVE
            and iters_on_task >= MIN_ITERS_PER_TASK
        )
        timed_out = iters_on_task >= MAX_ITERS_PER_TASK

        if not (reward_ready or timed_out):
            return

        if self.current_task < len(tasks) - 1:
            reason                 = "reward" if reward_ready else "timeout"
            self.current_task     += 1
            self.task_start_iter   = iteration
            self.consecutive_above = 0

            print(
                f"[Curriculum] Task {self.current_task}: "
                f"'{tasks[self.current_task]['name']}' "
                f"(reason={reason}, reward={mean_reward:.3f})"
            )

            # Push new task index to all remote workers
            current = self.current_task
            trainer.workers.foreach_worker(
                lambda w: w.foreach_env(lambda env: env.set_task(current))
            )
        else:
            # All tasks done — hand off to self-play phase
            if not self.curriculum_done:
                self.curriculum_done     = True
                self.selfplay_start_iter = iteration
                print(
                    f"[Curriculum] Complete at iteration {iteration}. "
                    f"Self-play warmup started ({SELFPLAY_WARMUP_ITERS} iters)."
                )

    # ------------------------------------------------------------------
    # Phase 2: Self-play opponent rotation (runs on driver)
    # Only activates after curriculum completes + warmup period
    # ------------------------------------------------------------------
    def _try_update_opponents(self, trainer, iteration):
        if not self.curriculum_done:
            return  # Still in curriculum phase

        if self.selfplay_start_iter is None:
            return

        iters_since_selfplay = iteration - self.selfplay_start_iter
        if iters_since_selfplay < SELFPLAY_WARMUP_ITERS:
            return  # Still warming up — let default policy build a baseline

        if iteration % SELFPLAY_UPDATE_FREQ == 0:
            print(f"[Self-play] Rotating snapshots at iteration {iteration}")
            weights = trainer.get_weights()
            trainer.set_weights({
                "opponent_3": weights["opponent_2"],
                "opponent_2": weights["opponent_1"],
                "opponent_1": weights["default"],
            })

    # ------------------------------------------------------------------
    # Main training hook
    # ------------------------------------------------------------------
    def on_train_result(self, *, trainer, result, **kwargs):
        iteration   = result["training_iteration"]
        mean_reward = result.get("episode_reward_mean", -np.inf)

        phase = "self-play" if self.curriculum_done else "curriculum"
        print(
            f"[Iter {iteration}] phase={phase} | "
            f"task={self.current_task} '{tasks[self.current_task]['name']}' | "
            f"reward={mean_reward:.3f}"
        )

        self._try_advance_curriculum(trainer, iteration, mean_reward)
        self._try_update_opponents(trainer, iteration)


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
        name="PPO_selfplay_full",
        config={
            # System settings
            "num_gpus": 0,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CombinedCallback,
            # RL setup
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
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 15_000_000,
            "time_total_s": 42_000,  # ~12 hours
        },
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_full/checkpoint_000600/checkpoint-600",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")