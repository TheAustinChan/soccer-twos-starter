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
# Curriculum settings
# ---------------------------
MIN_ITERS_PER_TASK   = 30    # Don't advance too quickly even if reward spikes
MAX_ITERS_PER_TASK   = 200   # Don't get stuck forever on a task
REQUIRED_CONSECUTIVE = 3     # Consecutive evals above threshold before advancing

# ---------------------------
# Self-play settings (Phase 2 — after curriculum)
# Paper: only snapshot on genuine improvement, not a fixed timer.
# We use a reward-improvement gate instead of Nash Averaging (simpler for PPO).
# ---------------------------
SELFPLAY_WARMUP_ITERS    = 50    # Wait after curriculum ends before rotating
SELFPLAY_UPDATE_FREQ     = 20    # How often to check for opponent rotation
IMPROVEMENT_THRESHOLD    = 0.05  # Min reward improvement to trigger a snapshot

# ---------------------------
# Policy mapping
# ---------------------------
def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    Agent 0 always uses the training policy (default).
    All other agents sample from frozen opponent snapshots.

    During curriculum:   opponents are frozen copies of the previous stage's
                         best policy — they don't improve.
    During self-play:    opponents are rotating snapshots of 'default',
                         providing an ever-improving training partner.

    40% chance of playing against the current default so the agent also
    learns to coordinate with a skilled teammate.
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

        # --- Curriculum state ---
        self.current_task        = 0
        self.task_start_iter     = 0
        self.consecutive_above   = 0

        # --- Self-play state ---
        # Per the paper: self-play only starts after all curriculum stages done.
        # Opponents are frozen until then (initialized to random weights = no-ops).
        self.curriculum_done     = False
        self.selfplay_start_iter = None

        # Improvement-gated snapshot (proxy for Nash Averaging in the paper)
        self.best_snapshot_reward = -np.inf

    # ------------------------------------------------------------------
    # Episode start — configure env for current curriculum task
    # Runs on WORKERS. self.current_task is synced via foreach_worker.
    # ------------------------------------------------------------------
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        for env in base_env.get_unwrapped():
            task_id       = self.current_task
            config_fn_key = tasks[task_id]["config_fn"]

            fn = config_fns.get(config_fn_key)
            if fn is None:
                raise ValueError(
                    f"Unknown config_fn '{config_fn_key}' "
                    f"in task '{tasks[task_id]['name']}'"
                )
            fn(env)

            # Set Unity side-channel parameters (ball + player spawn ranges)
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
    # Phase 1: Curriculum advancement
    # Runs on DRIVER. Advances task based on reward threshold.
    #
    # Per the paper: each curriculum stage is trained until the agent
    # reaches a performance criterion, then the best policy is frozen
    # and used as the opponent for the next stage.
    # ------------------------------------------------------------------
    def _try_advance_curriculum(self, trainer, iteration, mean_reward):
        if self.curriculum_done:
            return

        iters_on_task = iteration - self.task_start_iter
        threshold     = tasks[self.current_task].get("advance_threshold", 0.65)

        # Track consecutive successes; reset streak on failure
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
            reason = "reward" if reward_ready else "timeout"

            # --- Freeze current best policy into all opponent slots ---
            # Per the paper: the best agent from stage k becomes the fixed
            # opponent for stage k+1. We freeze into all three slots so
            # the new task starts against a competent fixed opponent
            # rather than random weights.
            print(
                f"[Curriculum] Freezing current policy as opponent "
                f"before advancing (reason={reason}, reward={mean_reward:.3f})"
            )
            weights = trainer.get_weights()
            trainer.set_weights({
                "opponent_1": weights["default"],
                "opponent_2": weights["default"],
                "opponent_3": weights["default"],
            })

            # Advance task
            self.current_task     += 1
            self.task_start_iter   = iteration
            self.consecutive_above = 0
            self.best_snapshot_reward = mean_reward  # reset improvement baseline

            print(
                f"[Curriculum] -> Task {self.current_task}: "
                f"'{tasks[self.current_task]['name']}'"
            )

            # Sync new task index to all remote workers
            current = self.current_task
            trainer.workers.foreach_worker(
                lambda w: w.foreach_env(lambda env: env.set_task(current))
            )

        else:
            # All curriculum stages complete — hand off to self-play
            if not self.curriculum_done:
                self.curriculum_done     = True
                self.selfplay_start_iter = iteration

                # Freeze the final curriculum policy as the starting opponent
                weights = trainer.get_weights()
                trainer.set_weights({
                    "opponent_1": weights["default"],
                    "opponent_2": weights["default"],
                    "opponent_3": weights["default"],
                })

                print(
                    f"[Curriculum] Complete at iteration {iteration}. "
                    f"Self-play warmup started ({SELFPLAY_WARMUP_ITERS} iters)."
                )

    # ------------------------------------------------------------------
    # Phase 2: Self-play opponent rotation
    # Runs on DRIVER. Only activates after curriculum is done + warmup.
    #
    # Per the paper: rather than Nash Averaging (requires many evaluations),
    # we use a simpler improvement gate — only snapshot when the default
    # policy has genuinely improved over the last snapshot, preventing
    # the agent from practicing against a policy it has already surpassed.
    # ------------------------------------------------------------------
    def _try_update_opponents(self, trainer, iteration, mean_reward):
        if not self.curriculum_done:
            return  # Still in curriculum phase

        if self.selfplay_start_iter is None:
            return

        iters_since_selfplay = iteration - self.selfplay_start_iter
        if iters_since_selfplay < SELFPLAY_WARMUP_ITERS:
            return  # Still warming up

        if iteration % SELFPLAY_UPDATE_FREQ != 0:
            return

        # Only rotate if the default policy has meaningfully improved
        # (proxy for Nash Averaging — avoids snapshotting a policy that
        #  hasn't improved, which would give the agent an easy target)
        if mean_reward > self.best_snapshot_reward + IMPROVEMENT_THRESHOLD:
            self.best_snapshot_reward = mean_reward
            print(
                f"[Self-play] Improvement detected (reward={mean_reward:.3f}). "
                f"Rotating opponent snapshots at iteration {iteration}."
            )
            weights = trainer.get_weights()
            trainer.set_weights({
                "opponent_3": weights["opponent_2"],
                "opponent_2": weights["opponent_1"],
                "opponent_1": weights["default"],
            })
        else:
            print(
                f"[Self-play] No improvement (reward={mean_reward:.3f} vs "
                f"best={self.best_snapshot_reward:.3f}). Keeping current opponents."
            )

    # ------------------------------------------------------------------
    # Main training hook (driver side)
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
        self._try_update_opponents(trainer, iteration, mean_reward)


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
        name="PPO_selfplay_paper",
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
                # Per the paper: only the default (training) policy gets gradient updates.
                # Opponent policies are frozen snapshots and never trained.
                "policies_to_train": ["default"],
            },

            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "base_port": BASE_PORT,
            },

            # Model: shared value function head, two hidden layers.
            # Paper uses separate actor/critic (TD3), but for PPO
            # shared layers with vf_share_layers=True is standard.
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
        checkpoint_freq=50,
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