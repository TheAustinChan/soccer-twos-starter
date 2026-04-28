# Soccer-Twos RL Project: System Architecture & Logic Summary

## 1. Observation Space & Feature Engineering
- **Base Observation (336):** Raw Unity raycasts providing environmental awareness.
- **Augmented Features (6):** Added to the end of the observation vector to provide explicit spatial reasoning.
    - [336, 337]: Relative Ball Position (Ball_Pos - Player_Pos)
    - [338, 339]: Relative Opponent Goal (Goal_Pos - Player_Pos)
    - [340, 341]: Relative Teammate (Teammate_Pos - Player_Pos)
- **Total Input Size:** 342 dimensions.

## 2. Reward Shaping Strategy (utils_mod.py)
- **Goal Reward:** Scaled to 10.0 to ensure the "Win Condition" is the primary driver.
- **Distance Rewards:** - Constant pressure to minimize distance to the ball.
    - Ball-to-Goal progress reward to encourage offensive movement.
- **Symmetry Handling:** Goal coordinates are dynamically assigned based on Agent ID.
    - Team 0 (IDs 0, 1): Target Goal = +16.0x
    - Team 1 (IDs 2, 3): Target Goal = -16.0x

## 3. Training & Curriculum Logic (train_ray_selfplay_new.py)
- **Phase 1: Curriculum Learning**
    - Progression through tasks (ball approach -> scoring -> defense).
    - Advancement: Requires 3 consecutive iterations above the reward threshold.
    - Constraints: Min 30 iters / Max 200 iters per task to prevent premature advancement or stagnation.
- **Phase 2: Self-Play Snapshot Rotation**
    - After curriculum ends: 50-iteration warmup.
    - Every 20 iterations: Snapshots rotate (Opponent_3 <- Opponent_2 <- Opponent_1 <- Current_Default).
    - Teammate Sampling: 40% chance to play with the learning "Default" policy to maintain coordination skills.

## 4. Execution & Compatibility Fixes
- **RLLib Env Factory:** - Switched from `hasattr` to dictionary key checking for `worker_index`.
    - Added "config scrubbing" to prevent Unity from crashing on Ray-specific metadata.
- **Symmetry Correction (mod_agent.py):**
    - Fixed the `watch.py` ID-mapping conflict.
    - The agent now detects if it is global ID 2 or 3 and shifts local IDs (0, 1) accordingly.
    - This prevents the "Self-Goal" bug where agents attack the wrong side of the pitch during evaluation.

## 5. Technical Stack
- **OS:** Ubuntu Linux (Dual-boot config).
- **Framework:** Ray/RLLib with PyTorch.
- **Drivers:** Fixes applied for `sof-hda-dsp` audio compatibility on HP Envy hardware.

Paper idea implementation that we modified from: https://2019.robocup.org/downloads/program/OcanaEtAl2019.pdf