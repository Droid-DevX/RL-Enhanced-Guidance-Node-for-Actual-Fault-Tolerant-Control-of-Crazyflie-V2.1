"""
train_nominal.py — SAC Training Script for Goal-Conditioned Hover
=================================================================
Trains a SAC agent (Stable Baselines3) to hover a Crazyflie v2 at any
target height z ∈ [0.2, 2.5] m.

SAC advantages over PPO for hardware deployment:
  • Off-policy → more sample-efficient, reuses past experience
  • Entropy-regularised → robust exploration without manual tuning
  • Deterministic policy extraction → clean, jitter-free actions for
    real hardware (no stochastic noise at deployment time)
  • Single-env training → simpler, lower memory footprint

Usage:
    python3 train_nominal.py                     # default 3M steps
    python3 train_nominal.py --steps 1000000     # custom step count
    python3 train_nominal.py --gui               # visual debug

Features:
  • EvalCallback — saves the best model by mean reward
  • CheckpointCallback — periodic checkpoints every 50k steps
  • TensorBoard logging — reward, hover error, episode length
  • Custom callback to log hover_error to TensorBoard

Author : Ayush
Date   : 2026-06-02
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Make sure we can import the custom environment ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_hover_env import GoalConditionedHoverEnv

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_500_000       # Total training timesteps
EVAL_FREQ       = 10_000          # Evaluate every N env steps
EVAL_EPISODES   = 10              # Episodes per evaluation round
CKPT_FREQ       = 50_000          # Save checkpoint every N steps
PRINT_FREQ      = 5_000           # Console progress print frequency


# SAC hyper-parameters (tuned for continuous drone control)
SAC_HYPERPARAMS = dict(
    policy            = "MlpPolicy",
    learning_rate     = 3e-4,
    buffer_size       = 1_000_000,     # Replay buffer capacity
    learning_starts   = 10_000,        # Random exploration steps before learning
    batch_size        = 256,           # Mini-batch size from replay buffer
    tau               = 0.005,         # Soft target update coefficient
    gamma             = 0.99,          # Discount factor
    train_freq        = 1,             # Update policy every step
    gradient_steps    = 1,             # Gradient updates per env step
    ent_coef          = "auto",        # Auto-tune entropy (key SAC feature)
    target_entropy    = "auto",        # Auto-target entropy
    use_sde           = False,         # gSDE exploration (False = default Gaussian)
    policy_kwargs     = dict(
        net_arch=[256, 256],           # Shared arch for actor & critic networks
    ),
    verbose           = 0,
    device            = "auto",
)


# ──────────────────────────────────────────────────────────────────────────────
# Custom TensorBoard Callback — logs hover error & episode stats
# ──────────────────────────────────────────────────────────────────────────────
class HoverMetricsCallback(BaseCallback):
    """
    Logs additional metrics to TensorBoard:
      • mean_hover_error  (from info dicts)
      • mean_episode_reward
      • mean_episode_length
    Also prints a progress line to stdout every `print_freq` steps.
    """

    def __init__(self, print_freq: int = PRINT_FREQ, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq  = print_freq
        self._last_print = 0
        self._t0         = None

        # Rolling buffers for smoothed metrics
        self._hover_errors = deque(maxlen=100)
        self._ep_rewards   = deque(maxlen=100)
        self._ep_lengths   = deque(maxlen=100)

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self) -> bool:
        # Collect info from all parallel envs
        for info in self.locals.get("infos", []):
            # Per-step hover error
            if "hover_error" in info:
                self._hover_errors.append(info["hover_error"])

            # Episode-level stats (added by Monitor wrapper)
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])

        # ── TensorBoard logging ──
        if self.num_timesteps % 1000 == 0:
            if self._hover_errors:
                self.logger.record("custom/mean_hover_error",
                                   float(np.mean(self._hover_errors)))
            if self._ep_rewards:
                self.logger.record("custom/mean_ep_reward",
                                   float(np.mean(self._ep_rewards)))
            if self._ep_lengths:
                self.logger.record("custom/mean_ep_length",
                                   float(np.mean(self._ep_lengths)))

        # ── Console print ──
        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            elapsed = time.time() - self._t0 if self._t0 else 1e-9
            fps = self.num_timesteps / max(elapsed, 1e-9)
            mean_r = (float(np.mean(self._ep_rewards))
                      if self._ep_rewards else float("nan"))
            mean_l = (float(np.mean(self._ep_lengths))
                      if self._ep_lengths else float("nan"))
            mean_he = (float(np.mean(self._hover_errors))
                       if self._hover_errors else float("nan"))
            print(
                f"  step {self.num_timesteps:>10,} | "
                f"reward={mean_r:+8.2f} | "
                f"ep_len={mean_l:7.1f} | "
                f"hover_err={mean_he:.4f} | "
                f"fps={fps:6.0f}"
            )

        return True  # Continue training


# ──────────────────────────────────────────────────────────────────────────────
# Environment factory functions
# ──────────────────────────────────────────────────────────────────────────────
def make_train_env(gui: bool = False):
    """Return a function that creates a Monitor-wrapped training env."""
    def _init():
        env = GoalConditionedHoverEnv(gui=gui)
        return Monitor(env)
    return _init


def make_eval_env():
    """Return a function that creates a Monitor-wrapped evaluation env."""
    def _init():
        env = GoalConditionedHoverEnv(gui=False)
        return Monitor(env)
    return _init


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────
def train(total_steps: int, log_dir: str, gui: bool = False):
    """
    Set up environments, callbacks, and run SAC training.

    Parameters
    ----------
    total_steps : int  – Total environment steps to train for.
    log_dir     : str  – Root directory for logs, checkpoints, and best model.
    gui         : bool – If True, run with the PyBullet GUI.
    """
    # ── Directory structure ──
    best_dir = os.path.join(log_dir, "best")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    tb_dir   = os.path.join(log_dir, "tensorboard")
    eval_log = os.path.join(log_dir, "eval_logs")
    for d in (best_dir, ckpt_dir, tb_dir, eval_log):
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Goal-Conditioned Hover — SAC Training")
    print(f"  Steps      : {total_steps:,}")
    print(f"  Log dir    : {log_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"{'='*70}\n")

    # ── Create training environment ──
    # SAC is off-policy → single env is standard (no SubprocVecEnv needed)
    train_vec = DummyVecEnv([make_train_env(gui=gui)])

    # ── Create evaluation environment ──
    eval_vec = DummyVecEnv([make_eval_env()])

    # ── Instantiate SAC model ──
    model = SAC(
        env=train_vec,
        tensorboard_log=tb_dir,
        **SAC_HYPERPARAMS,
    )

    print(f"  Model policy network: {model.policy}")
    print(f"  Action space:         {train_vec.action_space}")
    print(f"  Observation space:    {train_vec.observation_space}")
    print(f"  Replay buffer size:   {SAC_HYPERPARAMS['buffer_size']:,}")
    print(f"  Learning starts at:   {SAC_HYPERPARAMS['learning_starts']:,} steps")
    print()

    # ── Callbacks ──
    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=best_dir,
        log_path=eval_log,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=CKPT_FREQ,
        save_path=ckpt_dir,
        name_prefix="sac_hover",
        verbose=0,
    )

    metrics_cb = HoverMetricsCallback(print_freq=PRINT_FREQ)

    # ── Train! ──
    t0 = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([eval_cb, ckpt_cb, metrics_cb]),
        reset_num_timesteps=True,
        progress_bar=False,
    )
    elapsed = time.time() - t0

    # ── Save final model ──
    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)

    print(f"\n{'='*70}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best model  → {best_dir}/best_model.zip")
    print(f"  Final model → {final_path}.zip")
    print(f"  TensorBoard → tensorboard --logdir {tb_dir}")
    print(f"{'='*70}\n")

    # ── Cleanup ──
    train_vec.close()
    eval_vec.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train SAC for goal-conditioned Crazyflie hover",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=TOTAL_TIMESTEPS,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs_hover_sac",
        help="Directory for logs, checkpoints, TensorBoard",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Run with PyBullet GUI (slower, for debugging)",
    )
    args = parser.parse_args()

    # Resolve log dir relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, args.log_dir)

    train(
        total_steps=args.steps,
        log_dir=log_dir,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()
