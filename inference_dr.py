"""
inference_dr.py — Deployment / Testing Script for Domain Randomized Hover
=============================================================================
Loads a trained Domain Randomized SAC model, takes a target height from
the user via the terminal, and runs the Crazyflie v2 drone in PyBullet
to hover at that height.

To test fault-tolerance, the user/researcher hardcodes `self._thrust_factor`
in `custom_hover_env_dr.py`'s `_preprocessAction()` method (e.g. to 0.70
for 30% motor degradation).

Usage:
    python3 inference_dr.py                          # uses best model
    python3 inference_dr.py --model path/to/model    # custom model path
    python3 inference_dr.py --gui                    # with PyBullet GUI
    python3 inference_dr.py --episodes 5             # run 5 episodes

Author : Ayush
Date   : 2026-06-06
"""

import argparse
import os
import sys
import time

import math
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

# ── Import the DR custom environment ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_hover_env_dr import GoalConditionedHoverEnvDR, Z_TARGET_MIN, Z_TARGET_MAX


# ──────────────────────────────────────────────────────────────────────────────
# Helper: get z_target from user
# ──────────────────────────────────────────────────────────────────────────────
def get_target_height_from_user() -> float:
    """Prompt the user for a target height and validate it."""
    while True:
        try:
            val = float(input(
                f"\n  Enter target hover height "
                f"[{Z_TARGET_MIN:.1f} – {Z_TARGET_MAX:.1f} m]: "
            ))
            if Z_TARGET_MIN <= val <= Z_TARGET_MAX:
                return val
            else:
                print(f"  ⚠  Height must be between {Z_TARGET_MIN} and "
                      f"{Z_TARGET_MAX} m.  Try again.")
        except ValueError:
            print("  ⚠  Invalid input.  Please enter a number.")
        except KeyboardInterrupt:
            print("\n  Exiting.")
            sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model_path: str, gui: bool = True,
                  num_episodes: int = 1, slow_mo: bool = True):
    """
    Load the trained DR model and run inference episodes.

    Parameters
    ----------
    model_path   : str  – Path to the saved SAC .zip file (without .zip).
    gui          : bool – Open PyBullet GUI for visualisation.
    num_episodes : int  – Number of episodes to run.
    slow_mo      : bool – Slow down rendering to ~real-time if GUI is on.
    """
    # ── Load model ──
    print(f"\n  Loading DR model from: {model_path}")
    model = SAC.load(model_path)
    print("  ✓ DR Model loaded successfully.\n")

    # ── Create environment ──
    env = GoalConditionedHoverEnvDR(gui=gui)

    for ep in range(num_episodes):
        # ── Get target from user ──
        z_target = get_target_height_from_user()

        print(f"\n{'─'*60}")
        print(f"  Episode {ep+1}/{num_episodes}")
        print(f"  Target height: {z_target:.2f} m")
        print(f"{'─'*60}")
        print(f"  {'Step':>6}  {'x,y (m)':>12}  {'z (m)':>8}  {'z_target':>8}  "
              f"{'error (m)':>10}  {'reward':>8}")
        print(f"  {'─'*65}")

        # ── Reset and override z_target ──
        obs, info = env.reset()
        env._z_target = z_target  # Override the randomised target
        # Recompute obs so z_target in observation matches
        obs = env._computeObs()

        total_reward = 0.0
        step = 0
        done = False

        rpms_history = []
        z_error_history = []

        while not done:
            # ── Get deterministic action from policy ──
            action, _ = model.predict(obs, deterministic=True)

            # ── Step the environment ──
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            # ── Record data for plotting ──
            # Action is [0, 1]. Multiply by MAX_RPM to get commanded RPM
            rpms_history.append(action * env.MAX_RPM)
            z_cur_val = info.get("z", obs[2])
            z_error_history.append(z_cur_val - z_target)

            # ── Print hover error every 10 steps ──
            if step % 10 == 0 or done:
                x_cur = info.get("x", obs[0])
                y_cur = info.get("y", obs[1])
                z_cur = info.get("z", obs[2])
                h_err = info.get("hover_error", math.sqrt(x_cur**2 + y_cur**2 + (z_cur - z_target)**2))
                print(
                    f"  {step:>6}  {x_cur:>5.2f},{y_cur:>5.2f}  {z_cur:>8.4f}  {z_target:>8.4f}  "
                    f"{h_err:>10.4f}  {reward:>8.4f}"
                )

            # ── Slow down for real-time visualisation ──
            if gui and slow_mo:
                time.sleep(1.0 / env.CTRL_FREQ)

        # ── Episode summary ──
        final_x = info.get("x", obs[0])
        final_y = info.get("y", obs[1])
        final_z = info.get("z", obs[2])
        final_err = math.sqrt(final_x**2 + final_y**2 + (final_z - z_target)**2)
        status = "CRASHED" if terminated else "COMPLETED"
        mean_z_error = float(np.mean(np.abs(z_error_history)))

        print(f"\n  Episode {ep+1} {status}")
        print(f"  Steps:        {step}")
        print(f"  Total reward: {total_reward:+.2f}")
        print(f"  Final z:      {final_z:.4f} m")
        print(f"  Final error:  {final_err:.4f} m")
        print(f"  Mean Z Error: {mean_z_error:.4f} m")
        print()

        # ── Plotting ──
        print("  Generating plots...")
        rpms_array = np.array(rpms_history)
        z_err_array = np.array(z_error_history)
        steps_arr = np.arange(len(rpms_history))

        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        
        motor_labels = ['Motor 0 (FR)', 'Motor 1 (RL)', 'Motor 2 (FL)', 'Motor 3 (RR)']
        colors = ['blue', 'orange', 'green', 'purple']
        
        for i in range(4):
            axes[i].plot(steps_arr, rpms_array[:, i], label=motor_labels[i], color=colors[i])
            axes[i].set_ylabel("RPM")
            axes[i].legend(loc="upper right")
            axes[i].grid(True)
            if i == 0:
                axes[i].set_title(f"SAC DR Control - Motor RPMs (Target Z: {z_target}m)")

        axes[4].plot(steps_arr, z_err_array, color='red', label='Z Error (m)')
        axes[4].set_title("SAC DR Control - Hover Z-Error over Time")
        axes[4].set_xlabel("Steps")
        axes[4].set_ylabel("Error (m)")
        axes[4].axhline(0, color='black', linewidth=1, linestyle='--')
        axes[4].legend(loc="upper right")
        axes[4].grid(True)

        plt.tight_layout()
        plt.show()

    # ── Cleanup ──
    env.close()
    print("  Environment closed.  Done.\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Default model path: best DR model from training
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, "logs_hover_sac_dr", "best",
                                 "best_model")

    parser = argparse.ArgumentParser(
        description="Run trained DR SAC hover model on Crazyflie v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=default_model,
        help="Path to trained model (.zip, without extension)",
    )
    parser.add_argument(
        "--gui", action="store_true", default=True,
        help="Open PyBullet GUI for visualisation",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run headless (no GUI)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--no-slow-mo", action="store_true",
        help="Disable real-time slow-down",
    )
    args = parser.parse_args()

    gui = not args.no_gui
    slow_mo = not args.no_slow_mo

    # Verify model exists
    model_file = args.model if args.model.endswith(".zip") else args.model + ".zip"
    if not os.path.exists(model_file):
        print(f"\n  ✗ Model not found: {model_file}")
        print(f"    Train first:  python3 train_dr.py")
        print(f"    Or specify:   python3 inference_dr.py --model <path>\n")
        sys.exit(1)

    run_inference(
        model_path=args.model,
        gui=gui,
        num_episodes=args.episodes,
        slow_mo=slow_mo,
    )


if __name__ == "__main__":
    main()
