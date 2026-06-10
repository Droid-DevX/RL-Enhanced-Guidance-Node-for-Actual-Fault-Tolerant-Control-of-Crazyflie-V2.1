"""
inference_pid.py — PID Controller for Goal-Conditioned Hover
============================================================
Uses a standard PID controller to maintain the drone at a 
user-specified height, for comparison against the RL model.

Usage:
    python3 inference_pid.py
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_hover_env import GoalConditionedHoverEnv, Z_TARGET_MIN, Z_TARGET_MAX

try:
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel
except ImportError:
    print("Error: Could not import gym_pybullet_drones modules for PID control.")
    sys.exit(1)


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


def run_inference(gui: bool = True, num_episodes: int = 1, slow_mo: bool = True):
    """
    Run inference episodes using a PID controller instead of an RL model.
    """
    print("\n  Initializing PID Controller...")
    
    # ── Create environment ──
    env = GoalConditionedHoverEnv(gui=gui)
    
    # ── Initialize PID controller ──
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    
    for ep in range(num_episodes):
        # ── Get target from user ──
        z_target = get_target_height_from_user()

        print(f"\n{'─'*60}")
        print(f"  Episode {ep+1}/{num_episodes} (PID Control)")
        print(f"  Target height: {z_target:.2f} m")
        print(f"{'─'*60}")
        print(f"  {'Step':>6}  {'x,y (m)':>12}  {'z (m)':>8}  {'z_target':>8}  "
              f"{'error (m)':>10}  {'reward':>8}")
        print(f"  {'─'*65}")

        # ── Reset and override z_target ──
        obs, info = env.reset()
        env._z_target = z_target
        obs = env._computeObs()
        ctrl.reset()

        total_reward = 0.0
        step = 0
        done = False

        rpms_history = []
        z_error_history = []

        while not done:
            # We need the full state for DSLPIDControl
            state = env._getDroneStateVector(0)
            
            target_pos = np.array([0.0, 0.0, z_target])
            
            # Compute action using PID
            rpm, _, _ = ctrl.computeControlFromState(
                control_timestep=1.0 / env.CTRL_FREQ,
                state=state,
                target_pos=target_pos,
                target_rpy=np.zeros(3)
            )
            
            # Convert RPM back to the [0, 1] action space expected by custom_hover_env
            action = rpm / env.MAX_RPM

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
                axes[i].set_title(f"PID Control - Motor RPMs (Target Z: {z_target}m)")

        axes[4].plot(steps_arr, z_err_array, color='red', label='Z Error (m)')
        axes[4].set_title("PID Control - Hover Z-Error over Time")
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


def main():
    parser = argparse.ArgumentParser(
        description="Run PID hover model on Crazyflie v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    run_inference(
        gui=gui,
        num_episodes=args.episodes,
        slow_mo=slow_mo,
    )


if __name__ == "__main__":
    main()
