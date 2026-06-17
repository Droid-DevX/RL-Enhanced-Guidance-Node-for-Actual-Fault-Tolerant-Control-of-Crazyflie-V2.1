"""
run_experiment.py — Automated Fault-Injection Benchmark for Crazyflie v2 Hover Controllers
============================================================================================

This script runs a fully automated, multi-episode benchmark comparing three hover
controllers (Nominal SAC, Domain-Randomized SAC, and PID baseline) under four
motor-fault scenarios (test cases).  No user input is required.

EXPERIMENT DESIGN
-----------------
• 4 Test Cases (TC1–TC4): Each defines a per-motor fault vector applied via
  _preprocessAction monkeypatching.  TC1 is healthy (no fault); TC2–TC4 introduce
  increasing levels of motor degradation.
• 3 Controllers:
    - Nominal SAC  : trained on GoalConditionedHoverEnv (no fault during training)
    - DR SAC       : trained on GoalConditionedHoverEnvDR (motor randomization 60–100%)
    - PID          : DSLPIDControl from gym_pybullet_drones (classical baseline)
• 5 episodes per controller per TC, MAX_STEPS = 1000 per episode.
• z_target is fixed at 1.0 m (headless, no GUI, no slow-mo for speed).

FAULT INJECTION
---------------
  _preprocessAction is monkeypatched on the live env instance using types.MethodType.
  The fault vector is a 4-element array [f0, f1, f2, f3] where each fi ∈ (0, 1].
  A value of 1.0 means the motor is healthy; values < 1.0 simulate thrust loss.

  Note: The nominal env's _preprocessAction has a hardcoded fault vector already.
  The monkeypatch OVERRIDES that with the TC-specific vector, ensuring reproducibility.

OUTPUT
------
  • Console: Formatted table printed after each TC completes.
  • CSV    : experiment_results.csv saved in the script's directory.

Usage:
    python3 run_experiment.py

Author : Ayush
Date   : 2026-06-17
"""

import os
import sys
import csv
import math
import types

import numpy as np

# ── Make sure the project root is importable ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Stable Baselines 3 ────────────────────────────────────────────────────────
from stable_baselines3 import SAC

# ── Custom environments ───────────────────────────────────────────────────────
from custom_hover_env    import GoalConditionedHoverEnv
from custom_hover_env_dr import GoalConditionedHoverEnvDR

# ── PID controller (gym_pybullet_drones) ─────────────────────────────────────
try:
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel
except ImportError as e:
    print(f"\n  ✗ Could not import gym_pybullet_drones PID modules: {e}")
    print("    Ensure gym-pybullet-drones is installed.\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  EXPERIMENT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Fixed per-run settings ────────────────────────────────────────────────────
Z_TARGET     = 1.0    # Fixed hover target height (m) — no user input needed
MAX_STEPS    = 1000   # Maximum steps per episode before truncation
NUM_EPISODES = 5      # Number of episodes to run per controller per TC
GUI          = False  # Headless simulation for speed (no PyBullet GUI)
SLOW_MO      = False  # No real-time delay

# ── Model paths ───────────────────────────────────────────────────────────────
NOMINAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "logs_hover_sac",    "best", "best_model")
DR_MODEL_PATH      = os.path.join(SCRIPT_DIR, "logs_hover_sac_dr", "best", "best_model")

# ── CSV output path ───────────────────────────────────────────────────────────
CSV_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "experiment_results.csv")

# ── Test case fault vectors  [motor_0, motor_1, motor_2, motor_3] ─────────────
# TC1: No fault  — all motors at 100% efficiency (baseline)
# TC2: Motors 0 & 2 at 85% — symmetric diagonal pair degraded
# TC3: Motors 0 & 1 at 85% — one side degraded (induces roll torque)
# TC4: Motors 0 & 1 at 74.1% — severe degradation on one side
TEST_CASES = {
    "TC-1": [1.000, 1.000, 1.000, 1.000],
    "TC-2": [0.850, 1.000, 0.850, 1.000],
    "TC-3": [0.850, 0.850, 1.000, 1.000],
    "TC-4": [0.745, 0.745, 1.000, 1.000],
}

# ── Controller name constants (used in output and CSV) ────────────────────────
CTRL_NOMINAL = "Nominal"
CTRL_DR      = "DR SAC"
CTRL_PID     = "PID"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FAULT INJECTION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def apply_fault_monkeypatch(env, fault_vector: list):
    """
    Monkeypatch `env._preprocessAction` to apply the given per-motor fault
    vector, overriding any hardcoded FAULT_FACTOR inside the class.

    This is the preferred approach when we cannot (or should not) modify the
    environment source files directly.  The closure captures `fault_vector`
    so it is fixed for the lifetime of this env instance.

    Parameters
    ----------
    env          : GoalConditionedHoverEnv or GoalConditionedHoverEnvDR instance
    fault_vector : list of 4 floats, e.g. [0.85, 1.0, 0.85, 1.0]
                   Each element ∈ (0, 1]; 1.0 = healthy, <1.0 = degraded.
    """
    # Convert to numpy array and capture in closure for the bound method
    fv = np.array(fault_vector, dtype=np.float32)

    def patched_preprocess(self, action):
        """
        Patched _preprocessAction:
          rpm = action * MAX_RPM * FAULT_FACTOR
          Clipped to [0, MAX_RPM].

        FAULT_FACTOR is the per-motor degradation vector injected by the
        experiment runner, overriding the class-level hardcoded value.
        """
        FAULT_FACTOR = fv  # Per-motor fault vector from the TC definition
        rpm = action * self.MAX_RPM * FAULT_FACTOR
        return np.clip(rpm, 0.0, self.MAX_RPM)

    # Bind as a proper bound method on this specific env instance
    env._preprocessAction = types.MethodType(patched_preprocess, env)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SINGLE EPISODE RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_single_episode_sac(model, env, z_target: float, max_steps: int) -> dict:
    """
    Run one episode with a SAC controller (Nominal or DR).

    The env should already have been monkeypatched for fault injection before
    calling this function.  The z_target is forced by overriding env._z_target
    and recomputing the observation (same pattern as inference.py).

    Parameters
    ----------
    model     : stable_baselines3.SAC — pre-loaded, shared across episodes
    env       : GoalConditionedHoverEnv or GoalConditionedHoverEnvDR instance
    z_target  : float — fixed target altitude in metres
    max_steps : int   — hard cap on episode length

    Returns
    -------
    dict:
        mean_hover_error : float    — mean |z - z_target| across all steps
        crashed          : bool     — True if terminated before max_steps
        crash_step       : int|None — step number at crash (None if completed)
    """
    # ── Reset env and pin z_target ────────────────────────────────────────────
    obs, info = env.reset()
    env._z_target = z_target          # Override randomised target
    obs = env._computeObs()           # Recompute obs with correct z_target

    z_errors   = []
    step       = 0
    crashed    = False
    crash_step = None

    while step < max_steps:
        # Deterministic action from the SAC policy
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        # Record absolute z-error for this step
        z_cur = info.get("z", float(obs[2]))
        z_errors.append(abs(z_cur - z_target))

        if terminated:
            # Drone hit a crash condition (tilt, OOB, z too low/high)
            crashed    = True
            crash_step = step
            break

        if truncated:
            # Reached max_steps cleanly — episode completed
            break

    mean_hover_error = float(np.mean(z_errors)) if z_errors else float("nan")

    return {
        "mean_hover_error": mean_hover_error,
        "crashed":          crashed,
        "crash_step":       crash_step,
    }


def run_single_episode_pid(ctrl, env, z_target: float, max_steps: int) -> dict:
    """
    Run one episode with the DSLPIDControl baseline controller.

    The env should already have been monkeypatched for fault injection.
    The PID controller's internal integrators are reset at the start of each
    episode by calling ctrl.reset().

    Parameters
    ----------
    ctrl      : DSLPIDControl — shared PID controller instance
    env       : GoalConditionedHoverEnv instance
    z_target  : float — fixed target altitude in metres
    max_steps : int   — hard cap on episode length

    Returns
    -------
    dict:
        mean_hover_error : float    — mean |z - z_target| across all steps
        crashed          : bool     — True if terminated before max_steps
        crash_step       : int|None — step number at crash (None if completed)
    """
    # ── Reset env and PID integrators, pin z_target ───────────────────────────
    obs, info = env.reset()
    env._z_target = z_target
    obs = env._computeObs()
    ctrl.reset()  # Clear PID internal integral/derivative state

    target_pos = np.array([0.0, 0.0, z_target], dtype=np.float64)
    target_rpy = np.zeros(3, dtype=np.float64)

    z_errors   = []
    step       = 0
    crashed    = False
    crash_step = None

    while step < max_steps:
        # ── Fetch full 20-element drone state vector for PID ──────────────────
        # DSLPIDControl.computeControlFromState() requires the raw state vector
        # from BaseAviary (position, velocity, rotation matrix, angular rates).
        state = env._getDroneStateVector(0)

        # ── Compute PID RPM commands ──────────────────────────────────────────
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=1.0 / env.CTRL_FREQ,
            state=state,
            target_pos=target_pos,
            target_rpy=target_rpy,
        )

        # ── Normalise RPM → [0, 1] action space expected by env.step() ────────
        # The env internally scales action * MAX_RPM in _preprocessAction,
        # which will also apply the fault monkeypatch.
        action = rpm / env.MAX_RPM

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        # Record absolute z-error
        z_cur = info.get("z", float(obs[2]))
        z_errors.append(abs(z_cur - z_target))

        if terminated:
            crashed    = True
            crash_step = step
            break

        if truncated:
            break

    mean_hover_error = float(np.mean(z_errors)) if z_errors else float("nan")

    return {
        "mean_hover_error": mean_hover_error,
        "crashed":          crashed,
        "crash_step":       crash_step,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MULTI-EPISODE RUNNER FOR ONE CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

def run_controller_episodes(
    controller_name: str,
    fault_vector: list,
    num_episodes: int,
    z_target: float,
    max_steps: int,
    nominal_model,
    dr_model,
) -> dict:
    """
    Run `num_episodes` for a single controller under a given fault vector,
    then compute and return aggregated statistics.

    A fresh environment is created for each controller to avoid any cross-
    contamination of internal state between controllers.

    Parameters
    ----------
    controller_name : str   — CTRL_NOMINAL, CTRL_DR, or CTRL_PID
    fault_vector    : list  — 4-element per-motor fault multipliers [f0..f3]
    num_episodes    : int   — number of episodes to run
    z_target        : float — fixed target altitude in metres
    max_steps       : int   — max steps per episode
    nominal_model   : SAC   — pre-loaded nominal SAC model
    dr_model        : SAC   — pre-loaded DR SAC model

    Returns
    -------
    dict:
        mean_of_means      : float — mean of per-episode mean errors
        std_dev            : float — std dev of per-episode mean errors
        episodes_completed : int   — episodes that did NOT crash
        avg_crash_step     : float|None — mean crash step (None if no crashes)
        raw_results        : list  — one result dict per episode
    """
    episode_mean_errors = []
    crash_steps         = []
    episodes_completed  = 0
    raw_results         = []

    # ── Create a fresh environment for this controller ────────────────────────
    # Nominal and PID share GoalConditionedHoverEnv; DR uses the DR variant.
    if controller_name in (CTRL_NOMINAL, CTRL_PID):
        env = GoalConditionedHoverEnv(gui=GUI)
    else:  # CTRL_DR
        env = GoalConditionedHoverEnvDR(gui=GUI)

    # ── Inject the TC-specific fault vector via monkeypatch ───────────────────
    apply_fault_monkeypatch(env, fault_vector)

    # ── Create PID controller once; its reset() is called per-episode ─────────
    if controller_name == CTRL_PID:
        ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    else:
        ctrl = None

    # ── Run all episodes ──────────────────────────────────────────────────────
    for ep in range(num_episodes):
        if controller_name == CTRL_NOMINAL:
            result = run_single_episode_sac(nominal_model, env, z_target, max_steps)
        elif controller_name == CTRL_DR:
            result = run_single_episode_sac(dr_model,      env, z_target, max_steps)
        else:  # CTRL_PID
            result = run_single_episode_pid(ctrl,          env, z_target, max_steps)

        episode_mean_errors.append(result["mean_hover_error"])
        raw_results.append(result)

        if not result["crashed"]:
            episodes_completed += 1
        elif result["crash_step"] is not None:
            crash_steps.append(result["crash_step"])

    # ── Tear down the simulation ──────────────────────────────────────────────
    env.close()

    # ── Aggregate statistics across all episodes ──────────────────────────────
    valid_errors  = [e for e in episode_mean_errors if not math.isnan(e)]
    mean_of_means = float(np.mean(valid_errors)) if valid_errors else float("nan")
    std_dev       = float(np.std(valid_errors))  if valid_errors else float("nan")
    avg_crash_step = float(np.mean(crash_steps)) if crash_steps  else None

    return {
        "mean_of_means":      mean_of_means,
        "std_dev":            std_dev,
        "episodes_completed": episodes_completed,
        "avg_crash_step":     avg_crash_step,
        "raw_results":        raw_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN EXPERIMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Orchestrates the full benchmark:
      1. Validate trained model files exist on disk.
      2. Load both SAC models once (avoids repeated disk I/O across TCs).
      3. For each TC × controller pair, run NUM_EPISODES and collect statistics.
      4. Print a formatted results table to the console.
      5. Save all results to experiment_results.csv.
    """

    # ── 5.1  Validate model files exist on disk ───────────────────────────────
    nominal_zip = NOMINAL_MODEL_PATH + ".zip"
    dr_zip      = DR_MODEL_PATH      + ".zip"

    if not os.path.exists(nominal_zip):
        print(f"\n  ✗ Nominal model not found: {nominal_zip}")
        print("    Train first:  python3 train_nominal.py\n")
        sys.exit(1)

    if not os.path.exists(dr_zip):
        print(f"\n  ✗ DR model not found: {dr_zip}")
        print("    Train first:  python3 train_dr.py\n")
        sys.exit(1)

    # ── 5.2  Print experiment banner ──────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  FAULT-TOLERANT HOVER CONTROLLER BENCHMARK")
    print("═" * 70)
    print(f"  z_target       : {Z_TARGET:.1f} m  (fixed — no user input)")
    print(f"  Max steps/ep   : {MAX_STEPS}")
    print(f"  Episodes/run   : {NUM_EPISODES}")
    print(f"  Test cases     : {list(TEST_CASES.keys())}")
    print(f"  Controllers    : {CTRL_DR}, {CTRL_NOMINAL}, {CTRL_PID}")
    print(f"  GUI            : {GUI}")
    print(f"  Output CSV     : {CSV_OUTPUT_PATH}")
    print("═" * 70)

    # ── 5.3  Load SAC models once (shared across all TCs and episodes) ────────
    # Loading models inside the loop would incur unnecessary I/O overhead.
    print("\n  Loading SAC models...")
    nominal_model = SAC.load(NOMINAL_MODEL_PATH)
    print(f"  ✓ Nominal SAC  : {NOMINAL_MODEL_PATH}")
    dr_model = SAC.load(DR_MODEL_PATH)
    print(f"  ✓ DR SAC       : {DR_MODEL_PATH}\n")

    # ── 5.4  Storage for all result records (used for CSV export) ─────────────
    all_records = []  # List of flat dicts; one per TC × controller pair

    # ── 5.5  Main experiment loop: iterate over TCs then controllers ──────────
    for tc_name, fault_vector in TEST_CASES.items():
        print(f"\n{'─' * 70}")
        print(f"  {tc_name}  |  Fault vector: {fault_vector}")
        print(f"{'─' * 70}")

        tc_results = {}  # Maps controller_name → stats dict for this TC

        # Run each controller under this TC's fault conditions
        for ctrl_name in [CTRL_DR, CTRL_NOMINAL, CTRL_PID]:
            print(
                f"  [{tc_name}] Running {ctrl_name:<9} "
                f"({NUM_EPISODES} episodes) ...",
                end="", flush=True
            )

            stats = run_controller_episodes(
                controller_name=ctrl_name,
                fault_vector=fault_vector,
                num_episodes=NUM_EPISODES,
                z_target=Z_TARGET,
                max_steps=MAX_STEPS,
                nominal_model=nominal_model,
                dr_model=dr_model,
            )

            tc_results[ctrl_name] = stats
            print(f"  done.  "
                  f"(Mean: {stats['mean_of_means']:.4f} m, "
                  f"Completed: {stats['episodes_completed']}/{NUM_EPISODES})")

        # ── 5.6  Print formatted results for this TC in the requested format ──
        print()
        for ctrl_name in [CTRL_DR, CTRL_NOMINAL, CTRL_PID]:
            s = tc_results[ctrl_name]
            print(
                f"  {tc_name} | {ctrl_name:<9} | "
                f"Mean: {s['mean_of_means']:.4f} | "
                f"Std: {s['std_dev']:.4f} | "
                f"Completed: {s['episodes_completed']}/{NUM_EPISODES}"
            )

        # ── 5.7  Accumulate flat records for CSV ──────────────────────────────
        for ctrl_name in [CTRL_DR, CTRL_NOMINAL, CTRL_PID]:
            s = tc_results[ctrl_name]
            avg_cs = s["avg_crash_step"]
            all_records.append({
                "TC":                 tc_name,
                "Controller":         ctrl_name,
                "Mean_Error":         f"{s['mean_of_means']:.6f}",
                "Std_Dev":            f"{s['std_dev']:.6f}",
                "Episodes_Completed": s["episodes_completed"],
                "Avg_Crash_Step":     f"{avg_cs:.1f}" if avg_cs is not None else "N/A",
            })

    # ── 5.8  Print consolidated summary banner ────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  EXPERIMENT COMPLETE — FULL SUMMARY")
    print(f"{'═' * 70}")
    header = f"  {'TC':<6} | {'Controller':<9} | {'Mean':>8} | {'Std':>8} | Completed"
    print(header)
    print(f"  {'─' * 62}")
    for rec in all_records:
        print(
            f"  {rec['TC']:<6} | {rec['Controller']:<9} | "
            f"{float(rec['Mean_Error']):>8.4f} | "
            f"{float(rec['Std_Dev']):>8.4f} | "
            f"{rec['Episodes_Completed']}/{NUM_EPISODES}"
        )
    print(f"{'═' * 70}")

    # ── 5.9  Write results to CSV ─────────────────────────────────────────────
    csv_fields = [
        "TC", "Controller", "Mean_Error", "Std_Dev",
        "Episodes_Completed", "Avg_Crash_Step",
    ]

    with open(CSV_OUTPUT_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n  ✓ Results saved to: {CSV_OUTPUT_PATH}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
