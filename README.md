<div align="center">
  <h1> Crazyflie V2.1 Fault-Tolerant Hover Benchmark</h1>
  <h3>Benchmarking PID vs. Zero-Shot SAC vs. Domain Randomization</h3>

  <p><i>A reinforcement learning pipeline for training and evaluating a Crazyflie V2.1 nano-quadrotor's ability to hover under motor failure conditions in PyBullet.</i></p>

  <br/>
  <img src="CF2x.png" alt="Trained DR SAC agent demonstrating stable hover under adjacent-motor fault at 26% degradation" width="600" style="border-radius: 8px; margin: 15px 0;"/>
  <p>
    <img src="https://img.shields.io/badge/DR_SAC_Reward_(Nominal)-2929.98-blue?style=for-the-badge" alt="DR SAC Reward" />
    <img src="https://img.shields.io/badge/Mean_Z--Error_(TC--1)-0.019m-green?style=for-the-badge" alt="Mean Z-Error" />
    <img src="https://img.shields.io/badge/Fault_Threshold_(DR_SAC)->26%25_adj.-orange?style=for-the-badge" alt="Fault Threshold" />
  </p>
</div>

---

##  Benchmarking Overview

This repository benchmarks three distinct control approaches against progressive motor degradation (fault injection). The goal is to evaluate robustness and steady-state hover accuracy when motors lose efficiency.

1. **PID (Baseline)**: A standard cascaded PID controller, tuned for nominal flight.
2. **Zero-Shot SAC (Nominal)**: A Soft Actor-Critic agent trained *only* in nominal, fault-free conditions, tested "zero-shot" under unforeseen faults.
3. **Domain Randomization SAC (DR)**: A SAC agent trained with domain randomization (motor efficiencies uniformly sampled between 60% and 100% per episode), forcing it to implicitly learn redundant, fault-tolerant control allocation.

---

##  Benchmark Results

We evaluated the controllers against four test cases (TC), increasing in severity. Faults are represented by an efficiency vector `α = [m1, m2, m3, m4]`.

| Metric | PID (Baseline) | Zero-Shot SAC | DR SAC |
| :--- | :--- | :--- | :--- |
| **TC-1: Nominal `α = [1, 1, 1, 1]`** | +2934.01 | +2841.43 | **+2929.98** |
| Mean Z-Error (TC-1) | 0.0219 m | 0.0280 m | **0.0190 m** |
| **TC-2: Alt-motor 15% fault `α = [0.85, 1, 0.85, 1]`** | Yaw drift | Stable | **Stable** |
| **TC-3: Adj-motor 15% fault `α = [0.85, 0.85, 1, 1]`** | **Crash (step 6)** | Stable | **Stable** |
| **TC-4: Adj-motor 26% fault `α = [0.74, 0.74, 1, 1]`** | **Crash (step 5)** |  **Crash (step 657)** |  **Survived** |
| **Max Fault Tolerance** | < 15% adj. | ~20% adj. | **> 26% adj.** |

###  Deep Dive: Test Case Breakdown

#### TC-1 — Nominal (No Faults)
```text
PID           | Steps: 1001 | Reward: +2934.01 | Mean z-err: 0.0219 m
Zero-Shot SAC | Steps: 1001 | Reward: +2841.43 | Mean z-err: 0.0280 m
DR SAC        | Steps: 1001 | Reward: +2929.98 | Mean z-err: 0.0190 m
```

#### TC-2 — Alternate-motor 15% fault
```text
PID           | Steps: 1001 | Reward: +2803.94 | Mean z-err: 0.0497 m  [CONTINUOUS YAW ROTATION]
Zero-Shot SAC | Steps: 1001 | Reward: +2813.99 | Mean z-err: 0.0358 m
DR SAC        | Steps: 1001 | Reward: +2686.70 | Mean z-err: 0.0347 m
```

#### TC-3 — Adjacent-motor 15% fault
```text
PID           | Steps:    6 |  CRASHED immediately (z: 0.166 m)
Zero-Shot SAC | Steps: 1001 | Reward: +2482.50 | Mean z-err: 0.0507 m
DR SAC        | Steps: 1001 | Reward: +2549.12 | Mean z-err: 0.0414 m
```

#### TC-4 — Adjacent-motor 26% fault
```text
PID           | Steps:    5 |  CRASHED immediately (z: 0.135 m)
Zero-Shot SAC | Steps:  657 |  CRASHED at step 657 | Final z: 0.268 m
DR SAC        | Steps: 1001 | Reward: +2472.83   | Mean z-err: 0.0605 m  [SOLE SURVIVOR]
```

---

##  How it Works

**State Space (16-dim):** 
Position (x, y, z), linear velocity, Euler angles, angular rates, goal altitude (`z_target`), integral error accumulators (`Ix, Iy, Iz`).

**Action Space (4-dim):** 
Normalized per-motor RPM commands ∈ `[0, 1]`, scaled to `MAX_RPM`. Direct per-motor control enables fault injection by multiplying outputs by an efficiency vector `α`.

**Training Approaches:**
- **Zero-Shot SAC**: Fixed `α = [1, 1, 1, 1]`, trained for 1.5M timesteps.
- **DR SAC**: Motor efficiency randomized `f_ep ~ U[0.60, 1.00]` each episode, trained for 2M timesteps. The agent learns fault-robust motor redistribution implicitly to avoid crashing during low-efficiency episodes.

**Fault-Tolerance Mechanism:**
PID fails under adjacent-motor faults because its rigid cascade structure applies full corrective torque instantly, driving the drone into the ground. Zero-Shot SAC learns a somewhat smooth, robust action distribution just through exploration, but struggles with severe faults. DR SAC explicitly learns to map severe asymmetric state errors into redistributed thrust commands, surviving 26% degradation completely unseen during its `[0.60, 1.00]` randomized (but typically uniform) training.

---

##  Project Structure

```text
Crazyflie-FaultTolerance/
├── custom_hover_env.py     # Nominal Gymnasium env & reward function
├── custom_hover_env_dr.py  # Environment with Domain Randomization logic
├── train_nominal.py        # SAC training script for Zero-Shot agent
├── train_dr.py             # SAC training script for DR agent
├── inference.py            # Benchmark script for Zero-Shot SAC
├── inference_dr.py         # Benchmark script for DR SAC
├── inference_pid.py        # Benchmark script for PID baseline
├── run_experiment.py       # Automated multi-agent experiment runner
├── plot_ieee.py            # IEEE-compliant results plotter
├── logs_hover_sac/         # Saved models for Zero-Shot SAC
└── logs_hover_sac_dr/      # Saved models for DR SAC
```

---

##  Setup & Usage

### Prerequisites
Requires Python 3.8+.
```bash
pip install stable-baselines3 pybullet gymnasium torch numpy matplotlib gym-pybullet-drones pandas
```

### Running the Benchmarks

```bash
# Run the automated benchmark for all controllers and save to CSV
python run_experiment.py --name "(0.741,0.741,1,1)" --z_target 1.0 --steps 1000

# Generate IEEE-compliant comparison plots
python plot_ieee.py --csv "(0.741,0.741,1,1).csv"

# Optional: Evaluate individual controllers
python inference.py
python inference_dr.py
python inference_pid.py
```

### Training the Models

```bash
# 1. Train Zero-Shot (Nominal) Agent
python train_nominal.py

# 2. Train Domain Randomization Agent
python train_dr.py
```

---

##  Key Design Decisions

- **Goal-Conditioned Architecture**: A dynamic `z_target ∈ [0.2, 2.5] m` randomized each episode ensures the policy generalizes across altitudes rather than memorizing one fixed point.
- **Integral Error Augmentation**: Passing `(Ix, Iy, Iz)` to the SAC observation vector gives the agent learnable I-term memory, mimicking a PID controller to drastically reduce steady-state altitude error.
- **Smoothness Penalty**: Penalizing large delta actions (`|a_t − a_{t−1}|`) is critical. Unpenalized SAC outputs jittery RPM commands that would destroy physical motors.

---

