# hardware_exp/

Real hardware telemetry logging and plotting for Crazyflie v2.1 flights.

## Folder structure
```
hardware_exp/
├── hw_logger.py        ← Terminal 4: logs telemetry to CSV during flight
├── plot_hw_ieee.py     ← Generates IEEE-style plots from logged CSVs
├── logs/               ← Auto-saved CSVs (one per flight run)
└── plots/              ← Output PNG/PDF figures
```

## Workflow

### Step 1 — Fly the drone (Terminals 1–3 as usual)
```bash
# Terminal 1
source ~/crazyflie_ws/install/setup.bash
ros2 launch crazyflie launch.py mocap:=False gui:=False

# Terminal 2 (choose one controller)
ros2 launch launch_policy.py       # SAC-DR
ros2 launch launch_pid.py          # PID

# Terminal 3
python3 z_target_publisher.py
```

### Step 2 — Log telemetry (Terminal 4)
```bash
cd ~/Fault_Tolerant
source ~/crazyflie_ws/install/setup.bash

# No fault, SAC-DR:
python3 hardware_exp/hw_logger.py --controller sac_dr --fault 1.0_1.0_1.0_1.0

# 30% loss motor 2, PID:
python3 hardware_exp/hw_logger.py --controller pid --fault 1.0_1.0_0.7_1.0

# Press Ctrl+C to stop and save the CSV
```

### Step 3 — Plot results
```bash
# Show plots interactively:
python3 hardware_exp/plot_hw_ieee.py

# Save as PNG (300 DPI, IEEE width):
python3 hardware_exp/plot_hw_ieee.py --save

# Filter by fault scenario:
python3 hardware_exp/plot_hw_ieee.py --fault 1.0_1.0_0.7_1.0 --save
```

## Fault factor naming convention
| `--fault` value | Meaning |
|-----------------|---------|
| `1.0_1.0_1.0_1.0` | No fault |
| `1.0_1.0_0.7_1.0` | 30% thrust loss — motor 2 |
| `0.8_0.8_0.8_0.8` | 20% thrust loss — all motors |
| `0.6_0.6_0.6_0.6` | 40% thrust loss — all motors |

## CSV columns
| Column | Unit | Description |
|--------|------|-------------|
| `time_s` | s | Elapsed time since logging started |
| `x`, `y`, `z` | m | Drone position (Lighthouse v2) |
| `z_target` | m | Target hover height |
| `hover_error` | m | 3D Euclidean distance from target |
| `roll_deg`, `pitch_deg`, `yaw_deg` | ° | Attitude |
| `controller` | — | `pid` / `sac_nominal` / `sac_dr` |
| `fault` | — | Fault factor string |
