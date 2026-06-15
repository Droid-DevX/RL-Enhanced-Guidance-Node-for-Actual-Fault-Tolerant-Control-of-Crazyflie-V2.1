import os
import sys
import numpy as np
import pandas as pd
import argparse

from stable_baselines3 import SAC
from custom_hover_env import GoalConditionedHoverEnv
from custom_hover_env_dr import GoalConditionedHoverEnvDR
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

def run_single_experiment(exp_name, z_target=1.0, max_steps=1000):
    all_data = []

    print(f"Starting {exp_name}...")
    
    # 1. PID Agent
    print("Running PID Agent...")
    env_pid = GoalConditionedHoverEnv(gui=False)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, info = env_pid.reset()
    env_pid._z_target = z_target
    obs = env_pid._computeObs()
    ctrl.reset()
    
    done = False
    step = 0
    while not done and step < max_steps:
        state = env_pid._getDroneStateVector(0)
        target_pos = np.array([0.0, 0.0, z_target])
        rpm, _, _ = ctrl.computeControlFromState(
            control_timestep=1.0 / env_pid.CTRL_FREQ,
            state=state,
            target_pos=target_pos,
            target_rpy=np.zeros(3)
        )
        action = rpm / env_pid.MAX_RPM
        obs, reward, terminated, truncated, info = env_pid.step(action)
        done = terminated or truncated
        
        x = info.get("x", obs[0])
        y = info.get("y", obs[1])
        z = info.get("z", obs[2])
        z_error = z - z_target
        rpms = action * env_pid.MAX_RPM
        
        all_data.append({
            "step": step,
            "agent": "PID",
            "x": x, "y": y, "z": z,
            "z_error": z_error,
            "rpm0": rpms[0], "rpm1": rpms[1], "rpm2": rpms[2], "rpm3": rpms[3]
        })
        step += 1
    env_pid.close()

    # 2. Nominal SAC Agent
    print("Running Nominal SAC Agent...")
    model_nom_path = "logs_hover_sac/best/best_model.zip"
    model_nom = SAC.load(model_nom_path)
    env_nom = GoalConditionedHoverEnv(gui=False)
    obs, info = env_nom.reset()
    env_nom._z_target = z_target
    obs = env_nom._computeObs()
    
    done = False
    step = 0
    while not done and step < max_steps:
        action, _ = model_nom.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_nom.step(action)
        done = terminated or truncated
        
        x = info.get("x", obs[0])
        y = info.get("y", obs[1])
        z = info.get("z", obs[2])
        z_error = z - z_target
        rpms = action * env_nom.MAX_RPM
        
        all_data.append({
            "step": step,
            "agent": "SAC_Nominal",
            "x": x, "y": y, "z": z,
            "z_error": z_error,
            "rpm0": rpms[0], "rpm1": rpms[1], "rpm2": rpms[2], "rpm3": rpms[3]
        })
        step += 1
    env_nom.close()

    # 3. DR SAC Agent
    print("Running DR SAC Agent...")
    model_dr_path = "logs_hover_sac_dr/best/best_model.zip"
    model_dr = SAC.load(model_dr_path)
    env_dr = GoalConditionedHoverEnvDR(gui=False)
    obs, info = env_dr.reset()
    env_dr._z_target = z_target
    obs = env_dr._computeObs()
    
    done = False
    step = 0
    while not done and step < max_steps:
        action, _ = model_dr.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_dr.step(action)
        done = terminated or truncated
        
        x = info.get("x", obs[0])
        y = info.get("y", obs[1])
        z = info.get("z", obs[2])
        z_error = z - z_target
        rpms = action * env_dr.MAX_RPM
        
        all_data.append({
            "step": step,
            "agent": "SAC_DR",
            "x": x, "y": y, "z": z,
            "z_error": z_error,
            "rpm0": rpms[0], "rpm1": rpms[1], "rpm2": rpms[2], "rpm3": rpms[3]
        })
        step += 1
    env_dr.close()

    # Save to CSV
    df = pd.DataFrame(all_data)
    csv_name = f"{exp_name}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved {csv_name} with {len(df)} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="(0.741,0.741,1,1)", help="Name of the experiment CSV")
    parser.add_argument("--z_target", type=float, default=1.0, help="Target hover height")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per agent")
    args = parser.parse_args()
    
    run_single_experiment(args.name, args.z_target, args.steps)
