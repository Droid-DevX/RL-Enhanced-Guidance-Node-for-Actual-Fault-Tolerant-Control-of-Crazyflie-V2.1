"""
custom_hover_env_dr.py — Domain Randomized Goal-Conditioned Hover Environment
===============================================================================
A domain-randomized variant of GoalConditionedHoverEnv for training a
fault-tolerant SAC agent on the Crazyflie v2X (CF2X) drone.

DOMAIN RANDOMIZATION STRATEGY:
  • At the start of every episode, a thrust_factor is sampled uniformly
    from [0.60, 1.00].  This factor is applied equally to all 4 motors
    in _preprocessAction(), meaning the agent trains with motors running
    anywhere from 60% to 100% efficiency.
  • This forces the policy to learn a robust controller that can compensate
    for reduced motor thrust — a common real-world fault scenario.
  • At test time, a specific fault can be injected by hardcoding
    self._thrust_factor directly in _preprocessAction() (no user input
    needed; the researcher edits the code).

EVERYTHING ELSE is identical to custom_hover_env.py:
  • Observation space: 16-dim  [x, y, z, vx, vy, vz, r, p, y, wx, wy, wz,
    z_target, int_x, int_y, int_z]
  • Action space: Box(0, 1, shape=(4,))
  • Reward function: same exponential axis rewards + integral penalty
  • Termination: same crash / tilt / drift conditions

Author : Ayush
Date   : 2026-06-06
"""

import math
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


# ──────────────────────────────────────────────────────────────────────────────
# Constants  (identical to nominal env)
# ──────────────────────────────────────────────────────────────────────────────
Z_TARGET_MIN = 0.2          # Minimum target altitude (m)
Z_TARGET_MAX = 2.5          # Maximum target altitude (m)
MAX_EPISODE_STEPS = 1000    # Steps before truncation
CRASH_Z_MIN = 0.05          # Crash if z drops below this
CRASH_Z_MAX = 2.5           # Out-of-bounds ceiling
MAX_XY_DRIFT = 2.0          # Out-of-bounds horizontal drift
MAX_TILT_RAD = math.radians(60)  # 60° in radians — episode ends if exceeded

# ── Domain Randomization bounds ──
THRUST_FACTOR_MIN = 0.60    # Minimum motor efficiency (60%)
THRUST_FACTOR_MAX = 1.00    # Maximum motor efficiency (100%)


class GoalConditionedHoverEnvDR(BaseAviary):
    """
    Domain-Randomized Goal-Conditioned Hover Environment.

    Identical to GoalConditionedHoverEnv except that motor thrust efficiency
    is randomized every episode to train a fault-tolerant policy.
    """

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def __init__(self, gui: bool = False, pyb_freq: int = 240,
                 ctrl_freq: int = 48, record: bool = False):
        # ── Episode-level state (identical to nominal) ──
        self._z_target: float = 1.0
        self._step_count: int = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._current_raw_action = np.zeros((1, 4), dtype=np.float32)
        self._terminated_flag: bool = False

        # Integral error terms to eliminate steady-state offset
        self._int_x: float = 0.0
        self._int_y: float = 0.0
        self._int_z: float = 0.0

        # ── Domain Randomization state ──
        # Initialised to 1.0 (no fault).  Randomised in reset().
        self._thrust_factor: float = 1.0

        # ── Call BaseAviary constructor ──
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=np.inf,
            initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
            initial_rpys=np.array([[0.0, 0.0, 0.0]]),
            physics=Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=False,
            user_debug_gui=False,
        )

        # ── Override action & observation spaces ──
        self.action_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 16-dim vector (same as nominal)
        # [x, y, z, vx, vy, vz, r, p, y, wx, wy, wz, z_tgt, int_x, int_y, int_z]
        obs_lo = np.array([-5.0, -5.0, 0.0, -10.0, -10.0, -10.0,
                           -math.pi, -math.pi, -math.pi,
                           -20.0, -20.0, -20.0, Z_TARGET_MIN,
                           -2.0, -2.0, -2.0], dtype=np.float32)
        obs_hi = np.array([5.0, 5.0, 3.0, 10.0, 10.0, 10.0,
                           math.pi, math.pi, math.pi,
                           20.0, 20.0, 20.0, Z_TARGET_MAX,
                           2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_lo, high=obs_hi, dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    # Reset — now includes domain randomization of thrust efficiency
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        obs_raw, info = super().reset(seed=seed, options=options)

        rng = np.random.default_rng(seed)
        self._z_target = float(rng.uniform(Z_TARGET_MIN, Z_TARGET_MAX))

        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._current_raw_action = np.zeros((1, 4), dtype=np.float32)
        self._terminated_flag = False

        self._int_x = 0.0
        self._int_y = 0.0
        self._int_z = 0.0

        # ── Domain Randomization ──
        # Sample a new thrust factor for this episode.  This means every
        # episode the agent sees motors running at a different efficiency,
        # forcing it to learn a policy robust to motor degradation.
        self._thrust_factor = float(
            np.random.uniform(THRUST_FACTOR_MIN, THRUST_FACTOR_MAX)
        )

        init_z = 0.1
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            [0.0, 0.0, init_z],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.CLIENT,
        )
        p.resetBaseVelocity(
            self.DRONE_IDS[0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.CLIENT,
        )

        self._updateAndStoreKinematicInformation()

        return self._computeObs(), self._computeInfo()

    # ------------------------------------------------------------------ #
    # Step  (identical to nominal)
    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = super().step(
            action.reshape(1, 4)
        )
        self._step_count += 1
        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Action preprocessing — DOMAIN RANDOMIZATION APPLIED HERE
    # ------------------------------------------------------------------ #
    def _preprocessAction(self, action):
        # Store raw action for reward calculation
        self._current_raw_action = action.copy()
        
        # Thrust is proportional to RPM^2. To simulate a thrust efficiency fault 
        # (e.g. 60% thrust), we must multiply the RPM by sqrt(0.6).
        rpm_factor = math.sqrt(self._thrust_factor)
        # FAULT_FACTOR = np.array([rpm_factor,
        #                          rpm_factor,
        #                          rpm_factor,
        #                          rpm_factor])
        FAULT_FACTOR = np.array([1.0, 1.0, 1.0, 1.0])
        rpm = action * self.MAX_RPM * FAULT_FACTOR
        return np.clip(rpm, 0.0, self.MAX_RPM)
    # ------------------------------------------------------------------ #
    # Observation  (identical to nominal)
    # ------------------------------------------------------------------ #
    def _computeObs(self):
        state = self._getDroneStateVector(0)

        x       = float(state[0])
        y       = float(state[1])
        z       = float(state[2])
        vx      = float(state[10])
        vy      = float(state[11])
        vz      = float(state[12])
        roll    = float(state[7])
        pitch   = float(state[8])
        yaw     = float(state[9])
        wx      = float(state[13])
        wy      = float(state[14])
        wz      = float(state[15])

        # Integrate errors
        dt = 1.0 / self.CTRL_FREQ
        self._int_x = float(np.clip(self._int_x + x * dt, -2.0, 2.0))
        self._int_y = float(np.clip(self._int_y + y * dt, -2.0, 2.0))
        self._int_z = float(np.clip(self._int_z + (z - self._z_target) * dt, -2.0, 2.0))

        obs = np.array([
            x, y, z,
            vx, vy, vz,
            roll, pitch, yaw,
            wx, wy, wz,
            self._z_target,
            self._int_x, self._int_y, self._int_z
        ], dtype=np.float32)

        return np.clip(obs, self.observation_space.low,
                       self.observation_space.high)

    # ------------------------------------------------------------------ #
    # Reward  (identical to nominal)
    # ------------------------------------------------------------------ #
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        x     = float(state[0])
        y     = float(state[1])
        z     = float(state[2])
        vx    = float(state[10])
        vy    = float(state[11])
        roll  = float(state[7])
        pitch = float(state[8])

        # 1. Primary: Independent axis rewards (sharp gradients near zero)
        # exp(-4 * err) creates a strong peak specifically at 0.0
        err_x = abs(x)
        err_y = abs(y)
        err_z = abs(z - self._z_target)

        r_x = float(np.exp(-4.0 * err_x))
        r_y = float(np.exp(-4.0 * err_y))
        r_z = float(np.exp(-4.0 * err_z))

        r = r_x + r_y + r_z  # Max +3.0 when perfectly at origin

        # 2. Integral penalty to eliminate steady-state offset
        r -= 0.05 * (abs(self._int_x) + abs(self._int_y) + abs(self._int_z))

        # 3. Horizontal drift penalty
        r -= 0.05 * math.sqrt(vx**2 + vy**2)

        # 4. Stability penalty: penalise tilt
        r -= 0.1 * (abs(roll) + abs(pitch))

        # 5. Action smoothness penalty
        # Use the stored raw action from _preprocessAction to avoid penalizing domain randomization
        current_action_norm = self._current_raw_action[0]
        prev_action_norm = self._prev_action
        r -= 0.05 * float(np.sum(np.abs(current_action_norm - prev_action_norm)))

        # 6. Crash penalty
        crashed = (z < CRASH_Z_MIN) or (z > CRASH_Z_MAX) or \
                  (abs(x) > MAX_XY_DRIFT) or (abs(y) > MAX_XY_DRIFT) or \
                  (abs(roll) > MAX_TILT_RAD) or (abs(pitch) > MAX_TILT_RAD)
        if crashed:
            r -= 10.0

        return float(r)

    # ------------------------------------------------------------------ #
    # Termination conditions  (identical to nominal)
    # ------------------------------------------------------------------ #
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        x     = float(state[0])
        y     = float(state[1])
        z     = float(state[2])
        roll  = float(state[7])
        pitch = float(state[8])

        crashed = False

        if z < CRASH_Z_MIN or z > CRASH_Z_MAX:
            crashed = True

        if abs(x) > MAX_XY_DRIFT or abs(y) > MAX_XY_DRIFT:
            crashed = True

        if abs(roll) > MAX_TILT_RAD or abs(pitch) > MAX_TILT_RAD:
            crashed = True

        self._terminated_flag = crashed
        return crashed

    # ------------------------------------------------------------------ #
    # Truncation (time limit)  (identical to nominal)
    # ------------------------------------------------------------------ #
    def _computeTruncated(self):
        return self._step_count >= MAX_EPISODE_STEPS

    # ------------------------------------------------------------------ #
    # Info dict — includes thrust_factor for DR logging
    # ------------------------------------------------------------------ #
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        x = float(state[0])
        y = float(state[1])
        z = float(state[2])
        return {
            "x":              x,
            "y":              y,
            "z":              z,
            "z_target":       self._z_target,
            "hover_error":    math.sqrt(x**2 + y**2 + (z - self._z_target)**2),
            "step":           self._step_count,
            "thrust_factor":  self._thrust_factor,  # DR: log motor efficiency
        }

    # ------------------------------------------------------------------ #
    # Required overrides that BaseAviary expects
    # ------------------------------------------------------------------ #
    def _actionSpace(self):
        return spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

    def _observationSpace(self):
        obs_lo = np.array([-5.0, -5.0, 0.0, -10.0, -10.0, -10.0,
                           -math.pi, -math.pi, -math.pi,
                           -20.0, -20.0, -20.0, Z_TARGET_MIN,
                           -2.0, -2.0, -2.0], dtype=np.float32)
        obs_hi = np.array([5.0, 5.0, 3.0, 10.0, 10.0, 10.0,
                           math.pi, math.pi, math.pi,
                           20.0, 20.0, 20.0, Z_TARGET_MAX,
                           2.0, 2.0, 2.0], dtype=np.float32)
        return spaces.Box(low=obs_lo, high=obs_hi, dtype=np.float32)

    def _addObstacles(self):
        pass

    def render(self):
        pass
