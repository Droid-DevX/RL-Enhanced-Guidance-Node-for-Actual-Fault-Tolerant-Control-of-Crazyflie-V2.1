#!/usr/bin/env python3
"""
policy_node.py — ROS2 SAC Policy Deployment Node for Crazyflie v2.1
====================================================================
Deploys a trained SAC model onto a real Crazyflie v2.1 via ROS2 Humble
and Lighthouse v2 positioning (Crazyswarm2 / crazyflie_server).

NOTE ON TOPICS:
    Crazyswarm2's crazyflie_server publishes:
        /cf231/pose  → geometry_msgs/PoseStamped  (Lighthouse: position + orientation)
        /cf231/imu   → sensor_msgs/Imu            (angular rates) — OPTIONAL
    Velocity is computed via finite differencing of /cf231/pose.
    If /cf231/imu is not available, angular rates default to 0.

Subscribes:
    /cf231/pose   → geometry_msgs/PoseStamped  (required)
    /cf231/imu    → sensor_msgs/Imu            (optional — uses 0 if missing)
    /z_target     → std_msgs/Float32           (target hover height)

Publishes:
    /cf231/cmd_full_state → crazyflie_interfaces/FullState
        Collective thrust = sum(action_i^2 * 65535) for i in [0,3]
        Attitude setpoint = zero (Mellinger handles stabilisation)
        This is the native topic that crazyflie_server subscribes to.

THRUST CONVERSION (SAFETY-CRITICAL):
    rpm    = action * MAX_RPM          (MAX_RPM = 21702.5)
    thrust = (rpm / MAX_RPM)^2 * 65535 = action^2 * 65535
    thrust = int(clip(thrust, 0, 65535))

EMERGENCY STOP:
    Applies ONLY after drone has been airborne (z > 0.1m at least once).
    Triggers on: |roll|>45°  |pitch|>45°  z>2.8m  |x|>2.0m  |y|>2.0m
    And:         z < 0.05m  (only after having been airborne)

Author: Ayush   Date: 2026-06-21
"""

import math, os, sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from crazyflie_interfaces.msg import FullState
from stable_baselines3 import SAC

# ── Constants ──
MAX_RPM_DEFAULT = 21702.5
CTRL_FREQ_DEFAULT = 48
ESTOP_ROLL_DEG  = 45.0
ESTOP_PITCH_DEG = 45.0
ESTOP_Z_MIN  = 0.05    # Only checked after drone has been airborne
ESTOP_Z_MAX  = 2.8
ESTOP_XY_MAX = 2.0
AIRBORNE_THRESH = 0.10  # Drone considered airborne once z > this (m)

# Observation bounds (must match simulation exactly)
OBS_LOW = np.array([-5.0,-5.0,0.0, -10.0,-10.0,-10.0,
    -math.pi,-math.pi,-math.pi, -20.0,-20.0,-20.0,
    0.2, -2.0,-2.0,-2.0], dtype=np.float32)
OBS_HIGH = np.array([5.0,5.0,3.0, 10.0,10.0,10.0,
    math.pi,math.pi,math.pi, 20.0,20.0,20.0,
    2.5, 2.0,2.0,2.0], dtype=np.float32)


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to Euler (roll, pitch, yaw) in radians. ZYX convention."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class SACPolicyNode(Node):
    """ROS2 node: runs trained SAC policy on Crazyflie v2.1 at 48 Hz."""

    def __init__(self):
        super().__init__('sac_policy_node')

        # ── Declare & read parameters ──
        self.declare_parameter('model_path', 'logs_hover_sac_dr/best/best_model')
        self.declare_parameter('max_rpm', MAX_RPM_DEFAULT)
        self.declare_parameter('ctrl_freq', CTRL_FREQ_DEFAULT)
        self.declare_parameter('fault_factor', [1.0, 1.0, 1.0, 1.0])
        self.declare_parameter('z_target_default', 1.0)
        # action_alpha: exponential moving average for action smoothing
        # 1.0 = no smoothing (use full new action)
        # 0.3 = heavy smoothing (good for nominal SAC sim-to-real)
        # 0.6 = light smoothing (good for DR-SAC, already robust)
        self.declare_parameter('action_alpha', 0.6)

        model_path  = self.get_parameter('model_path').get_parameter_value().string_value
        self._max_rpm = self.get_parameter('max_rpm').get_parameter_value().double_value
        ctrl_freq   = self.get_parameter('ctrl_freq').get_parameter_value().integer_value
        fault_list  = self.get_parameter('fault_factor').get_parameter_value().double_array_value
        z_default   = self.get_parameter('z_target_default').get_parameter_value().double_value
        self._action_alpha = self.get_parameter('action_alpha').get_parameter_value().double_value

        # FAULT_FACTOR: [1.0,1.0,1.0,1.0]=no fault; [1.0,1.0,0.7,1.0]=30% loss on motor 2
        self._fault_factor = np.array(fault_list, dtype=np.float64)
        assert self._fault_factor.shape == (4,)

        # ── Load SAC model ──
        self.get_logger().info(f'Loading SAC model from: {model_path}')
        model_file = model_path if model_path.endswith('.zip') else model_path + '.zip'
        if not os.path.exists(model_file):
            self.get_logger().fatal(f'Model NOT FOUND: {model_file}')
            sys.exit(1)
        self._model = SAC.load(model_path)
        self.get_logger().info('✓ SAC model loaded.')

        # ── State variables ──
        self._z_target = z_default
        self._x = 0.0;  self._y = 0.0;  self._z = 0.0
        self._roll = 0.0; self._pitch = 0.0; self._yaw = 0.0
        # Angular rates — default 0 if IMU not available
        self._wx = 0.0;  self._wy = 0.0;  self._wz = 0.0
        # Velocity via finite-difference of pose (pose arrives at ~10 Hz)
        self._vx = 0.0;  self._vy = 0.0;  self._vz = 0.0
        self._prev_x = None  # None means no previous pose yet
        self._prev_y = 0.0;  self._prev_z = 0.0
        self._prev_pose_time = None
        self._vel_alpha = 0.3   # Low-pass filter coefficient

        # Integral errors — clipped to [-2,2] exactly like simulation
        self._int_x = 0.0; self._int_y = 0.0; self._int_z = 0.0

        # ── Readiness & safety state ──
        # Only /cf231/pose is required. IMU is optional.
        self._pose_received = False
        self._imu_received  = False   # informational only — not required to fly
        self._estop_triggered = False

        # E-stop z_min only applies AFTER drone has been airborne at least once.
        # This prevents immediately killing motors when starting from the ground.
        self._has_been_airborne = False

        self._step_count = 0

        # Action smoothing — previous action for EMA filter
        self._prev_action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        # ── QoS for sensor topics ──
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── Subscribers ──
        self.create_subscription(PoseStamped, '/cf231/pose', self._pose_cb, sensor_qos)
        self.create_subscription(Imu, '/cf231/imu', self._imu_cb, sensor_qos)
        self.create_subscription(Float32, '/z_target', self._ztarget_cb, 10)

        # ── Publisher: FullState to crazyflie_server's native topic ──
        # crazyflie_server subscribes to this and forwards to drone via radio.
        self._cmd_pub = self.create_publisher(
            FullState, '/cf231/cmd_full_state', 10)

        # ── Control timer at ctrl_freq Hz ──
        self._dt = 1.0 / float(ctrl_freq)
        self.create_timer(self._dt, self._control_step)

        # ── Startup banner ──
        self.get_logger().info('=' * 60)
        self.get_logger().info('  SAC Policy Node — Crazyflie v2.1 Deployment')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Model       : {model_path}')
        self.get_logger().info(f'  MAX_RPM     : {self._max_rpm}')
        self.get_logger().info(f'  Ctrl freq   : {ctrl_freq} Hz (dt={self._dt:.6f}s)')
        self.get_logger().info(f'  z_target    : {self._z_target:.2f} m')
        self.get_logger().info(f'  FAULT_FACTOR: {self._fault_factor.tolist()}')
        self.get_logger().info(f'  action_alpha: {self._action_alpha}  (1.0=raw, 0.3=smooth)')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Waiting for /cf231/pose ...')
        self.get_logger().info('(IMU optional — will use wx=wy=wz=0 if not available)')

    # ─────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        """
        Receive position + orientation from Lighthouse v2 via crazyflie_server.
        Topic: /cf231/pose  (geometry_msgs/PoseStamped, ~10 Hz)
        Also computes velocity via finite-difference + low-pass filter.
        """
        now = self.get_clock().now()
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        new_z = msg.pose.position.z

        # Finite-difference velocity
        if self._prev_x is not None and self._prev_pose_time is not None:
            dt_pose = (now - self._prev_pose_time).nanoseconds * 1e-9
            if dt_pose > 1e-6:
                a = self._vel_alpha
                self._vx = a * (new_x - self._prev_x) / dt_pose + (1-a) * self._vx
                self._vy = a * (new_y - self._prev_y) / dt_pose + (1-a) * self._vy
                self._vz = a * (new_z - self._prev_z) / dt_pose + (1-a) * self._vz

        self._prev_x = new_x; self._prev_y = new_y; self._prev_z = new_z
        self._prev_pose_time = now
        self._x = new_x; self._y = new_y; self._z = new_z

        # Orientation from quaternion
        q = msg.pose.orientation
        self._roll, self._pitch, self._yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)

        # Track airborne state for e-stop
        if self._z > AIRBORNE_THRESH:
            self._has_been_airborne = True

        if not self._pose_received:
            self._pose_received = True
            self.get_logger().info(
                f'✓ First pose: pos=({self._x:.3f},{self._y:.3f},{self._z:.3f}) '
                f'rpy=({math.degrees(self._roll):.1f}°,'
                f'{math.degrees(self._pitch):.1f}°,'
                f'{math.degrees(self._yaw):.1f}°)')
            self.get_logger().info('▶  STARTING MOTORS — stand clear!')

    def _imu_cb(self, msg: Imu):
        """
        Receive angular velocity. Topic: /cf231/imu (optional).
        If this topic never arrives, wx=wy=wz=0 is used (safe fallback).
        """
        self._wx = msg.angular_velocity.x
        self._wy = msg.angular_velocity.y
        self._wz = msg.angular_velocity.z
        if not self._imu_received:
            self._imu_received = True
            self.get_logger().info(
                f'✓ IMU online: w=({self._wx:.3f},{self._wy:.3f},{self._wz:.3f}) rad/s')

    def _ztarget_cb(self, msg: Float32):
        """Receive new z_target. Resets integral errors on change."""
        new_t = float(msg.data)
        if new_t < 0.2 or new_t > 2.5:
            self.get_logger().warn(f'z_target {new_t:.2f} out of [0.2,2.5] — ignoring')
            return
        if abs(new_t - self._z_target) > 1e-4:
            self.get_logger().info(
                f'z_target: {self._z_target:.2f} → {new_t:.2f} (integrals RESET)')
            self._int_x = 0.0; self._int_y = 0.0; self._int_z = 0.0
        self._z_target = new_t

    # ─────────────────────────────────────────────────────────────────────
    # Emergency stop
    # ─────────────────────────────────────────────────────────────────────

    def _check_estop(self) -> bool:
        """
        SAFETY: Check e-stop conditions.

        Attitude limits always active:
            |roll|  > 45°
            |pitch| > 45°
            z > 2.8 m
            |x| > 2.0 m
            |y| > 2.0 m

        z_min (z < 0.05m) check is ONLY active after drone has been airborne.
        This prevents killing motors while drone is still on the ground.
        """
        reasons = []
        if abs(self._roll) > math.radians(ESTOP_ROLL_DEG):
            reasons.append(f'|roll|={math.degrees(self._roll):.1f}°>{ESTOP_ROLL_DEG}°')
        if abs(self._pitch) > math.radians(ESTOP_PITCH_DEG):
            reasons.append(f'|pitch|={math.degrees(self._pitch):.1f}°>{ESTOP_PITCH_DEG}°')
        if self._z > ESTOP_Z_MAX:
            reasons.append(f'z={self._z:.3f}m>{ESTOP_Z_MAX}m')
        if abs(self._x) > ESTOP_XY_MAX:
            reasons.append(f'|x|={abs(self._x):.3f}m>{ESTOP_XY_MAX}m')
        if abs(self._y) > ESTOP_XY_MAX:
            reasons.append(f'|y|={abs(self._y):.3f}m>{ESTOP_XY_MAX}m')
        # z_min only after airborne — don't kill motors on ground
        if self._has_been_airborne and self._z < ESTOP_Z_MIN:
            reasons.append(f'z={self._z:.3f}m<{ESTOP_Z_MIN}m (post-airborne crash)')

        if reasons:
            self.get_logger().error('!!! EMERGENCY STOP TRIGGERED !!!')
            for r in reasons:
                self.get_logger().error(f'  → {r}')
            self.get_logger().error('All motor thrusts → 0')
            return True
        return False

    def _publish_zero_thrust(self):
        """Send zero thrust — SAFE STATE: position setpoint at current position, zero velocity."""
        msg = FullState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(self._x)
        msg.pose.position.y = float(self._y)
        msg.pose.position.z = float(self._z)  # hold current altitude
        msg.pose.orientation.w = 1.0
        # Zero velocity and acceleration
        msg.twist.linear.x = 0.0; msg.twist.linear.y = 0.0; msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0; msg.twist.angular.y = 0.0; msg.twist.angular.z = 0.0
        msg.acc.x = 0.0; msg.acc.y = 0.0; msg.acc.z = 0.0
        self._cmd_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────────
    # Thrust conversion (SAFETY-CRITICAL)
    # ─────────────────────────────────────────────────────────────────────

    def _action_to_fullstate(self, action: np.ndarray) -> FullState:
        """
        Convert normalized SAC action [0,1]^4 → crazyflie_interfaces/FullState.

        THRUST CONVERSION (CRITICAL):
          per-motor: thrust_i = action_i^2 * 65535  (same as original spec)
          Apply FAULT_FACTOR per motor.
          collective thrust = sum of all 4 motor thrusts (range 0..4*65535)
          Scaled to uint16: collective_thrust = sum / 4  (mean, in [0,65535])

        Attitude setpoint = current orientation (hold level).
        Velocity/acc setpoint = zero.
        Mellinger controller handles stabilisation from there.
        """
        # Per-motor thrust [0, 65535] — used for logging and fault injection
        thrust_raw     = (action ** 2) * 65535.0
        thrust_clipped = np.clip(thrust_raw, 0.0, 65535.0)
        thrust_faulted = thrust_clipped * self._fault_factor
        per_motor      = np.clip(thrust_faulted, 0.0, 65535.0)

        # Build FullState message
        msg = FullState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        # Position setpoint = current z_target (hold x=0, y=0)
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = float(self._z_target)

        # Lock yaw to 0° — identity quaternion (w=1, x=y=z=0)
        # Feeding current yaw back as setpoint = zero error = no correction = spin.
        # Fixed yaw=0 tells Mellinger to actively hold heading north.
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        # Zero velocity and acceleration setpoints
        msg.twist.linear.x = 0.0; msg.twist.linear.y = 0.0; msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0; msg.twist.angular.y = 0.0; msg.twist.angular.z = 0.0
        msg.acc.x = 0.0; msg.acc.y = 0.0; msg.acc.z = 0.0

        return msg, per_motor

    # ─────────────────────────────────────────────────────────────────────
    # Main control loop — 48 Hz
    # ─────────────────────────────────────────────────────────────────────

    def _control_step(self):
        """Timer callback: build obs → predict → convert → publish."""

        # Only require pose. IMU is optional (defaults to 0).
        if not self._pose_received:
            self._publish_zero_thrust()
            return

        if self._estop_triggered:
            self._publish_zero_thrust()
            return
        if self._check_estop():
            self._estop_triggered = True
            self._publish_zero_thrust()
            return

        # ── Update integral errors ──
        # Use 0.5x dt to slow integral accumulation — prevents windup at hover
        int_dt = self._dt * 0.5
        self._int_x = float(np.clip(self._int_x + self._x * int_dt, -2.0, 2.0))
        self._int_y = float(np.clip(self._int_y + self._y * int_dt, -2.0, 2.0))
        self._int_z = float(np.clip(
            self._int_z + (self._z - self._z_target) * int_dt, -2.0, 2.0))

        # ── Build 16-dim observation (MUST match simulation order exactly) ──
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz,
        #  z_target, int_x, int_y, int_z]
        #
        # Velocity deadband: zero out tiny velocities (sensor noise at hover)
        VEL_DEADBAND = 0.05   # m/s — ignore noise below this
        vx = self._vx if abs(self._vx) > VEL_DEADBAND else 0.0
        vy = self._vy if abs(self._vy) > VEL_DEADBAND else 0.0
        vz = self._vz if abs(self._vz) > VEL_DEADBAND else 0.0
        # Clip velocity to ±1.5 m/s — prevents noisy finite-diff spikes
        vx = float(np.clip(vx, -1.5, 1.5))
        vy = float(np.clip(vy, -1.5, 1.5))
        vz = float(np.clip(vz, -1.5, 1.5))

        obs = np.array([
            self._x,    self._y,    self._z,
            vx,         vy,         vz,
            self._roll, self._pitch, self._yaw,
            self._wx,   self._wy,   self._wz,
            self._z_target,
            self._int_x, self._int_y, self._int_z,
        ], dtype=np.float32)
        obs = np.clip(obs, OBS_LOW, OBS_HIGH)

        # ── SAC model prediction ──
        action, _ = self._model.predict(obs, deterministic=True)
        action = np.clip(action, 0.0, 1.0)

        # ── Action smoothing (EMA) — reduces oscillations from sim-to-real gap ──
        # smoothed = alpha * new + (1-alpha) * prev
        action = self._action_alpha * action + (1.0 - self._action_alpha) * self._prev_action
        action = np.clip(action, 0.0, 1.0)
        self._prev_action = action.copy()

        # ── Convert to FullState & publish ──
        cmd_msg, per_motor = self._action_to_fullstate(action)
        self._cmd_pub.publish(cmd_msg)

        # ── Periodic logging every 10 steps ──
        self._step_count += 1
        if self._step_count % 10 == 0:
            hover_err = math.sqrt(
                self._x**2 + self._y**2 + (self._z - self._z_target)**2)
            self.get_logger().info(
                f'[step {self._step_count:>6}] '
                f'pos=({self._x:+.3f},{self._y:+.3f},{self._z:.3f}) '
                f'vel=({self._vx:+.2f},{self._vy:+.2f},{self._vz:+.2f}) '
                f'z_tgt={self._z_target:.2f} err={hover_err:.4f}m '
                f'per_motor={[int(v) for v in per_motor]} '
                f'rpy=({math.degrees(self._roll):.1f}°,'
                f'{math.degrees(self._pitch):.1f}°,'
                f'{math.degrees(self._yaw):.1f}°)')


def main(args=None):
    rclpy.init(args=args)
    node = SACPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_zero_thrust()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
