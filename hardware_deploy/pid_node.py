#!/usr/bin/env python3
"""
pid_node.py — ROS2 Classical PID Hover Controller for Crazyflie v2.1
=====================================================================
Runs a 3-axis PID controller on real hardware for comparison against
the SAC RL policy.  Same interface, same topics, same fault injection
mechanism as policy_node.py — enabling a fair apples-to-apples comparison.

CONTROL STRATEGY:
    Altitude (Z):   PID on z-error → vertical acceleration command
    Lateral (X,Y):  P control on position error → velocity setpoint
    Attitude:       Delegated to Mellinger via FullState position setpoint

FAULT INJECTION (identical to policy_node.py):
    FAULT_FACTOR = [1.0, 1.0, 1.0, 1.0]  → no fault
    FAULT_FACTOR = [1.0, 1.0, 0.7, 1.0]  → 30% thrust loss on motor 2
    Applied by scaling the z-acceleration command (collective thrust reduction).

Subscribes:
    /cf231/pose   → geometry_msgs/PoseStamped  (Lighthouse v2 position + orientation)
    /cf231/imu    → sensor_msgs/Imu            (angular rates, optional)
    /z_target     → std_msgs/Float32           (target hover height)

Publishes:
    /cf231/cmd_full_state → crazyflie_interfaces/FullState

Usage:
    python3 pid_node.py --ros-args -p fault_factor:=[1.0,1.0,0.7,1.0]
    ros2 launch launch_pid.py
    ros2 launch launch_pid.py fault_factor_str:="[1.0,1.0,0.7,1.0]"

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
from stable_baselines3 import SAC  # Not used — imported only to match env

# ── Control constants ──
CTRL_FREQ_DEFAULT = 48
GRAVITY = 9.81          # m/s^2

# ── PID gains (tuned for 48 Hz, matching simulation DSLPIDControl) ──
# Altitude (Z)
KP_Z  = 2.0    # Proportional
KI_Z  = 0.5    # Integral
KD_Z  = 1.0    # Derivative

# Horizontal (X, Y) — P-only position control
KP_XY = 0.8

# ── Safety thresholds (same as policy_node.py) ──
ESTOP_ROLL_DEG  = 45.0
ESTOP_PITCH_DEG = 45.0
ESTOP_Z_MIN     = 0.05
ESTOP_Z_MAX     = 2.8
ESTOP_XY_MAX    = 2.0
AIRBORNE_THRESH = 0.10

# ── Integral wind-up limits ──
INT_Z_MAX = 2.0     # m·s (matches simulation)
INT_XY_MAX = 2.0


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to Euler (roll, pitch, yaw) in radians."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class PIDHoverNode(Node):
    """
    ROS2 node: Classical PID hover controller for Crazyflie v2.1 at 48 Hz.
    Identical control interface to SACPolicyNode for fair comparison.
    """

    def __init__(self):
        super().__init__('pid_hover_node')

        # ── Declare & read parameters ──
        self.declare_parameter('ctrl_freq',  CTRL_FREQ_DEFAULT)
        self.declare_parameter('fault_factor', [1.0, 1.0, 1.0, 1.0])
        self.declare_parameter('z_target_default', 1.0)
        self.declare_parameter('kp_z',  KP_Z)
        self.declare_parameter('ki_z',  KI_Z)
        self.declare_parameter('kd_z',  KD_Z)
        self.declare_parameter('kp_xy', KP_XY)

        ctrl_freq    = self.get_parameter('ctrl_freq').get_parameter_value().integer_value
        fault_list   = self.get_parameter('fault_factor').get_parameter_value().double_array_value
        z_default    = self.get_parameter('z_target_default').get_parameter_value().double_value
        self._kp_z   = self.get_parameter('kp_z').get_parameter_value().double_value
        self._ki_z   = self.get_parameter('ki_z').get_parameter_value().double_value
        self._kd_z   = self.get_parameter('kd_z').get_parameter_value().double_value
        self._kp_xy  = self.get_parameter('kp_xy').get_parameter_value().double_value

        # FAULT_FACTOR: mean applied as collective thrust scale
        # [1.0, 1.0, 1.0, 1.0] = no fault
        # [1.0, 1.0, 0.7, 1.0] = 30% thrust loss on motor 2
        self._fault_factor = np.array(fault_list, dtype=np.float64)
        assert self._fault_factor.shape == (4,)
        # Mean of fault factors = collective thrust scale (0.0–1.0)
        self._thrust_scale = float(np.mean(self._fault_factor))

        # ── State variables ──
        self._z_target = z_default
        self._x = 0.0;  self._y = 0.0;  self._z = 0.0
        self._roll = 0.0; self._pitch = 0.0; self._yaw = 0.0
        self._wx = 0.0;  self._wy = 0.0;  self._wz = 0.0
        self._vx = 0.0;  self._vy = 0.0;  self._vz = 0.0
        self._prev_x = None; self._prev_y = 0.0; self._prev_z = 0.0
        self._prev_pose_time = None
        self._vel_alpha = 0.3

        # ── PID state ──
        self._err_z_prev = 0.0      # Previous z error (for derivative)
        self._int_z  = 0.0          # Z integral
        self._int_x  = 0.0          # X integral (for logging, unused in control)
        self._int_y  = 0.0

        # ── Safety state ──
        self._pose_received    = False
        self._imu_received     = False
        self._estop_triggered  = False
        self._has_been_airborne = False
        self._step_count       = 0

        # ── QoS ──
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── Subscribers ──
        self.create_subscription(PoseStamped, '/cf231/pose', self._pose_cb, sensor_qos)
        self.create_subscription(Imu, '/cf231/imu', self._imu_cb, sensor_qos)
        self.create_subscription(Float32, '/z_target', self._ztarget_cb, 10)

        # ── Publisher ──
        self._cmd_pub = self.create_publisher(
            FullState, '/cf231/cmd_full_state', 10)

        # ── Control timer ──
        self._dt = 1.0 / float(ctrl_freq)
        self.create_timer(self._dt, self._control_step)

        # ── Startup banner ──
        self.get_logger().info('=' * 60)
        self.get_logger().info('  PID Hover Node — Crazyflie v2.1 Hardware')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Controller  : Classical PID (no RL)')
        self.get_logger().info(f'  Ctrl freq   : {ctrl_freq} Hz (dt={self._dt:.6f}s)')
        self.get_logger().info(f'  z_target    : {self._z_target:.2f} m')
        self.get_logger().info(f'  FAULT_FACTOR: {self._fault_factor.tolist()}')
        self.get_logger().info(f'  Thrust scale: {self._thrust_scale:.3f}  (mean of fault factors)')
        self.get_logger().info(f'  PID gains   : Kp={self._kp_z} Ki={self._ki_z} Kd={self._kd_z}')
        self.get_logger().info(f'  XY Kp       : {self._kp_xy}')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Waiting for /cf231/pose ...')

    # ─────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _pose_cb(self, msg: PoseStamped):
        """Receive position + orientation. Compute velocity via finite-diff."""
        now = self.get_clock().now()
        new_x = msg.pose.position.x
        new_y = msg.pose.position.y
        new_z = msg.pose.position.z

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

        q = msg.pose.orientation
        self._roll, self._pitch, self._yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)

        if self._z > AIRBORNE_THRESH:
            self._has_been_airborne = True

        if not self._pose_received:
            self._pose_received = True
            self.get_logger().info(
                f'✓ First pose: pos=({self._x:.3f},{self._y:.3f},{self._z:.3f}) '
                f'rpy=({math.degrees(self._roll):.1f}°,'
                f'{math.degrees(self._pitch):.1f}°,'
                f'{math.degrees(self._yaw):.1f}°)')
            self.get_logger().info('▶  STARTING PID — stand clear!')

    def _imu_cb(self, msg: Imu):
        self._wx = msg.angular_velocity.x
        self._wy = msg.angular_velocity.y
        self._wz = msg.angular_velocity.z
        if not self._imu_received:
            self._imu_received = True
            self.get_logger().info('✓ IMU online.')

    def _ztarget_cb(self, msg: Float32):
        new_t = float(msg.data)
        if new_t < 0.2 or new_t > 2.5:
            self.get_logger().warn(f'z_target {new_t:.2f} out of [0.2,2.5] — ignoring')
            return
        if abs(new_t - self._z_target) > 1e-4:
            self.get_logger().info(
                f'z_target: {self._z_target:.2f} → {new_t:.2f} (PID integrals RESET)')
            # Reset PID integrator and derivative on setpoint change
            self._int_z = 0.0
            self._err_z_prev = 0.0
        self._z_target = new_t

    # ─────────────────────────────────────────────────────────────────────
    # Emergency stop
    # ─────────────────────────────────────────────────────────────────────

    def _check_estop(self) -> bool:
        reasons = []
        if abs(self._roll)  > math.radians(ESTOP_ROLL_DEG):
            reasons.append(f'|roll|={math.degrees(self._roll):.1f}°')
        if abs(self._pitch) > math.radians(ESTOP_PITCH_DEG):
            reasons.append(f'|pitch|={math.degrees(self._pitch):.1f}°')
        if self._z > ESTOP_Z_MAX:
            reasons.append(f'z={self._z:.3f}m>{ESTOP_Z_MAX}m')
        if abs(self._x) > ESTOP_XY_MAX:
            reasons.append(f'|x|={abs(self._x):.3f}m')
        if abs(self._y) > ESTOP_XY_MAX:
            reasons.append(f'|y|={abs(self._y):.3f}m')
        if self._has_been_airborne and self._z < ESTOP_Z_MIN:
            reasons.append(f'z={self._z:.3f}m<{ESTOP_Z_MIN}m (crash)')
        if reasons:
            self.get_logger().error('!!! EMERGENCY STOP !!!')
            for r in reasons:
                self.get_logger().error(f'  → {r}')
            return True
        return False

    def _publish_zero_cmd(self):
        """Hold current position — safe stop."""
        msg = FullState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(self._x)
        msg.pose.position.y = float(self._y)
        msg.pose.position.z = float(self._z)
        msg.pose.orientation.w = 1.0
        msg.twist.linear.x = 0.0; msg.twist.linear.y = 0.0; msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0; msg.twist.angular.y = 0.0; msg.twist.angular.z = 0.0
        msg.acc.x = 0.0; msg.acc.y = 0.0; msg.acc.z = 0.0
        self._cmd_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────────
    # PID computation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_pid(self) -> FullState:
        """
        Compute PID output and build FullState command.

        ALTITUDE (Z) — Full PID:
            err_z  = z_target - z
            pid_z  = Kp * err_z + Ki * ∫err_z dt + Kd * d(err_z)/dt
            Fault injection: pid_z *= thrust_scale (mean of FAULT_FACTOR)

        The PID output is used as a POSITION OFFSET added to z_target:
            cmd_z = z_target + clip(pid_z * dt, -0.15, 0.15)
        This keeps the same FullState format as SAC (position-only, no
        velocity/acc override) so Mellinger computes thrust correctly.
        """
        dt = self._dt

        # ── Z PID ──
        err_z = self._z_target - self._z
        self._int_z = float(np.clip(self._int_z + err_z * dt, -INT_Z_MAX, INT_Z_MAX))
        d_err_z = (err_z - self._err_z_prev) / dt
        self._err_z_prev = err_z

        pid_z = (self._kp_z  * err_z +
                 self._ki_z  * self._int_z +
                 self._kd_z  * d_err_z)

        # Fault injection: weaker motors → less effective climb
        pid_z *= self._thrust_scale

        # Convert PID output → z position offset, clamp to ±0.15 m per step
        z_offset = float(np.clip(pid_z * dt, -0.15, 0.15))
        cmd_z    = float(np.clip(self._z_target + z_offset, 0.05, ESTOP_Z_MAX))

        # XY: P-control → small position correction toward origin
        cmd_x = float(np.clip(-self._kp_xy * self._x * dt, -0.05, 0.05))
        cmd_y = float(np.clip(-self._kp_xy * self._y * dt, -0.05, 0.05))

        # ── Build FullState (position-only, same format as SAC) ──
        msg = FullState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        msg.pose.position.x = cmd_x
        msg.pose.position.y = cmd_y
        msg.pose.position.z = cmd_z

        # Yaw locked to 0° — identity quaternion
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        # Zero velocity/acc — Mellinger computes thrust internally
        msg.twist.linear.x = 0.0; msg.twist.linear.y = 0.0; msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0; msg.twist.angular.y = 0.0; msg.twist.angular.z = 0.0
        msg.acc.x = 0.0; msg.acc.y = 0.0; msg.acc.z = 0.0

        return msg

    # ─────────────────────────────────────────────────────────────────────
    # Control loop — 48 Hz
    # ─────────────────────────────────────────────────────────────────────

    def _control_step(self):
        if not self._pose_received:
            self._publish_zero_cmd()
            return

        if self._estop_triggered:
            self._publish_zero_cmd()
            return
        if self._check_estop():
            self._estop_triggered = True
            self._publish_zero_cmd()
            return

        cmd_msg = self._compute_pid()
        self._cmd_pub.publish(cmd_msg)

        self._step_count += 1
        if self._step_count % 10 == 0:
            err_z = self._z_target - self._z
            hover_err = math.sqrt(self._x**2 + self._y**2 + err_z**2)
            self.get_logger().info(
                f'[step {self._step_count:>6}] '
                f'pos=({self._x:+.3f},{self._y:+.3f},{self._z:.3f}) '
                f'z_tgt={self._z_target:.2f} err={hover_err:.4f}m '
                f'int_z={self._int_z:.3f} '
                f'thrust_scale={self._thrust_scale:.2f} '
                f'rpy=({math.degrees(self._roll):.1f}°,'
                f'{math.degrees(self._pitch):.1f}°,'
                f'{math.degrees(self._yaw):.1f}°)')


def main(args=None):
    rclpy.init(args=args)
    node = PIDHoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_zero_cmd()
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
