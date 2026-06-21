#!/usr/bin/env python3
"""
hw_logger.py — Hardware Telemetry Logger for Crazyflie v2.1
============================================================
Run this as Terminal 4 alongside your control node. It subscribes to
/cf231/pose and /z_target and logs telemetry to a timestamped CSV file
inside hardware_exp/logs/.

Each row records:
    time_s, x, y, z, z_target, hover_error, roll_deg, pitch_deg, yaw_deg

Usage (Terminal 4):
    cd ~/Fault_Tolerant
    source ~/crazyflie_ws/install/setup.bash

    # No fault, SAC-DR controller:
    python3 hardware_exp/hw_logger.py --controller sac_dr --fault 1.0_1.0_1.0_1.0

    # 30% motor 2 fault, PID controller:
    python3 hardware_exp/hw_logger.py --controller pid --fault 1.0_1.0_0.7_1.0

    Press Ctrl+C to stop and save the CSV.

The CSV is saved automatically on Ctrl+C.
Author: Ayush   Date: 2026-06-21
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32


def quaternion_to_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = (math.copysign(math.pi / 2.0, sinp)
             if abs(sinp) >= 1.0 else math.asin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class HWLoggerNode(Node):
    def __init__(self, log_path: str, controller: str, fault: str):
        super().__init__('hw_logger')

        self._log_path   = log_path
        self._controller = controller
        self._fault      = fault
        self._z_target   = 1.0   # default until topic arrives
        self._start_time = None
        self._rows       = []

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(PoseStamped, '/cf231/pose',
                                 self._pose_cb, sensor_qos)
        self.create_subscription(Float32, '/z_target',
                                 self._ztarget_cb, 10)

        self.get_logger().info('=' * 55)
        self.get_logger().info('  Hardware Telemetry Logger')
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'  Controller : {controller}')
        self.get_logger().info(f'  Fault      : {fault}')
        self.get_logger().info(f'  Output     : {log_path}')
        self.get_logger().info('=' * 55)
        self.get_logger().info('Waiting for /cf231/pose ... Press Ctrl+C to stop & save.')

    def _ztarget_cb(self, msg: Float32):
        self._z_target = float(msg.data)

    def _pose_cb(self, msg: PoseStamped):
        now = time.time()
        if self._start_time is None:
            self._start_time = now
            self.get_logger().info('✓ Logging started.')

        t = now - self._start_time
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        q = msg.pose.orientation
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)

        hover_err = math.sqrt(x**2 + y**2 + (z - self._z_target)**2)

        self._rows.append({
            'time_s':      round(t, 4),
            'x':           round(x, 5),
            'y':           round(y, 5),
            'z':           round(z, 5),
            'z_target':    round(self._z_target, 3),
            'hover_error': round(hover_err, 5),
            'roll_deg':    round(math.degrees(roll), 3),
            'pitch_deg':   round(math.degrees(pitch), 3),
            'yaw_deg':     round(math.degrees(yaw), 3),
            'controller':  self._controller,
            'fault':       self._fault,
        })

        # Live print every ~50 rows (~5 s at 10 Hz)
        if len(self._rows) % 50 == 0:
            self.get_logger().info(
                f't={t:.1f}s  z={z:.3f}  z_tgt={self._z_target:.2f}'
                f'  err={hover_err:.4f}m  rows={len(self._rows)}')

    def save_csv(self):
        if not self._rows:
            self.get_logger().warn('No data recorded — CSV not saved.')
            return

        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        fieldnames = ['time_s', 'x', 'y', 'z', 'z_target', 'hover_error',
                      'roll_deg', 'pitch_deg', 'yaw_deg', 'controller', 'fault']
        with open(self._log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        n = len(self._rows)
        import numpy as np
        errors = [r['hover_error'] for r in self._rows]
        self.get_logger().info('=' * 55)
        self.get_logger().info(f'  Saved {n} rows → {self._log_path}')
        self.get_logger().info(f'  Mean hover error : {np.mean(errors):.4f} m')
        self.get_logger().info(f'  Std  hover error : {np.std(errors):.4f} m')
        self.get_logger().info(f'  Max  hover error : {np.max(errors):.4f} m')
        self.get_logger().info('=' * 55)


def main():
    parser = argparse.ArgumentParser(description='Hardware telemetry logger')
    parser.add_argument('--controller', default='sac_dr',
                        choices=['sac_dr', 'sac_nominal', 'pid'],
                        help='Controller being tested')
    parser.add_argument('--fault', default='1.0_1.0_1.0_1.0',
                        help='Fault factor string, e.g. 1.0_1.0_0.7_1.0')
    args = parser.parse_args()

    # Build output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f'{args.controller}_fault_{args.fault}_{timestamp}.csv'
    log_path  = os.path.join(log_dir, filename)

    rclpy.init()
    node = HWLoggerNode(log_path, args.controller, args.fault)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_csv()
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
