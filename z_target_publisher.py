#!/usr/bin/env python3
"""
z_target_publisher.py — Interactive Target Height Publisher
============================================================
Simple CLI tool that lets the operator set the target hover height
for the Crazyflie drone in real-time by publishing to /z_target.

The target height is validated to be within [0.2, 2.5] meters before
publishing.  The script loops, asking for a new height each time.

NOTE: When z_target changes, the policy_node.py automatically resets
its integral error terms to prevent control transients from the
previous setpoint.

Usage:
    python3 z_target_publisher.py

Author: Ayush   Date: 2026-06-21
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# ── Valid target height range (meters) ──
Z_TARGET_MIN = 0.2
Z_TARGET_MAX = 2.5


class ZTargetPublisher(Node):
    """Minimal ROS2 node that publishes target height to /z_target."""

    def __init__(self):
        super().__init__('z_target_publisher')
        self._pub = self.create_publisher(Float32, '/z_target', 10)
        self.get_logger().info('z_target publisher ready.')

    def publish_target(self, z: float):
        """Publish a validated z_target value."""
        msg = Float32()
        msg.data = z
        self._pub.publish(msg)
        self.get_logger().info(f'Published z_target = {z:.2f} m')


def main():
    rclpy.init()
    node = ZTargetPublisher()

    print()
    print('=' * 50)
    print('  Crazyflie Target Height Publisher')
    print(f'  Valid range: [{Z_TARGET_MIN} — {Z_TARGET_MAX}] meters')
    print('  Type "q" or Ctrl+C to quit')
    print('=' * 50)
    print()

    try:
        while True:
            try:
                raw = input(f'  Enter target height [{Z_TARGET_MIN}–{Z_TARGET_MAX} m]: ')
            except EOFError:
                break

            raw = raw.strip()
            if raw.lower() in ('q', 'quit', 'exit'):
                break

            try:
                val = float(raw)
            except ValueError:
                print(f'  ⚠  Invalid input "{raw}". Enter a number.\n')
                continue

            if val < Z_TARGET_MIN or val > Z_TARGET_MAX:
                print(f'  ⚠  {val:.2f} m is out of range '
                      f'[{Z_TARGET_MIN}, {Z_TARGET_MAX}]. Try again.\n')
                continue

            node.publish_target(val)
            print()

    except KeyboardInterrupt:
        print('\n  Exiting.')

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
