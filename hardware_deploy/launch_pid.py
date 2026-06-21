#!/usr/bin/env python3
"""
launch_pid.py — ROS2 Launch File for PID Hover Node
====================================================
Launches pid_node.py for comparison against the SAC RL policy.

Usage:
    ros2 launch launch_pid.py                              # no fault
    ros2 launch launch_pid.py fault_factor_str:="[1.0,1.0,0.7,1.0]"   # 30% loss motor 2
    ros2 launch launch_pid.py fault_factor_str:="[0.6,0.6,0.6,0.6]"   # 40% loss all

Author: Ayush   Date: 2026-06-21
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ctrl_freq_arg = DeclareLaunchArgument(
        'ctrl_freq', default_value='48',
        description='Control loop frequency Hz')

    z_target_default_arg = DeclareLaunchArgument(
        'z_target_default', default_value='1.0',
        description='Initial hover height [0.2-2.5 m]')

    # fault_factor passed as a string because ROS2 launch can't pass arrays
    # directly to ExecuteProcess. It is parsed in pid_node.py via ros-args.
    fault_factor_arg = DeclareLaunchArgument(
        'fault_factor_str', default_value='[1.0, 1.0, 1.0, 1.0]',
        description='Per-motor fault factor. E.g. [1.0,1.0,0.7,1.0] = 30% loss on motor 2')

    kp_z_arg  = DeclareLaunchArgument('kp_z',  default_value='2.0')
    ki_z_arg  = DeclareLaunchArgument('ki_z',  default_value='0.5')
    kd_z_arg  = DeclareLaunchArgument('kd_z',  default_value='1.0')
    kp_xy_arg = DeclareLaunchArgument('kp_xy', default_value='0.8')

    import os
    from launch.actions import ExecuteProcess

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pid_node_path = os.path.join(script_dir, 'pid_node.py')

    pid_node = ExecuteProcess(
        cmd=[
            'python3', pid_node_path,
            '--ros-args',
            '-p', ['ctrl_freq:=',       LaunchConfiguration('ctrl_freq')],
            '-p', 'fault_factor:=[1.0, 1.0, 1.0, 1.0]',  # edit here for faults
            '-p', ['z_target_default:=', LaunchConfiguration('z_target_default')],
            '-p', ['kp_z:=',            LaunchConfiguration('kp_z')],
            '-p', ['ki_z:=',            LaunchConfiguration('ki_z')],
            '-p', ['kd_z:=',            LaunchConfiguration('kd_z')],
            '-p', ['kp_xy:=',           LaunchConfiguration('kp_xy')],
        ],
        output='screen',
    )

    return LaunchDescription([
        ctrl_freq_arg,
        z_target_default_arg,
        fault_factor_arg,
        kp_z_arg, ki_z_arg, kd_z_arg, kp_xy_arg,
        pid_node,
    ])
