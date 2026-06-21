#!/usr/bin/env python3
"""
launch_policy.py — ROS2 Launch File for SAC Policy Node
========================================================
Launches the SACPolicyNode with configurable parameters for
deploying the trained SAC hover policy on a Crazyflie v2.1.

Parameters:
    model_path       : Path to trained SAC model (without .zip)
    max_rpm          : Maximum RPM from simulation (21702.5)
    ctrl_freq        : Control loop frequency in Hz (48)
    fault_factor     : Per-motor thrust multiplier [1.0, 1.0, 1.0, 1.0]
    z_target_default : Initial target hover height in meters (1.0)

Usage:
    ros2 launch launch_policy.py
    ros2 launch launch_policy.py model_path:=path/to/model
    ros2 launch launch_policy.py fault_factor:=[1.0,1.0,0.7,1.0]

Author: Ayush   Date: 2026-06-21
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Declare launch arguments ──
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/ayush/Fault_Tolerant/logs_hover_sac_dr/best/best_model',
        description='Path to trained SAC model (.zip, without extension)'
    )

    max_rpm_arg = DeclareLaunchArgument(
        'max_rpm',
        default_value='21702.5',
        description='Maximum RPM from simulation (must match training env)'
    )

    ctrl_freq_arg = DeclareLaunchArgument(
        'ctrl_freq',
        default_value='48',
        description='Control loop frequency in Hz (must match training env)'
    )

    z_target_default_arg = DeclareLaunchArgument(
        'z_target_default',
        default_value='1.0',
        description='Initial target hover height in meters [0.2 - 2.5]'
    )

    import os
    from launch.actions import ExecuteProcess
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_node_path = os.path.join(script_dir, 'policy_node.py')

    action_alpha_arg = DeclareLaunchArgument(
        'action_alpha', default_value='0.6',
        description='Action smoothing EMA: 1.0=no smoothing, 0.3=heavy (use for nominal SAC)')

    # ── Policy node ──
    policy_node = ExecuteProcess(
        cmd=[
            'python3',
            policy_node_path,
            '--ros-args',
            '-p', ['model_path:=', LaunchConfiguration('model_path')],
            '-p', ['max_rpm:=', LaunchConfiguration('max_rpm')],
            '-p', ['ctrl_freq:=', LaunchConfiguration('ctrl_freq')],
            '-p', 'fault_factor:=[1.0, 1.0, 1.0, 1.0]',
            '-p', ['z_target_default:=', LaunchConfiguration('z_target_default')],
            '-p', ['action_alpha:=', LaunchConfiguration('action_alpha')],
        ],
        cwd='/home/ayush/Fault_Tolerant',
        output='screen',
    )

    return LaunchDescription([
        model_path_arg,
        max_rpm_arg,
        ctrl_freq_arg,
        z_target_default_arg,
        action_alpha_arg,
        policy_node,
    ])
