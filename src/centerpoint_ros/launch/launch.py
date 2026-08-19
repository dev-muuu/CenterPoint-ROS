#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# CenterPoint only, with RViz2. Topics, QoS and detection thresholds are
# configured at the top of scripts/ros.py.
#
# For the full pipeline with disturbance injection and detection logging,
# see launch_full.py - it needs packages that are not in this workspace.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('centerpoint_ros')
    rviz_config = os.path.join(pkg_dir, 'launch', 'centerpoint.rviz')

    use_rviz = LaunchConfiguration('use_rviz')

    centerpoint_node = Node(
        package='centerpoint_ros',
        executable='ros.py',
        name='centerpoint_ros_node',
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Start RViz2 alongside the detector'),
        centerpoint_node,
        rviz_node,
    ])
