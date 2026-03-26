"""
field_nav full-stack launch file.

Usage:
    ros2 launch field_nav field_nav.launch.py
    ros2 launch field_nav field_nav.launch.py sim_mode:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("field_nav")
    config  = os.path.join(pkg_dir, "config", "params.yaml")

    sim_mode = LaunchConfiguration("sim_mode", default="false")

    return LaunchDescription([
        DeclareLaunchArgument("sim_mode", default_value="false",
                              description="Use simulated sensors"),

        Node(
            package="field_nav",
            executable="ekf_localizer",
            name="ekf_localizer",
            parameters=[config],
            output="screen",
            remappings=[("/odom", "/robot/odom"),
                        ("/gps/fix", "/robot/gps/fix")],
        ),

        Node(
            package="field_nav",
            executable="crop_row_detector",
            name="crop_row_detector",
            parameters=[config],
            output="screen",
            remappings=[("/camera/image_raw", "/robot/camera/image_raw"),
                        ("/camera/info",      "/robot/camera/info")],
        ),

        Node(
            package="field_nav",
            executable="row_following_planner",
            name="row_following_planner",
            parameters=[config],
            output="screen",
        ),
    ])
