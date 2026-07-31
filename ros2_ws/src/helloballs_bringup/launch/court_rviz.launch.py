from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    rmw_implementation = LaunchConfiguration("rmw_implementation")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rmw_implementation",
                default_value="rmw_fastrtps_cpp",
                description="ROS 2 middleware used by RViz.",
            ),
            SetEnvironmentVariable(
                name="RMW_IMPLEMENTATION",
                value=rmw_implementation,
            ),
            DeclareLaunchArgument(
                "rviz_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("helloballs_bringup"),
                        "config",
                        "court_remote.rviz",
                    ]
                ),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="court_remote_rviz",
                arguments=["-d", rviz_config_file],
                output="screen",
            ),
        ]
    )
