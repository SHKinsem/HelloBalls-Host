from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    image_topic = LaunchConfiguration("image_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    vins_package = LaunchConfiguration("vins_package")
    estimator_executable = LaunchConfiguration("estimator_executable")

    static_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("helloballs_bringup"), "launch", "static_tf.launch.py"]
            )
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("helloballs_bringup"),
                        "config",
                        "vins_mono_imu.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("imu_topic", default_value="/imu/data_raw"),
            DeclareLaunchArgument("vins_package", default_value="vins"),
            DeclareLaunchArgument("estimator_executable", default_value="vins_node"),
            static_tf_launch,
            Node(
                package=vins_package,
                executable=estimator_executable,
                name="vins_estimator",
                output="screen",
                parameters=[{"config_file": config_file}],
                remappings=[
                    ("/camera/image_raw", image_topic),
                    ("/imu/data_raw", imu_topic),
                    ("odometry", "/vio/odometry"),
                    ("path", "/vio/path"),
                    ("imu_propagate", "/vio/imu_propagate"),
                ],
            ),
        ]
    )
