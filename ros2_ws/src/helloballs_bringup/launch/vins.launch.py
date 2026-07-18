from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
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
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config_file = LaunchConfiguration("rviz_config_file")

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
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument(
                "rviz_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("vins"),
                        "config",
                        "vins_rviz_config.rviz",
                    ]
                ),
            ),
            static_tf_launch,
            Node(
                package=vins_package,
                executable=estimator_executable,
                name="vins_estimator",
                output="screen",
                # The estimator keeps only a bounded image backlog, but an
                # occasional device or library failure should not leave VIO
                # permanently unavailable for the rest of a mission.
                respawn=True,
                respawn_delay=2.0,
                parameters=[{"config_file": config_file}],
                remappings=[
                    ("/camera/image_raw", image_topic),
                    ("/imu/data_raw", imu_topic),
                    ("odometry", "/vio/odometry"),
                    ("path", "/vio/path"),
                    ("imu_propagate", "/vio/imu_propagate"),
                    ("point_cloud", "/vio/point_cloud"),
                    ("margin_cloud", "/vio/margin_cloud"),
                    ("key_poses", "/vio/key_poses"),
                    ("camera_pose", "/vio/camera_pose"),
                    ("camera_pose_visual", "/vio/camera_pose_visual"),
                    ("keyframe_pose", "/vio/keyframe_pose"),
                    ("keyframe_point", "/vio/keyframe_point"),
                    ("extrinsic", "/vio/extrinsic"),
                    ("image_track", "/vio/image_track"),
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="vio_debug_rviz",
                arguments=["-d", rviz_config_file],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
