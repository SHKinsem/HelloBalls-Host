from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _world_to_court_tf(context):
    """Place the local odometry origin at the configured court start pose.

    The visual odometry (or optional external VIO) starts its local `world`
    frame at the vehicle's initial pose. The transform below makes that local
    origin a child of `court` when the fusion node is disabled.
    """
    fusion_enabled = LaunchConfiguration("use_fusion").perform(context)
    if fusion_enabled.lower() in ("1", "true", "yes", "on"):
        return []

    enabled = LaunchConfiguration("publish_world_to_court_tf").perform(context)
    legacy_enabled = LaunchConfiguration("publish_identity_world_to_court_tf").perform(context)
    if (
        enabled.lower() not in ("1", "true", "yes", "on") or
        legacy_enabled.lower() not in ("1", "true", "yes", "on")
    ):
        return []

    start_side = LaunchConfiguration("start_side").perform(context)
    court_length = float(LaunchConfiguration("court_length_m").perform(context))
    court_width = float(LaunchConfiguration("court_width_m").perform(context))

    half_length = court_length * 0.5
    half_doubles_width = court_width * 0.5
    poses = {
        "sideline_left": (
            -half_length,
            half_doubles_width,
            -1.5707963267948966,
        ),
        "sideline_right": (
            -half_length,
            -half_doubles_width,
            1.5707963267948966,
        ),
    }
    if start_side == "unknown":
        x, y, yaw = 0.0, 0.0, 0.0
    elif start_side in poses:
        x, y, yaw = poses[start_side]
    else:
        raise RuntimeError(
            "start_side must be unknown, sideline_left, or sideline_right"
        )

    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="court_to_vio_world_tf",
            arguments=[
                "--x", str(x),
                "--y", str(y),
                "--z", "0",
                "--roll", "0",
                "--pitch", "0",
                "--yaw", str(yaw),
                "--frame-id", LaunchConfiguration("court_frame").perform(context),
                "--child-frame-id", LaunchConfiguration("world_frame").perform(context),
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_calibration_file = LaunchConfiguration("camera_calibration_file")
    vio_odom_topic = LaunchConfiguration("vio_odom_topic")
    visual_odom_topic = LaunchConfiguration("visual_odom_topic")
    use_vio_prediction = LaunchConfiguration("use_vio_prediction")
    wheel_odom_topic = LaunchConfiguration("wheel_odom_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    use_wheel_prediction = LaunchConfiguration("use_wheel_prediction")
    use_imu_yaw_prediction = LaunchConfiguration("use_imu_yaw_prediction")
    imu_auto_bias = LaunchConfiguration("imu_auto_bias")
    imu_bias_calibration_s = LaunchConfiguration("imu_bias_calibration_s")
    imu_yaw_bias_rad_s = LaunchConfiguration("imu_yaw_bias_rad_s")
    imu_yaw_sign = LaunchConfiguration("imu_yaw_sign")
    world_frame = LaunchConfiguration("world_frame")
    base_frame = LaunchConfiguration("base_frame")
    court_frame = LaunchConfiguration("court_frame")
    start_side = LaunchConfiguration("start_side")
    camera_height_m = LaunchConfiguration("camera_height_m")
    camera_pitch_rad = LaunchConfiguration("camera_pitch_rad")
    camera_offset_x_m = LaunchConfiguration("camera_offset_x_m")
    camera_offset_y_m = LaunchConfiguration("camera_offset_y_m")
    flow_roi_start_fraction = LaunchConfiguration("flow_roi_start_fraction")
    flow_max_ground_range_m = LaunchConfiguration("flow_max_ground_range_m")
    enforce_nonholonomic_motion = LaunchConfiguration("enforce_nonholonomic_motion")
    court_length_m = LaunchConfiguration("court_length_m")
    court_width_m = LaunchConfiguration("court_width_m")
    singles_width_m = LaunchConfiguration("singles_width_m")
    service_line_distance_from_net_m = LaunchConfiguration("service_line_distance_from_net_m")
    match_doubles_sidelines = LaunchConfiguration("match_doubles_sidelines")
    court_pose_update_rate_hz = LaunchConfiguration("court_pose_update_rate_hz")
    full_map_min_vio_translation_m = LaunchConfiguration("full_map_min_vio_translation_m")
    full_map_min_vio_rotation_rad = LaunchConfiguration("full_map_min_vio_rotation_rad")
    use_fusion = LaunchConfiguration("use_fusion")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    rmw_implementation = LaunchConfiguration("rmw_implementation")
    fusion_output_topic = LaunchConfiguration("fusion_output_topic")
    fusion_publish_tf = LaunchConfiguration("fusion_publish_tf")
    fusion_confirmation_count = LaunchConfiguration("fusion_confirmation_count")
    fusion_max_sync_error_s = LaunchConfiguration("fusion_max_sync_error_s")
    fusion_correction_gain = LaunchConfiguration("fusion_correction_gain")
    fusion_max_correction_jump_m = LaunchConfiguration("fusion_max_correction_jump_m")
    fusion_max_correction_jump_yaw_rad = LaunchConfiguration(
        "fusion_max_correction_jump_yaw_rad"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rmw_implementation",
                default_value="rmw_fastrtps_cpp",
                description="ROS 2 middleware used by localization and RViz.",
            ),
            SetEnvironmentVariable(
                name="RMW_IMPLEMENTATION",
                value=rmw_implementation,
            ),
            DeclareLaunchArgument("image_topic", default_value="/camera/image_mono"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument(
                "camera_calibration_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("helloballs_bringup"),
                        "config",
                        "helloballs_camera_pinhole.yaml",
                    ]
                ),
                description="Fallback intrinsics when CameraInfo is absent or invalid.",
            ),
            DeclareLaunchArgument("vio_odom_topic", default_value="/vio/odometry"),
            DeclareLaunchArgument(
                "visual_odom_topic",
                default_value="/court/visual_odometry",
                description="Planar frame-to-frame white-line odometry output.",
            ),
            DeclareLaunchArgument(
                "use_vio_prediction",
                default_value="false",
                description="Use external VIO to predict court-map search candidates.",
            ),
            DeclareLaunchArgument("wheel_odom_topic", default_value="/wheel/odom"),
            DeclareLaunchArgument("imu_topic", default_value="/imu/data_raw"),
            DeclareLaunchArgument(
                "use_wheel_prediction",
                default_value="true",
                description="Use wheel odometry as a frame-motion prior/fallback when available.",
            ),
            DeclareLaunchArgument(
                "use_imu_yaw_prediction",
                default_value="true",
                description="Use bias-corrected IMU yaw as the frame rotation constraint.",
            ),
            DeclareLaunchArgument("imu_auto_bias", default_value="true"),
            DeclareLaunchArgument("imu_bias_calibration_s", default_value="2.0"),
            DeclareLaunchArgument("imu_yaw_bias_rad_s", default_value="0.0"),
            DeclareLaunchArgument("imu_yaw_sign", default_value="1.0"),
            DeclareLaunchArgument("world_frame", default_value="world"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("court_frame", default_value="court"),
            DeclareLaunchArgument(
                "use_fusion",
                default_value="true",
                description=(
                    "Start court_vio_fusion_node. When enabled, it is the sole "
                    "publisher of court->world."
                ),
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Open the top-down court localization RViz view.",
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
            DeclareLaunchArgument(
                "fusion_output_topic",
                default_value="/localization/odometry",
            ),
            DeclareLaunchArgument("fusion_publish_tf", default_value="true"),
            DeclareLaunchArgument("fusion_confirmation_count", default_value="3"),
            DeclareLaunchArgument("fusion_max_sync_error_s", default_value="0.15"),
            DeclareLaunchArgument("fusion_correction_gain", default_value="0.2"),
            DeclareLaunchArgument("fusion_max_correction_jump_m", default_value="1.0"),
            DeclareLaunchArgument(
                "fusion_max_correction_jump_yaw_rad",
                default_value="0.35",
            ),
            DeclareLaunchArgument("publish_world_to_court_tf", default_value="true"),
            # Retain the previous argument as a disable-only compatibility alias.
            DeclareLaunchArgument("publish_identity_world_to_court_tf", default_value="true"),
            DeclareLaunchArgument(
                "start_side",
                default_value="unknown",
                description=(
                    "Initial hypothesis: unknown, sideline_left, or sideline_right. "
                    "The vehicle heading is parallel to the short baseline."
                ),
            ),
            DeclareLaunchArgument("camera_height_m", default_value="0.14214"),
            DeclareLaunchArgument("camera_pitch_rad", default_value="0.0"),
            DeclareLaunchArgument("camera_offset_x_m", default_value="0.23668"),
            DeclareLaunchArgument("camera_offset_y_m", default_value="0.0"),
            DeclareLaunchArgument("flow_roi_start_fraction", default_value="0.60"),
            DeclareLaunchArgument("flow_max_ground_range_m", default_value="4.0"),
            DeclareLaunchArgument("enforce_nonholonomic_motion", default_value="true"),
            DeclareLaunchArgument("court_length_m", default_value="23.77"),
            DeclareLaunchArgument("court_width_m", default_value="10.97"),
            DeclareLaunchArgument("singles_width_m", default_value="8.23"),
            DeclareLaunchArgument(
                "match_doubles_sidelines",
                default_value="true",
                description=(
                    "Include outer doubles sidelines after switching from the "
                    "startup map to the full court map."
                ),
            ),
            DeclareLaunchArgument("court_pose_update_rate_hz", default_value="4.0"),
            DeclareLaunchArgument("full_map_min_vio_translation_m", default_value="0.5"),
            DeclareLaunchArgument("full_map_min_vio_rotation_rad", default_value="0.35"),
            DeclareLaunchArgument("service_line_distance_from_net_m", default_value="6.40"),
            Node(
                package="helloballs_bringup",
                executable="court_line_localizer_node",
                name="court_line_localizer",
                output="screen",
                parameters=[
                    {
                        "image_topic": image_topic,
                        "camera_info_topic": camera_info_topic,
                        "camera_calibration_file": camera_calibration_file,
                        "vio_odom_topic": vio_odom_topic,
                        "visual_odom_topic": visual_odom_topic,
                        "visual_odom_frame": world_frame,
                        "use_vio_prediction": ParameterValue(
                            use_vio_prediction,
                            value_type=bool,
                        ),
                        "wheel_odom_topic": wheel_odom_topic,
                        "imu_topic": imu_topic,
                        "use_wheel_prediction": ParameterValue(
                            use_wheel_prediction,
                            value_type=bool,
                        ),
                        "use_imu_yaw_prediction": ParameterValue(
                            use_imu_yaw_prediction,
                            value_type=bool,
                        ),
                        "imu_auto_bias": ParameterValue(imu_auto_bias, value_type=bool),
                        "imu_bias_calibration_s": ParameterValue(
                            imu_bias_calibration_s,
                            value_type=float,
                        ),
                        "imu_yaw_bias_rad_s": ParameterValue(
                            imu_yaw_bias_rad_s,
                            value_type=float,
                        ),
                        "imu_yaw_sign": ParameterValue(imu_yaw_sign, value_type=float),
                        "base_frame": base_frame,
                        "court_frame": court_frame,
                        "start_side": start_side,
                        "camera_height_m": ParameterValue(camera_height_m, value_type=float),
                        "camera_pitch_rad": ParameterValue(camera_pitch_rad, value_type=float),
                        "camera_offset_x_m": ParameterValue(
                            camera_offset_x_m,
                            value_type=float,
                        ),
                        "camera_offset_y_m": ParameterValue(
                            camera_offset_y_m,
                            value_type=float,
                        ),
                        "flow_roi_start_fraction": ParameterValue(
                            flow_roi_start_fraction,
                            value_type=float,
                        ),
                        "flow_max_ground_range_m": ParameterValue(
                            flow_max_ground_range_m,
                            value_type=float,
                        ),
                        "enforce_nonholonomic_motion": ParameterValue(
                            enforce_nonholonomic_motion,
                            value_type=bool,
                        ),
                        "court_length_m": ParameterValue(court_length_m, value_type=float),
                        "court_width_m": ParameterValue(court_width_m, value_type=float),
                        "singles_width_m": ParameterValue(singles_width_m, value_type=float),
                        "match_doubles_sidelines": ParameterValue(
                            match_doubles_sidelines,
                            value_type=bool,
                        ),
                        "court_pose_update_rate_hz": ParameterValue(
                            court_pose_update_rate_hz,
                            value_type=float,
                        ),
                        "full_map_min_vio_translation_m": ParameterValue(
                            full_map_min_vio_translation_m,
                            value_type=float,
                        ),
                        "full_map_min_vio_rotation_rad": ParameterValue(
                            full_map_min_vio_rotation_rad,
                            value_type=float,
                        ),
                        "service_line_distance_from_net_m": ParameterValue(
                            service_line_distance_from_net_m,
                            value_type=float,
                        ),
                    }
                ],
            ),
            Node(
                package="helloballs_bringup",
                executable="court_vio_fusion_node",
                name="court_vio_fusion",
                output="screen",
                condition=IfCondition(use_fusion),
                parameters=[
                    {
                        "vio_topic": visual_odom_topic,
                        "court_pose_topic": "/court/pose_measurement",
                        "output_topic": fusion_output_topic,
                        "court_frame": court_frame,
                        "world_frame": world_frame,
                        "base_frame": base_frame,
                        "publish_tf": ParameterValue(fusion_publish_tf, value_type=bool),
                        "confirmation_count": ParameterValue(
                            fusion_confirmation_count,
                            value_type=int,
                        ),
                        "max_sync_error_s": ParameterValue(
                            fusion_max_sync_error_s,
                            value_type=float,
                        ),
                        "correction_gain": ParameterValue(
                            fusion_correction_gain,
                            value_type=float,
                        ),
                        "max_correction_jump_m": ParameterValue(
                            fusion_max_correction_jump_m,
                            value_type=float,
                        ),
                        "max_correction_jump_yaw_rad": ParameterValue(
                            fusion_max_correction_jump_yaw_rad,
                            value_type=float,
                        ),
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="court_localization_rviz",
                arguments=["-d", rviz_config_file],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
            OpaqueFunction(function=_world_to_court_tf),
        ]
    )
