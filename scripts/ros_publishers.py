from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

from scripts.HelloBalls_Serial import ImuState

G_TO_MPS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0
RPM_TO_RAD_S = 2.0 * math.pi / 60.0
DEFAULT_WHEEL_GEAR_RATIO = 3591.0 / 187.0


@dataclass(frozen=True)
class RosPublisherConfig:
    node_name: str = "helloballs_sensor_publisher"
    imu_topic: str = "/imu/data_raw"
    imu_frame_id: str = "imu_link"
    # The current IMU reports a stationary acceleration norm of about
    # 10.80097 m/s^2. Correct it to standard gravity before VINS receives it.
    # A future six-position calibration can replace this uniform scale with
    # per-axis scale and bias parameters.
    imu_accel_scale: float = 0.907942
    imu_angular_velocity_variance: float = 0.0
    imu_linear_acceleration_variance: float = 0.0
    allow_legacy_imu_timestamps: bool = False
    wheel_odom_topic: str = "/wheel/odom"
    wheel_odom_frame_id: str = "wheel_odom"
    wheel_base_frame_id: str = "base_link"
    wheel_radius_m: float = 0.0
    wheel_track_m: float = 0.0
    wheel_gear_ratio: float = DEFAULT_WHEEL_GEAR_RATIO
    wheel1_sign: float = 1.0
    wheel2_sign: float = 1.0
    wheel_max_sample_gap_s: float = 0.25


@dataclass(frozen=True)
class WheelOdometryState:
    x_m: float
    y_m: float
    yaw_rad: float
    linear_m_s: float
    angular_rad_s: float


class DifferentialDriveOdometry:
    """Integrate two wheel RPM values with differential-drive kinematics."""

    def __init__(
        self,
        wheel_radius_m: float,
        wheel_track_m: float,
        wheel_gear_ratio: float = 1.0,
        wheel1_sign: float = 1.0,
        wheel2_sign: float = 1.0,
        max_sample_gap_s: float = 0.25,
    ) -> None:
        if not math.isfinite(wheel_radius_m) or wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be a positive finite value")
        if not math.isfinite(wheel_track_m) or wheel_track_m <= 0.0:
            raise ValueError("wheel_track_m must be a positive finite value")
        if not math.isfinite(wheel_gear_ratio) or wheel_gear_ratio <= 0.0:
            raise ValueError("wheel_gear_ratio must be a positive finite value")
        if wheel1_sign not in (-1.0, 1.0) or wheel2_sign not in (-1.0, 1.0):
            raise ValueError("wheel signs must be either -1 or 1")
        if not math.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0.0:
            raise ValueError("wheel_max_sample_gap_s must be a positive finite value")
        self.wheel_radius_m = wheel_radius_m
        self.wheel_track_m = wheel_track_m
        self.wheel_gear_ratio = wheel_gear_ratio
        self.wheel1_sign = wheel1_sign
        self.wheel2_sign = wheel2_sign
        self.max_sample_gap_s = max_sample_gap_s
        self.x_m = 0.0
        self.y_m = 0.0
        self.yaw_rad = 0.0
        self._last_timestamp: float | None = None

    def update(self, wheel1_rpm: float, wheel2_rpm: float, timestamp: float) -> WheelOdometryState:
        values = (wheel1_rpm, wheel2_rpm, timestamp)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("wheel RPM and timestamp must be finite")

        speed_factor = RPM_TO_RAD_S * self.wheel_radius_m / self.wheel_gear_ratio
        wheel1_m_s = self.wheel1_sign * wheel1_rpm * speed_factor
        wheel2_m_s = self.wheel2_sign * wheel2_rpm * speed_factor
        linear_m_s = 0.5 * (wheel1_m_s + wheel2_m_s)
        angular_rad_s = (wheel2_m_s - wheel1_m_s) / self.wheel_track_m

        if self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt <= 0.0:
                raise ValueError("wheel timestamp must be strictly increasing")
            if dt <= self.max_sample_gap_s:
                yaw_delta = angular_rad_s * dt
                midpoint_yaw = self.yaw_rad + 0.5 * yaw_delta
                distance = linear_m_s * dt
                self.x_m += distance * math.cos(midpoint_yaw)
                self.y_m += distance * math.sin(midpoint_yaw)
                self.yaw_rad = math.atan2(
                    math.sin(self.yaw_rad + yaw_delta),
                    math.cos(self.yaw_rad + yaw_delta),
                )
        self._last_timestamp = timestamp
        return WheelOdometryState(
            self.x_m,
            self.y_m,
            self.yaw_rad,
            linear_m_s,
            angular_rad_s,
        )


class McuClockSynchronizer:
    """Map an MCU monotonic sampling clock onto the host Unix time axis."""

    def __init__(self) -> None:
        self._anchor_sample_time_us: int | None = None
        self._anchor_host_time: float | None = None
        self._last_sample_time_us: int | None = None
        self._last_sequence: int | None = None
        self._last_host_time: float | None = None

    def to_host_time(
        self,
        sample_time_us: int,
        sample_sequence: int,
        received_at: float,
    ) -> float:
        if sample_time_us < 0:
            raise ValueError("MCU sample_time_us must be non-negative")
        if not 0 <= sample_sequence <= 0xFFFFFFFF:
            raise ValueError("MCU sample_sequence must fit uint32")
        if not math.isfinite(received_at) or received_at <= 0.0:
            raise ValueError("Host receive time must be a positive finite Unix time")

        if self._anchor_sample_time_us is None:
            self._reset_anchor(sample_time_us, received_at)
        else:
            assert self._last_sample_time_us is not None
            assert self._last_sequence is not None
            timestamp_regressed = sample_time_us <= self._last_sample_time_us
            sequence_delta = (sample_sequence - self._last_sequence) & 0xFFFFFFFF
            sequence_regressed = sequence_delta >= 0x80000000

            if timestamp_regressed and sequence_regressed:
                # Both MCU counters moving backwards indicates an MCU reboot.
                anchor_host_time = received_at
                if self._last_host_time is not None:
                    anchor_host_time = max(anchor_host_time, self._last_host_time + 1e-6)
                self._reset_anchor(sample_time_us, anchor_host_time)
            elif timestamp_regressed:
                raise ValueError("duplicate or out-of-order MCU sample timestamp")
            elif sequence_delta == 0 or sequence_regressed:
                raise ValueError("duplicate or out-of-order MCU sample sequence")

        assert self._anchor_sample_time_us is not None
        assert self._anchor_host_time is not None
        host_time = self._anchor_host_time + (
            sample_time_us - self._anchor_sample_time_us
        ) / 1_000_000.0
        if self._last_host_time is not None and host_time <= self._last_host_time:
            raise ValueError("reconstructed IMU timestamp is not strictly increasing")

        self._last_sample_time_us = sample_time_us
        self._last_sequence = sample_sequence
        self._last_host_time = host_time
        return host_time

    def _reset_anchor(self, sample_time_us: int, host_time: float) -> None:
        self._anchor_sample_time_us = sample_time_us
        self._anchor_host_time = host_time
        self._last_sample_time_us = None
        self._last_sequence = None


class RosSensorPublisher:
    def __init__(self, config: RosPublisherConfig | None = None) -> None:
        self.config = config or RosPublisherConfig()
        if not math.isfinite(self.config.imu_accel_scale) or self.config.imu_accel_scale <= 0.0:
            raise ValueError("imu_accel_scale must be a positive finite value")
        if not math.isfinite(self.config.wheel_radius_m) or self.config.wheel_radius_m < 0.0:
            raise ValueError("wheel_radius_m must be a finite non-negative value")
        if not math.isfinite(self.config.wheel_track_m) or self.config.wheel_track_m < 0.0:
            raise ValueError("wheel_track_m must be a finite non-negative value")
        if not math.isfinite(self.config.wheel_gear_ratio) or self.config.wheel_gear_ratio <= 0.0:
            raise ValueError("wheel_gear_ratio must be a positive finite value")
        wheel_radius_set = self.config.wheel_radius_m > 0.0
        wheel_track_set = self.config.wheel_track_m > 0.0
        if wheel_radius_set != wheel_track_set:
            raise ValueError("wheel_radius_m and wheel_track_m must be set together")
        self.rclpy = import_module("rclpy")
        self.imu_msg = import_module("sensor_msgs.msg").Imu
        self.time_msg = import_module("builtin_interfaces.msg").Time
        qos = import_module("rclpy.qos")

        self.rclpy.init(args=None)
        self.node = self.rclpy.create_node(self.config.node_name)
        sensor_qos = qos.qos_profile_sensor_data
        self.imu_pub = self.node.create_publisher(self.imu_msg, self.config.imu_topic, sensor_qos)
        self.wheel_odom = None
        self.wheel_odom_pub = None
        self.wheel_odom_msg = None
        if wheel_radius_set:
            self.wheel_odom_msg = import_module("nav_msgs.msg").Odometry
            self.wheel_odom_pub = self.node.create_publisher(
                self.wheel_odom_msg,
                self.config.wheel_odom_topic,
                sensor_qos,
            )
            self.wheel_odom = DifferentialDriveOdometry(
                self.config.wheel_radius_m,
                self.config.wheel_track_m,
                self.config.wheel_gear_ratio,
                self.config.wheel1_sign,
                self.config.wheel2_sign,
                self.config.wheel_max_sample_gap_s,
            )
        self._mcu_clock = McuClockSynchronizer()
        self._legacy_warning_emitted = False

    def publish_imu(self, state: ImuState) -> bool:
        if state.sample_time_us is not None and state.sample_sequence is not None:
            try:
                timestamp = self._mcu_clock.to_host_time(
                    state.sample_time_us,
                    state.sample_sequence,
                    state.received_at,
                )
            except ValueError as error:
                self.node.get_logger().warning(f"Dropping invalid IMU v2 frame: {error}")
                return False
        elif self.config.allow_legacy_imu_timestamps:
            timestamp = state.received_at
        else:
            if not self._legacy_warning_emitted:
                self.node.get_logger().error(
                    "Dropping legacy IMU frames without MCU sample timestamps. "
                    "Upgrade the MCU to IMU protocol v2 or explicitly pass "
                    "--allow-legacy-imu-timestamps for temporary diagnostics."
                )
                self._legacy_warning_emitted = True
            return False

        stamp = _stamp_from_unix_time(timestamp, self.time_msg)
        acc_x, acc_y, acc_z = _remap_imu_raw_to_base(state.acc_g)
        gyro_x, gyro_y, gyro_z = _remap_imu_raw_to_base(state.gyro_dps)

        imu_msg = self.imu_msg()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = self.config.imu_frame_id
        imu_msg.orientation_covariance[0] = -1.0
        _fill_diagonal_covariance(
            imu_msg.angular_velocity_covariance,
            self.config.imu_angular_velocity_variance,
        )
        _fill_diagonal_covariance(
            imu_msg.linear_acceleration_covariance,
            self.config.imu_linear_acceleration_variance,
        )
        acc_factor = G_TO_MPS2 * self.config.imu_accel_scale
        imu_msg.linear_acceleration.x = acc_x * acc_factor
        imu_msg.linear_acceleration.y = acc_y * acc_factor
        imu_msg.linear_acceleration.z = acc_z * acc_factor
        imu_msg.angular_velocity.x = gyro_x * DEG_TO_RAD
        imu_msg.angular_velocity.y = gyro_y * DEG_TO_RAD
        imu_msg.angular_velocity.z = gyro_z * DEG_TO_RAD
        self.imu_pub.publish(imu_msg)
        self._publish_wheel_odometry(state, timestamp, stamp)
        return True

    def _publish_wheel_odometry(self, state: ImuState, timestamp: float, stamp) -> None:
        if self.wheel_odom is None or self.wheel_odom_pub is None:
            return
        try:
            odom = self.wheel_odom.update(state.wheel1_rpm, state.wheel2_rpm, timestamp)
        except ValueError as error:
            self.node.get_logger().warning(f"Dropping invalid wheel RPM sample: {error}")
            return

        msg = self.wheel_odom_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.config.wheel_odom_frame_id
        msg.child_frame_id = self.config.wheel_base_frame_id
        msg.pose.pose.position.x = odom.x_m
        msg.pose.pose.position.y = odom.y_m
        half_yaw = 0.5 * odom.yaw_rad
        msg.pose.pose.orientation.z = math.sin(half_yaw)
        msg.pose.pose.orientation.w = math.cos(half_yaw)
        msg.twist.twist.linear.x = odom.linear_m_s
        msg.twist.twist.angular.z = odom.angular_rad_s
        self.wheel_odom_pub.publish(msg)

    def spin_once(self) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self) -> None:
        self.node.destroy_node()
        # rclpy's default SIGINT handler may already have shut the context down
        # before main.py reaches its cleanup block.  Keep close() idempotent so
        # Ctrl+C does not abort the rest of the process cleanup.
        self.rclpy.try_shutdown()


def _remap_imu_raw_to_base(values: tuple[float, float, float]) -> tuple[float, float, float]:
    raw_x, raw_y, raw_z = values
    return raw_x, -raw_z, raw_y


def _stamp_from_unix_time(timestamp: float, time_msg):
    if timestamp <= 0.0:
        timestamp = 0.0
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1_000_000_000)
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    stamp = time_msg()
    stamp.sec = sec
    stamp.nanosec = nanosec
    return stamp


def _fill_diagonal_covariance(covariance, variance: float) -> None:
    covariance[0] = variance
    covariance[4] = variance
    covariance[8] = variance
