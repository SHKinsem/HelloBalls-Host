from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

from scripts.HelloBalls_Serial import ImuState

G_TO_MPS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0


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
        self.rclpy = import_module("rclpy")
        self.imu_msg = import_module("sensor_msgs.msg").Imu
        self.time_msg = import_module("builtin_interfaces.msg").Time
        qos = import_module("rclpy.qos")

        self.rclpy.init(args=None)
        self.node = self.rclpy.create_node(self.config.node_name)
        sensor_qos = qos.qos_profile_sensor_data
        self.imu_pub = self.node.create_publisher(self.imu_msg, self.config.imu_topic, sensor_qos)
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
        return True

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
