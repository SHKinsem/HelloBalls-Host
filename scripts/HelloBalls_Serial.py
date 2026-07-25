from __future__ import annotations

import argparse
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import serial
except ImportError:  # pragma: no cover - handled when opening the port.
    serial = None


HEADER = b"\xAA\x55"
MSG_TYPE_IMU_LEGACY = 0x01
MSG_TYPE_IMU_V2 = 0x02
IMU_LEGACY_PAYLOAD_LEN = 22
IMU_V2_PAYLOAD_LEN = 34
FRAME_OVERHEAD_LEN = 5

_IMU_LEGACY_STRUCT = struct.Struct(">BBiihhhhhh")
_IMU_V2_STRUCT = struct.Struct(">BBiihhhhhhIQ")
_SUPPORTED_PAYLOAD_LENGTHS = {
    MSG_TYPE_IMU_LEGACY: IMU_LEGACY_PAYLOAD_LEN,
    MSG_TYPE_IMU_V2: IMU_V2_PAYLOAD_LEN,
}

ACC_RAW_PER_G = 32768.0 / 4.0
GYRO_RAW_PER_DPS = 32768.0 / 512.0


@dataclass(frozen=True)
class ImuState:
    mcu_state: int
    host_state: int
    wheel1_distance: int
    wheel2_distance: int
    acc_x: int
    acc_y: int
    acc_z: int
    gyr_x: int
    gyr_y: int
    gyr_z: int
    received_at: float = 0.0
    protocol_version: int = 1
    sample_sequence: Optional[int] = None
    sample_time_us: Optional[int] = None

    @property
    def wheel1_rpm(self) -> int:
        """Wheel 1 speed; the wire field keeps its legacy name for compatibility."""
        return self.wheel1_distance

    @property
    def wheel2_rpm(self) -> int:
        """Wheel 2 speed; the wire field keeps its legacy name for compatibility."""
        return self.wheel2_distance

    @property
    def acc_g(self) -> tuple[float, float, float]:
        return (
            self.acc_x / ACC_RAW_PER_G,
            self.acc_y / ACC_RAW_PER_G,
            self.acc_z / ACC_RAW_PER_G,
        )

    @property
    def gyro_dps(self) -> tuple[float, float, float]:
        return (
            self.gyr_x / GYRO_RAW_PER_DPS,
            self.gyr_y / GYRO_RAW_PER_DPS,
            self.gyr_z / GYRO_RAW_PER_DPS,
        )


def checksum(frame_without_header_and_checksum: bytes) -> int:
    return sum(frame_without_header_and_checksum) & 0xFF


def parse_payload(
    payload: bytes,
    received_at: Optional[float] = None,
    protocol_version: int = 1,
) -> ImuState:
    if protocol_version == 1:
        expected_length = IMU_LEGACY_PAYLOAD_LEN
        payload_struct = _IMU_LEGACY_STRUCT
    elif protocol_version == 2:
        expected_length = IMU_V2_PAYLOAD_LEN
        payload_struct = _IMU_V2_STRUCT
    else:
        raise ValueError(f"Unsupported IMU protocol version: {protocol_version}")

    if len(payload) != expected_length:
        raise ValueError(f"Invalid payload length: {len(payload)}")

    unpacked = payload_struct.unpack(payload)
    (
        mcu_state,
        host_state,
        wheel1_distance,
        wheel2_distance,
        acc_x,
        acc_y,
        acc_z,
        gyr_x,
        gyr_y,
        gyr_z,
    ) = unpacked[:10]
    sample_sequence = unpacked[10] if protocol_version == 2 else None
    sample_time_us = unpacked[11] if protocol_version == 2 else None

    return ImuState(
        mcu_state=mcu_state,
        host_state=host_state,
        wheel1_distance=wheel1_distance,
        wheel2_distance=wheel2_distance,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        gyr_x=gyr_x,
        gyr_y=gyr_y,
        gyr_z=gyr_z,
        received_at=time.time() if received_at is None else received_at,
        protocol_version=protocol_version,
        sample_sequence=sample_sequence,
        sample_time_us=sample_time_us,
    )


def try_parse_frame(frame: bytes, received_at: Optional[float] = None) -> Optional[ImuState]:
    if len(frame) < FRAME_OVERHEAD_LEN or frame[:2] != HEADER:
        return None

    msg_type = frame[2]
    payload_len = frame[3]
    expected_payload_len = _SUPPORTED_PAYLOAD_LENGTHS.get(msg_type)
    if expected_payload_len is None or payload_len != expected_payload_len:
        return None
    if len(frame) != payload_len + FRAME_OVERHEAD_LEN:
        return None

    payload_end = 4 + payload_len
    payload = frame[4:payload_end]
    received_checksum = frame[payload_end]
    expected_checksum = checksum(frame[2:payload_end])

    if received_checksum != expected_checksum:
        return None

    protocol_version = 2 if msg_type == MSG_TYPE_IMU_V2 else 1
    return parse_payload(
        payload,
        received_at=received_at,
        protocol_version=protocol_version,
    )


class FrameParser:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[ImuState]:
        self._buffer.extend(data)
        frames: list[ImuState] = []

        while True:
            header_index = self._buffer.find(HEADER)
            if header_index < 0:
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                break
            if header_index > 0:
                del self._buffer[:header_index]
            if len(self._buffer) < 4:
                break

            msg_type = self._buffer[2]
            payload_len = self._buffer[3]
            expected_payload_len = _SUPPORTED_PAYLOAD_LENGTHS.get(msg_type)
            if expected_payload_len is None or payload_len != expected_payload_len:
                del self._buffer[0]
                continue

            frame_len = payload_len + FRAME_OVERHEAD_LEN
            if len(self._buffer) < frame_len:
                break

            candidate = bytes(self._buffer[:frame_len])
            parsed = try_parse_frame(candidate)
            if parsed is not None:
                frames.append(parsed)
                del self._buffer[:frame_len]
                continue

            del self._buffer[0]

        return frames


class SerialReceiver:
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.05,
        read_size: int = 64,
        state_callback: Optional[Callable[[ImuState], None]] = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.read_size = read_size
        self.latest_state: Optional[ImuState] = None
        self.frames_received = 0
        self.state_callback = state_callback
        self._parser = FrameParser()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._serial = None

    def open(self) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install it with: pip install pyserial")

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_ODD,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
        )

    def close(self) -> None:
        self.stop()
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def read_frame(self) -> Optional[ImuState]:
        if self._serial is None:
            self.open()

        data = self._serial.read(self.read_size)
        if not data:
            return None

        frames = self._parser.feed(data)
        if not frames:
            return None

        state = frames[-1]
        with self._lock:
            self.latest_state = state
            self.frames_received += len(frames)
        if self.state_callback is not None:
            for parsed_state in frames:
                self.state_callback(parsed_state)
        return state

    def send_command(
        self,
        state: int,
        motor_speed_1: int,
        motor_speed_2: int,
        tilt_angle: int = 0,
        friction_wheel_speed: int = 0,
    ) -> None:
        if self._serial is None:
            self.open()

        command = f"{state},{motor_speed_1},{motor_speed_2},{tilt_angle},{friction_wheel_speed}\n"
        with self._write_lock:
            self._serial.write(command.encode("ascii"))

    def send_movement_command(
        self,
        state: int,
        wheel1_speed: int,
        wheel2_speed: int,
    ) -> None:
        if self._serial is None:
            self.open()

        command = f"{state},{wheel1_speed},{wheel2_speed}\n"
        with self._write_lock:
            self._serial.write(command.encode("ascii"))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        if self._serial is None:
            self.open()

        self._thread = threading.Thread(target=self._read_loop, name="SerialReceiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def get_latest_state(self) -> Optional[ImuState]:
        with self._lock:
            return self.latest_state

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            self.read_frame()


def format_state(state: ImuState) -> str:
    acc_g = state.acc_g
    gyro_dps = state.gyro_dps
    return (
        f"protocol=v{state.protocol_version} seq={state.sample_sequence} "
        f"sample_time_us={state.sample_time_us} "
        f"mcu={state.mcu_state} host={state.host_state} "
        f"wheel=({state.wheel1_distance}, {state.wheel2_distance}) "
        f"acc_raw=({state.acc_x}, {state.acc_y}, {state.acc_z}) "
        f"acc_g=({acc_g[0]:.3f}, {acc_g[1]:.3f}, {acc_g[2]:.3f}) "
        f"gyro_raw=({state.gyr_x}, {state.gyr_y}, {state.gyr_z}) "
        f"gyro_dps=({gyro_dps[0]:.2f}, {gyro_dps[1]:.2f}, {gyro_dps[2]:.2f})"
    )


def serial_monitor(port: str, baudrate: int = 115200) -> None:
    receiver = SerialReceiver(port=port, baudrate=baudrate)
    receiver.open()
    print(f"Listening on {port} at {baudrate} baud, odd parity. Press Ctrl+C to stop.")

    try:
        while True:
            state = receiver.read_frame()
            if state is not None:
                print(format_state(state))
    except KeyboardInterrupt:
        print("\nSerial monitor stopped.")
    finally:
        receiver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive ESP32S3 IMU frames over UART.")
    parser.add_argument("--port", default="/dev/ttyS1", help="UART device path, for example /dev/ttyS0")
    parser.add_argument("--baudrate", type=int, default=115200)
    args = parser.parse_args()
    serial_monitor(port=args.port, baudrate=args.baudrate)


if __name__ == "__main__":
    main()
