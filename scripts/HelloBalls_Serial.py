from __future__ import annotations

import argparse
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import serial
except ImportError:  # pragma: no cover - handled when opening the port.
    serial = None


HEADER = b"\xAA\x55"
MSG_TYPE_IMU = 0x01
PAYLOAD_LEN = 22
FRAME_LEN = 27

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


def parse_payload(payload: bytes, received_at: Optional[float] = None) -> ImuState:
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"Invalid payload length: {len(payload)}")

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
    ) = struct.unpack(">BBiihhhhhh", payload)

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
    )


def try_parse_frame(frame: bytes, received_at: Optional[float] = None) -> Optional[ImuState]:
    if len(frame) != FRAME_LEN:
        return None
    if frame[:2] != HEADER:
        return None

    msg_type = frame[2]
    payload_len = frame[3]
    if msg_type != MSG_TYPE_IMU or payload_len != PAYLOAD_LEN:
        return None

    payload_end = 4 + payload_len
    payload = frame[4:payload_end]
    received_checksum = frame[payload_end]
    expected_checksum = checksum(frame[2:payload_end])

    if received_checksum != expected_checksum:
        return None

    return parse_payload(payload, received_at=received_at)


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
            if len(self._buffer) < FRAME_LEN:
                break

            candidate = bytes(self._buffer[:FRAME_LEN])
            parsed = try_parse_frame(candidate)
            if parsed is not None:
                frames.append(parsed)
                del self._buffer[:FRAME_LEN]
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
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.read_size = read_size
        self.latest_state: Optional[ImuState] = None
        self.frames_received = 0
        self._parser = FrameParser()
        self._lock = threading.Lock()
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
        return state

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
