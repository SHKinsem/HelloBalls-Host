from __future__ import annotations

import argparse
import pathlib
import select
import sys
import termios
import threading
import time
import tty

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts import HelloBalls_Serial


HOST_MANUAL_CONTROL = 4


class KeyboardController:
    def __init__(
        self,
        serial_receiver: HelloBalls_Serial.SerialReceiver,
        speed: int,
        state: int = HOST_MANUAL_CONTROL,
        command_rate: float = 50.0,
        key_timeout: float = 0.15,
    ) -> None:
        if command_rate <= 0:
            raise ValueError("command_rate must be greater than zero")
        if key_timeout <= 0:
            raise ValueError("key_timeout must be greater than zero")

        self.serial_receiver = serial_receiver
        self.speed = abs(int(speed))
        self.state = int(state)
        self.command_period = 1.0 / command_rate
        self.key_timeout = float(key_timeout)
        self.left_speed = 0
        self.right_speed = 0
        self._movement_deadline = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_terminal_settings = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread = threading.Thread(target=self._run, name="KeyboardController", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._send_stop()
        if self._old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)
            self._old_terminal_settings = None

    def _run(self) -> None:
        print(
            "Keyboard control enabled: "
            "hold w=forward, s=backward, a=rotate left, d=rotate right; space=stop."
        )
        print(
            f"Keyboard speed: {self.speed}; commands stop after "
            f"{self.key_timeout:.2f}s without another movement key."
        )

        while not self._stop_event.is_set():
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                self._handle_key(key)

            self._stop_if_key_stale(time.monotonic())
            self._send_current_command()
            time.sleep(self.command_period)

    def _handle_key(self, key: str) -> None:
        if key == "w":
            self._set_movement(self.speed, self.speed)
            print(f"\rForward L={self.left_speed} R={self.right_speed}   ", end="", flush=True)
        elif key == "s":
            self._set_movement(-self.speed, -self.speed)
            print(f"\rBackward L={self.left_speed} R={self.right_speed}   ", end="", flush=True)
        elif key == "a":
            self._set_movement(-self.speed, self.speed)
            print(f"\rRotate left L={self.left_speed} R={self.right_speed}   ", end="", flush=True)
        elif key == "d":
            self._set_movement(self.speed, -self.speed)
            print(f"\rRotate right L={self.left_speed} R={self.right_speed}   ", end="", flush=True)
        elif key == " ":
            self._set_stop()
            print("\rStop L=0 R=0   ", end="", flush=True)

    def _set_movement(self, left_speed: int, right_speed: int) -> None:
        self.left_speed = left_speed
        self.right_speed = right_speed
        self._movement_deadline = time.monotonic() + self.key_timeout

    def _set_stop(self) -> None:
        self.left_speed = 0
        self.right_speed = 0
        self._movement_deadline = 0.0

    def _stop_if_key_stale(self, now: float) -> None:
        if self._movement_deadline and now >= self._movement_deadline:
            self._set_stop()

    def _send_current_command(self) -> None:
        self.serial_receiver.send_command(
            self.state,
            self.left_speed,
            self.right_speed,
            0,
            0,
        )

    def _send_stop(self) -> None:
        try:
            self.serial_receiver.send_command(self.state, 0, 0, 0, 0)
        except Exception as exc:
            print(f"\nWarning: failed to send stop command: {exc}")


def run_keyboard_control(
    port: str,
    baudrate: int,
    speed: int,
    state: int,
    key_timeout: float = 0.15,
) -> None:
    receiver = HelloBalls_Serial.SerialReceiver(port=port, baudrate=baudrate)
    controller = KeyboardController(
        serial_receiver=receiver,
        speed=speed,
        state=state,
        key_timeout=key_timeout,
    )

    receiver.start()
    controller.start()
    print(f"Serial receiver started on {port}. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping keyboard control.")
    finally:
        controller.close()
        receiver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard drive control for HelloBalls.")
    parser.add_argument("--serial-port", default="/dev/ttyS1", help="UART device path for the MCU.")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--speed", type=int, default=100, help="Wheel speed for keyboard movement.")
    parser.add_argument(
        "--state",
        type=int,
        default=HOST_MANUAL_CONTROL,
        help="MCU state value for movement commands (default: 4, manual/scanning mode).",
    )
    parser.add_argument(
        "--key-timeout",
        type=float,
        default=0.15,
        help="Stop unless another movement key arrives within this many seconds.",
    )
    args = parser.parse_args()

    run_keyboard_control(
        port=args.serial_port,
        baudrate=args.baudrate,
        speed=args.speed,
        state=args.state,
        key_timeout=args.key_timeout,
    )


if __name__ == "__main__":
    main()
