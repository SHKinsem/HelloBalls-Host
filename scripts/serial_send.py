import sys
import time
import threading
import queue
import glob
import os
import importlib
from pathlib import Path

# Serial config
SERIAL_PORT        = None   # e.g., "/dev/tty.usbmodem12345", "COM5", etc.; None = auto-discover
SERIAL_BAUD        = 115200
WRITE_TIMEOUT      = 0.05
OPEN_RETRY_SECS    = 2.0

# Debug logging
DEBUG_SERIAL       = True

def _log(msg: str):
    sys.stderr.write(msg.rstrip() + "\n")

# Try pyserial; if not present, we stay in dry mode
try:
    import serial
    import serial.tools.list_ports as list_ports
except Exception:
    serial = None
    list_ports = None

# --- Robot state constants (mirror CV) ---
ROBOT_STATE_STOP         = 0
ROBOT_STATE_CHASE_BALL   = 1
ROBOT_STATE_RETURN_HOME  = 2
ROBOT_STATE_DELIVER_BALL = 3
ROBOT_STATE_SEARCH       = 4


class SerialBridge:
    """
    Non-blocking serial sender with auto-discovery & auto-reconnect.
    Lines are sent as: "{state},{m1},{m2},{tilt},{friction}\n" (ASCII).
    """
    def __init__(self,
                 port: str | None = SERIAL_PORT,
                 baud: int = SERIAL_BAUD,
                 write_timeout: float = WRITE_TIMEOUT,
                 open_retry_secs: float = OPEN_RETRY_SECS):
        self.user_port = port
        self.baud = baud
        self.write_timeout = write_timeout
        self.open_retry_secs = open_retry_secs

        self._ser = None
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---------- public API ----------
    def send_state(self, state: int, m1: int = 0, m2: int = 0, tilt: int = 0, friction: int = 0):
        line = f"{int(state)},{int(m1)},{int(m2)},{int(tilt)},{int(friction)}\n"
        try:
            self._q.put_nowait(line)
            if DEBUG_SERIAL:
                _log(f"[Serial] Queued: {line.strip()}")
        except queue.Full:
            _log("[Serial] WARNING: send queue full; dropping command")

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass
        self._close_port()

    # ---------- internals ----------
    def _open_port(self) -> bool:
        if serial is None:
            _log("[Serial] pySerial not installed; running in dry mode (no serial). pip install pyserial")
            return False

        # Use user-specified port if provided
        candidates = []
        if self.user_port:
            candidates = [self.user_port]
        else:
            # Try common patterns, prioritizing the ones that work with HelloBalls
            patterns = [
                "/dev/ttyS*",         # Hardware serial ports
                "/dev/ttyUSB*",       # USB serial adapters
                "/dev/ttyACM*",       # Arduino/CDC
                "/dev/tty.usbmodem*", # macOS
                "/dev/tty.usbserial*",# macOS
                "COM*",               # Windows
            ]
            for p in patterns:
                candidates.extend(sorted(glob.glob(p)))
            if list_ports:
                for p in list_ports.comports():
                    if p.device not in candidates:
                        candidates.append(p.device)

        for dev in candidates:
            try:
                if DEBUG_SERIAL:
                    _log(f"[Serial] Trying port: {dev} @ {self.baud}...")

                # match HelloBalls_Serial_25 settings; keep short timeouts
                s = serial.Serial(
                    port=dev,
                    baudrate=self.baud,
                    timeout=0.1,
                    write_timeout=0.1,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_ODD,        # 8O1 (as per your serial module)
                    stopbits=serial.STOPBITS_ONE
                )

                time.sleep(0.05)  # let ESP32 settle
                try:
                    s.reset_input_buffer()
                    s.reset_output_buffer()
                except Exception:
                    pass

                # quick write-test to confirm link
                try:
                    test_cmd = "0,0,0,0,0\n"
                    s.write(test_cmd.encode("ascii"))
                    s.flush()
                    time.sleep(0.01)
                except Exception as write_test_error:
                    _log(f"[Serial] Write test failed on {dev}: {write_test_error}")
                    s.close()
                    continue

                self._ser = s
                _log(f"[Serial] Opened: {dev}")
                return True
            except Exception as e:
                if DEBUG_SERIAL:
                    _log(f"[Serial] Open failed on {dev}: {e}")

        if DEBUG_SERIAL:
            _log("[Serial] No suitable serial port found.")
        return False

    def _close_port(self):
        if self._ser:
            try:
                dev = self._ser.port
            except Exception:
                dev = "<?>"
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            if DEBUG_SERIAL:
                _log(f"[Serial] Closed port {dev}")

    def _run(self):
        """Background sender: opens port, sends queued lines, auto-reconnects on failure."""
        while not self._stop.is_set():
            if self._ser is None:
                if not self._open_port():
                    self._stop.wait(self.open_retry_secs)
                    continue

            try:
                line = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._ser is None:
                continue

            try:
                if DEBUG_SERIAL:
                    _log(f"[Serial] Attempting to send: {line.strip()}")
                written = self._ser.write(line.encode("ascii"))
                self._ser.flush()
                if DEBUG_SERIAL:
                    _log(f"[Serial] Sent {written} bytes.")
            except Exception as e:
                _log(f"[Serial] Write failed: {e}")
                self._close_port()
                # requeue for retry after reconnect
                try:
                    self._q.put_nowait(line)
                except Exception:
                    pass
                self._stop.wait(self.open_retry_secs)
