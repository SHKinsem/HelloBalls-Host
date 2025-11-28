#!/usr/bin/env python3
"""
VOICE31_serial_bridge_v2.py

Bridges VOICE31's 4 voice intents to the robot via a robust, non-blocking
serial sender thread. Avoids VOSK grammar logs, keeps "fetch" asserted for a
few seconds, and never blocks the VOICE loop.

Voice mapping (presets):
  "ball please"  -> DELIVER_BALL (3)
  "fetch" / "fetch ball" -> CHASE_BALL (1) for FETCH_HOLD_SECONDS
  "switch"       -> SEARCH (4)
  "stop"         -> STOP (0)
"""

import sys
import time
import threading
import queue
import glob
import os
import importlib
from pathlib import Path

# ===================== TUNABLES =====================
# How long "fetch" lasts, and how often to reassert the state.
FETCH_HOLD_SECONDS = 8.0
REASSERT_INTERVAL  = 1.0

# Serial config
SERIAL_PORT        = None   # e.g., "/dev/tty.usbmodem12345", "COM5", etc.; None = auto-discover
SERIAL_BAUD        = 115200
WRITE_TIMEOUT      = 0.05
OPEN_RETRY_SECS    = 2.0

# Debug logging
DEBUG_SERIAL       = True
# ===================================================


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


def _log(msg: str):
    sys.stderr.write(msg.rstrip() + "\n")


class SerialBridge:
    """
    Non-blocking serial sender with auto-discovery & auto-reconnect.
    Lines are sent as: "{state},{m1},{m2},{tilt},{friction}\\n" (ASCII).
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
                "/dev/ttyS*",        # Hardware serial ports (like /dev/ttyS1, /dev/ttyS5)
                "/dev/ttyUSB*",      # USB serial adapters
                "/dev/ttyACM*",      # Arduino boards
                "/dev/tty.usbmodem*", # macOS
                "/dev/tty.usbserial*", # macOS
                "COM*",              # Windows
            ]
            for p in patterns:
                candidates.extend(sorted(glob.glob(p)))  # Sort to get consistent ordering
            if list_ports:
                for p in list_ports.comports():
                    if p.device not in candidates:
                        candidates.append(p.device)

        for dev in candidates:
            try:
                if DEBUG_SERIAL:
                    _log(f"[Serial] Trying port: {dev} @ {self.baud}...")
                
                # Use exact same settings as HelloBalls_Serial_25.py but without the problematic nonblocking
                s = serial.Serial(
                    port=dev,
                    baudrate=self.baud,
                    timeout=0.1,  # Small timeout instead of 0 to avoid blocking issues
                    write_timeout=0.1,  # Small write timeout to prevent hanging
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_ODD,  # Same as HelloBalls_Serial_25
                    stopbits=serial.STOPBITS_ONE
                )
                
                # Don't set nonblocking attribute - it doesn't exist and causes issues
                
                # Small delay helps some MCUs settle after opening
                time.sleep(0.05)
                # Clear buffers
                try:
                    s.reset_input_buffer()
                    s.reset_output_buffer()
                except Exception:
                    pass
                    
                # Test the connection by trying to write a simple command
                try:
                    test_cmd = "0,0,0,0,0\n"
                    s.write(test_cmd.encode("ascii"))
                    s.flush()
                    time.sleep(0.01)  # Brief pause after test write
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
                    # Wait then retry open
                    self._stop.wait(self.open_retry_secs)
                    continue

            try:
                line = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._ser is None:
                # Port dropped between get and write
                continue

            try:
                # Ensure we write ASCII bytes, then flush
                if DEBUG_SERIAL:
                    _log(f"[Serial] Attempting to send: {line.strip()}")
                written = self._ser.write(line.encode("ascii"))
                self._ser.flush()
                if DEBUG_SERIAL:
                    _log(f"[Serial] Sent {written} bytes.")
            except Exception as e:
                _log(f"[Serial] Write failed: {e}")
                self._close_port()
                # Re-enqueue the line for retry after reconnect (best-effort)
                try:
                    self._q.put_nowait(line)
                except Exception:
                    pass
                # Back off a bit before trying to reopen
                self._stop.wait(self.open_retry_secs)


# Global bridge instance
_bridge = SerialBridge()

def _send_state(state, m1=0, m2=0, tilt=0, friction=0):
    """Thin wrapper so handler code stays tidy."""
    _bridge.send_state(state, m1, m2, tilt, friction)


# --- "fetch" hold support ---
_fetch_lock = threading.Lock()
_fetch_thread: threading.Thread | None = None
_fetch_cancel = threading.Event()

def _hold_state_for(seconds: float, state: int):
    """Re-assert a state every REASSERT_INTERVAL seconds for `seconds`."""
    global _fetch_thread
    with _fetch_lock:
        if _fetch_thread and _fetch_thread.is_alive():
            _fetch_cancel.set()
            _fetch_thread.join(timeout=0.5)

        _fetch_cancel.clear()

        def _worker():
            end_time = time.time() + max(0.0, seconds)
            while time.time() < end_time and not _fetch_cancel.is_set():
                # Check cancellation more frequently
                _send_state(state, 0, 0, 0, 0)
                # Sleep in smaller chunks to be more responsive to cancellation
                sleep_remaining = REASSERT_INTERVAL
                while sleep_remaining > 0 and not _fetch_cancel.is_set():
                    chunk = min(0.1, sleep_remaining)  # Sleep in 100ms chunks
                    time.sleep(chunk)
                    sleep_remaining -= chunk
            
            # Send immediate stop when cancelled or completed
            if _fetch_cancel.is_set():
                print("[Bridge] Fetch sequence cancelled, sending STOP")
            else:
                print("[Bridge] Fetch sequence completed, sending STOP")
            _send_state(ROBOT_STATE_STOP, 0, 0, 0, 0)

        _fetch_thread = threading.Thread(target=_worker, daemon=True)
        _fetch_thread.start()

def on_fetch_ball():
    try:
        V.on_fetch_ball.__wrapped__()  # type: ignore[attr-defined]
    except AttributeError:
        print("[ACTION] Fetching the ball")
    _hold_state_for(FETCH_HOLD_SECONDS, ROBOT_STATE_CHASE_BALL)

def on_ball_please():
    try:
        V.on_ball_please.__wrapped__()  # type: ignore[attr-defined]
    except AttributeError:
        print("[ACTION] Serving one ball")
    # Serial action: send DELIVER_BALL with proper tilt and friction parameters
    # These values should be tuned for your specific robot setup
    tilt_angle = 45  # Adjust this value based on your robot's delivery mechanism
    friction_speed = 5000  # Adjust this value (range: 1000-9000) for proper ball delivery
    _send_state(ROBOT_STATE_DELIVER_BALL, 0, 0, tilt_angle, friction_speed)

def on_switch():
    try:
        V.on_switch.__wrapped__()  # type: ignore[attr-defined]
    except AttributeError:
        print("[ACTION] Switching target")
    _send_state(ROBOT_STATE_SEARCH, 0, 0, 0, 0)

def on_stop():
    try:
        V.on_stop.__wrapped__()  # type: ignore[attr-defined]
    except AttributeError:
        print("[ACTION] Stopping")
    # Immediately cancel any ongoing fetch sequence
    _fetch_cancel.set()
    # Send stop command immediately
    _send_state(ROBOT_STATE_STOP, 0, 0, 0, 0)
    print("[Bridge] Stop command sent immediately")


def _wrap_once():
    # Preserve originals so we can call them
    if hasattr(V, "on_fetch_ball") and not hasattr(V.on_fetch_ball, "__wrapped__"):
        setattr(on_fetch_ball, "__wrapped__", V.on_fetch_ball)
    if hasattr(V, "on_ball_please") and not hasattr(V.on_ball_please, "__wrapped__"):
        setattr(on_ball_please, "__wrapped__", V.on_ball_please)
    if hasattr(V, "on_switch") and not hasattr(V.on_switch, "__wrapped__"):
        setattr(on_switch, "__wrapped__", V.on_switch)
    if hasattr(V, "on_stop") and not hasattr(V.on_stop, "__wrapped__"):
        setattr(on_stop, "__wrapped__", V.on_stop)

    # Patch module functions
    V.on_fetch_ball  = on_fetch_ball
    V.on_ball_please = on_ball_please
    V.on_switch      = on_switch
    V.on_stop        = on_stop

    # Update intent map if present
    try:
        V.INTENT_MAP.update({
            "fetch": V.on_fetch_ball,
            "fetch ball": V.on_fetch_ball,
            "ball please": V.on_ball_please,
            "switch": V.on_switch,
            "stop": V.on_stop,
        })
    except Exception:
        pass


def main():
    _wrap_once()
    try:
        V.main()
    finally:
        _bridge.close()


if __name__ == "__main__":
    try:
        # Allow quick override via env var without editing file:
        #   export BRIDGE_SERIAL_PORT=/dev/ttyUSB0
        #   export BRIDGE_SERIAL_BAUD=115200
        sp = os.environ.get("BRIDGE_SERIAL_PORT")
        if sp:
            SERIAL_PORT = sp  # type: ignore[assignment]
            _log(f"[Serial] Using BRIDGE_SERIAL_PORT={sp}")
        sb = os.environ.get("BRIDGE_SERIAL_BAUD")
        if sb:
            try:
                SERIAL_BAUD = int(sb)  # type: ignore[assignment]
                _log(f"[Serial] Using BRIDGE_SERIAL_BAUD={SERIAL_BAUD}")
            except Exception:
                _log(f"[Serial] Invalid BRIDGE_SERIAL_BAUD={sb}")
        main()
    except KeyboardInterrupt:
        _fetch_cancel.set()
        _bridge.close()
        print("\n[INFO] Exiting bridge.")
