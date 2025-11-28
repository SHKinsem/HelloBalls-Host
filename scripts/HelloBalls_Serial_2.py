import serial
import time
import glob
import sys
import serial.tools.list_ports
import os
import termios
import tty
import fcntl
import select
import threading
import queue


  
class SerialComm:
    """
    Class to handle serial communication with MCU for the HelloBalls project.
    Sends messages in the format "state,motor_speed_1,motor_speed_2".
    """
    
    def __init__(self, port=None, baud_rate=115200, timeout=0.1, auto_reconnect=True, reconnect_interval=2):
        """
        Initialize the serial communication.
        
        Args:
            port (str, optional): Serial port name. If None, auto-detection will be used
            baud_rate (int): Baud rate for the serial communication
            timeout (float): Read timeout in seconds
            auto_reconnect (bool): Whether to automatically try to reconnect if connection is lost
            reconnect_interval (float): Interval in seconds between reconnection attempts
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.connected = False
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.last_reconnect_time = 0
        self.message_buffer = []
        
    @staticmethod
    def list_available_ports():
        """
        List all available serial ports.
        
        Returns:
            list: List of available port names
        """
        return [port.device for port in serial.tools.list_ports.comports()]
    
    @staticmethod
    def find_port(common_patterns=None):
        """
        Find a suitable serial port based on common patterns.
        
        Args:
            common_patterns (list, optional): List of patterns to search for. If None, default patterns will be used.
            
        Returns:
            str or None: Found port name or None if not found
        """
        if common_patterns is None:
            # Default patterns for different platforms
            if sys.platform.startswith('linux'):
                # Common patterns for Linux
                common_patterns = [
                    '/dev/ttyUSB*',  # USB-to-Serial adapters
                    '/dev/ttyACM*',  # Arduino boards
                    '/dev/ttyS*',    # Hardware serial ports
                    '/dev/serial/by-id/*'  # Persistent serial port IDs
                ]
            elif sys.platform.startswith('win'):
                # On Windows, we'll use list_available_ports directly
                ports = SerialComm.list_available_ports()
                if ports:
                    return ports[0]  # Return the first available port
                return None
            elif sys.platform.startswith('darwin'):
                # Common patterns for macOS
                common_patterns = [
                    '/dev/tty.usbmodem*',  # Arduino boards
                    '/dev/tty.usbserial*'  # USB-to-Serial adapters
                ]
            else:
                return None
        
        # Search for ports matching the patterns
        available_ports = []
        for pattern in common_patterns:
            available_ports.extend(glob.glob(pattern))
        
        if available_ports:
            return available_ports[0]  # Return the first found port
        else:
            return None
        
    def connect(self, port=None):
        """
        Connect to the serial port.
        
        Args:
            port (str, optional): The port to connect to. If None, use the instance's port
                                 or auto-detect if that's also None.
                                 
        Returns:
            bool: True if connection successful, False otherwise
        """
        # If port is provided, update the instance's port
        if port is not None:
            self.port = port
        
        # If port is still None, try to find a suitable port
        if self.port is None:
            self.port = self.find_port()
            if self.port is None:
                print("No suitable serial port found")
                return False
        
        try:
            # Try with PARITY_ODD first (existing default in repo)
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_ODD,
                stopbits=serial.STOPBITS_ONE
            )
        except serial.SerialException as e_first:
            # If initial attempt failed, try again with no parity which is a common default
            try:
                print(f"Initial serial connect failed ({e_first}), retrying with PARITY_NONE...")
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=self.timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
            except serial.SerialException as e_second:
                # Report the second error and fall back to original handling below
                print(f"Retry with PARITY_NONE also failed: {e_second}")
                raise
            
            # Set non-blocking mode
            try:
                # Some pyserial versions may not have nonblocking attribute
                self.ser.nonblocking = True
            except Exception:
                pass
            # For non-blocking reads we set a tiny timeout
            self.ser.timeout = 0
            
            self.connected = True
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Disconnect from the serial port"""
        if self.connected and self.ser:
            self.ser.close()
            self.connected = False
            print(f"Disconnected from {self.port}")
    
    def ensure_connection(self):
        """
        Ensure that the serial connection is active.
        If auto_reconnect is enabled and the connection is lost, try to reconnect.
        
        Returns:
            bool: True if connected (either already or after reconnect), False otherwise
        """
        if self.connected and self.ser:
            # Check if the connection is still valid
            try:
                # This will raise an exception if the port is no longer available
                if not self.ser.is_open:
                    self.ser.open()
                return True
            except (serial.SerialException, AttributeError):
                self.connected = False
                print(f"Serial connection to {self.port} lost")
        
        # If auto_reconnect is enabled, try to reconnect
        if self.auto_reconnect and not self.connected:
            current_time = time.time()
            # Only try to reconnect if enough time has passed since the last attempt
            if current_time - self.last_reconnect_time >= self.reconnect_interval:
                self.last_reconnect_time = current_time
                print(f"Attempting to reconnect to {self.port}...")
                return self.connect()
        
        return self.connected
            
    def send_command(self, state, motor_speed_1, motor_speed_2, tilt_angle=0, friction_wheel_speed=0):
        """
        Send a command to the MCU in the format "state,motor_speed_1,motor_speed_2,tilt_angle,friction_wheel_speed".
        If auto_reconnect is enabled and the connection is lost, tries to reconnect.
        
        Args:
            state (int): 3 value (e.g., 0 for stop, 1 for run)
            motor_speed_1 (int): Speed value for motor 1
            motor_speed_2 (int): Speed value for motor 2
            tilt_angle (int): Tilt angle value (default: 0)
            friction_wheel_speed (int): Friction wheel speed value (default: 0, range: 1000-9000)
            
        Returns:
            bool: True if command was sent successfully, False otherwise
        """
        # Ensure connection is active (only check if not connected to avoid overhead)
        if not self.connected and not self.ensure_connection():
            print("Not connected to serial port and reconnection failed")
            return False
            
        try:
            # Format the command as "state,motor_speed_1,motor_speed_2,tilt_angle,friction_wheel_speed"
            command = f"{state},{motor_speed_1},{motor_speed_2},{tilt_angle},{friction_wheel_speed}\n"
            
            # Debug output for state 3 commands
            if state == 3 and (tilt_angle != 0 or friction_wheel_speed != 0):
                print(f"Sending command: {command.strip()}")
            
            self.ser.write(command.encode('ascii'))
            # Remove flush() for non-blocking operation - let OS buffer handle transmission
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            self.connected = False  # Mark as disconnected to trigger reconnect on next attempt
            return False
    
    def read_response(self, timeout=0.01):
        """
        Read a response from the MCU with optimized non-blocking reads.
        If auto_reconnect is enabled and the connection is lost, tries to reconnect.
        
        Args:
            timeout (float): Maximum time to wait for a response
            
        Returns:
            str: Response from the MCU or None if no response
        """
        # Ensure connection is active
        if not self.ensure_connection():
            return None
            
        try:
            buffer = b''
            start_time = time.time()
            
            # Non-blocking read with timeout
            while (time.time() - start_time) < timeout:
                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting)
                    buffer += chunk
                    if b'\n' in buffer:
                        break
                else:
                    # Tiny sleep to prevent CPU hogging
                    time.sleep(0.001)
                    
            if buffer:
                # Process and return the first complete line
                lines = buffer.split(b'\n')
                return lines[0].decode('ascii', errors='ignore').strip()
            return None
        except Exception as e:
            print(f"Error reading response: {e}")
            self.connected = False
            return None
            
    def receive_status_message(self, timeout=0.01):
        """
        Receive and parse a status message in the format:
        "MSG,state,wheel1_distance,wheel2_distance,imu_x,imu_y,imu_z,imu_yaw"
        
        This function uses optimized non-blocking reads.
        
        Args:
            timeout (float): Maximum time to wait for a valid message
            
        Returns:
            dict: Parsed message or None if no valid message received
        """
        # Ensure connection is active
        if not self.ensure_connection():
            return None
            
        try:
            buffer = b''
            start_time = time.time()
            
            # Process any existing data in buffer
            if self.ser.in_waiting > 0:
                # Read all available data
                buffer = self.ser.read(self.ser.in_waiting)
                
                # Process all complete lines in buffer
                if b'\n' in buffer:
                    lines = buffer.split(b'\n')
                    # Keep the incomplete last line in the buffer
                    if not buffer.endswith(b'\n'):
                        buffer = lines[-1]
                        lines = lines[:-1]
                    else:
                        buffer = b''
                    
                    # Process all complete lines
                    for line in lines:
                        if line:
                            decoded_line = line.decode('ascii', errors='ignore').strip()
                            # Check if this is a status message (starts with MSG)
                            if decoded_line.startswith("MSG"):
                                parts = decoded_line.split(',')
                                
                                # Verify message format
                                if len(parts) >= 8:  # MSG + 7 data fields
                                    try:
                                        # Parse the message into a dictionary
                                        status = {
                                            'state': int(parts[1]),
                                            'wheel1_distance': float(parts[2]),
                                            'wheel2_distance': float(parts[3]),
                                            'imu_x': float(parts[4]),
                                            'imu_y': float(parts[5]), 
                                            'imu_z': float(parts[6]),
                                            'imu_yaw': float(parts[7])
                                        }
                                        return status
                                    except (ValueError, IndexError):
                                        pass
            return None
        except Exception as e:
            print(f"Error reading status message: {e}")
            self.connected = False
            return None

    @staticmethod
    def get_state3_inputs_threaded(current_tilt, current_friction, input_queue, old_settings):
        """Function to run in separate thread for tilt angle and friction wheel speed input"""
        try:
            # Temporarily restore terminal settings for input
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            # Force flush stdout and stderr to ensure prompt appears immediately
            sys.stdout.flush()
            sys.stderr.flush()
            
            print(f"\nEnter tilt angle for state 3 (current: {current_tilt}): ", end="", flush=True)
            
            # Make stdin blocking temporarily for this thread
            stdin_fd = sys.stdin.fileno()
            old_flags = fcntl.fcntl(stdin_fd, fcntl.F_GETFL)
            fcntl.fcntl(stdin_fd, fcntl.F_SETFL, old_flags & ~os.O_NONBLOCK)
            
            try:
                # Get tilt angle
                tilt_input = sys.stdin.readline().strip()
                new_tilt = current_tilt
                if tilt_input:
                    try:
                        new_tilt = int(tilt_input)
                        print(f"Tilt angle set to: {new_tilt}")
                    except ValueError:
                        print(f"Invalid tilt input, keeping current tilt angle: {current_tilt}")
                else:
                    print(f"Keeping current tilt angle: {current_tilt}")
                
                # Get friction wheel speed
                print(f"Enter friction wheel speed (1000-9000, current: {current_friction}): ", end="", flush=True)
                friction_input = sys.stdin.readline().strip()
                new_friction = current_friction
                if friction_input:
                    try:
                        friction_value = int(friction_input)
                        if 1000 <= friction_value <= 9000:
                            new_friction = friction_value
                            print(f"Friction wheel speed set to: {new_friction}")
                        else:
                            print(f"Friction speed out of range (1000-9000), keeping current: {current_friction}")
                    except ValueError:
                        print(f"Invalid friction input, keeping current friction speed: {current_friction}")
                else:
                    print(f"Keeping current friction wheel speed: {current_friction}")
                
                input_queue.put(('success', new_tilt, new_friction))
                
            finally:
                # Restore non-blocking mode
                fcntl.fcntl(stdin_fd, fcntl.F_SETFL, old_flags)
                
        except (EOFError, KeyboardInterrupt):
            input_queue.put(('cancelled', current_tilt, current_friction))
            print(f"Input cancelled, keeping current values: tilt={current_tilt}, friction={current_friction}")
        except Exception as e:
            input_queue.put(('error', current_tilt, current_friction))
            print(f"Error getting input: {e}, keeping current values: tilt={current_tilt}, friction={current_friction}")
        
        print("Returning to control mode...")

    @staticmethod
    def setup_non_blocking_input():
        """Setup non-blocking input for keyboard control"""
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        # Set terminal to raw mode
        tty_settings = termios.tcgetattr(sys.stdin)
        tty_settings[3] = tty_settings[3] & ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, tty_settings)
        
        # Set non-blocking
        fcntl.fcntl(sys.stdin, fcntl.F_SETFL, os.O_NONBLOCK)
        
        return old_settings

    @staticmethod
    def restore_terminal_settings(old_settings):
        """Restore terminal settings"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# For testing purposes
if __name__ == "__main__":
    print("Available ports:", SerialComm.list_available_ports())
    auto_port = SerialComm.find_port()
    print(f"Auto-detected port: {auto_port}")
    # Create a SerialComm instance with auto-detection, auto-reconnect and shorter timeout
    serial_comm = SerialComm(auto_reconnect=False, timeout=0.01)
    serial_comm.connect('/dev/ttyS1')
    try:
        # Connect to the auto-detected serial port
        if serial_comm.connect():
            print("Connected successfully. Starting test...")
            
            # Send some test commands
            print("Sending stop command (0, 0, 0, 0, 0)")
            serial_comm.send_command(0, 0, 0, 0, 0)
            time.sleep(0.1)  # Reduced sleep time
            
            # Choose mode
            mode = input("Choose mode - [1] Send at 50Hz, [2] Receive status messages, [3] Both, [4] Keyboard controller, [5] Debug mode: ")
            send_active = mode in ["1", "3"]
            receive_active = mode in ["2", "3"]
            keyboard_active = mode == "4"
            debug_active = mode == "5"
            
            # Common control variables for all modes
            current_state = 1
            current_tilt_angle = 0
            current_friction_wheel_speed = 0
            waiting_for_inputs = False
            input_queue = queue.Queue()
            input_thread = None
            previous_state = current_state
            
            def handle_state_input(key, old_settings):
                """Handle state change input (shared across all modes)"""
                global current_state, previous_state, waiting_for_inputs, input_thread
                global current_tilt_angle, current_friction_wheel_speed
                
                if key in ['0', '1', '2', '3']:
                    new_state = int(key)
                    if new_state != current_state:
                        previous_state = current_state
                        current_state = new_state
                        print(f"State changed to: {current_state}")
                        
                        # Special handling for state 3
                        if current_state == 3 and not waiting_for_inputs:
                            waiting_for_inputs = True
                            print(f"Continuing with previous state ({previous_state}) while waiting for inputs...")
                            # Start input thread
                            input_thread = threading.Thread(
                                target=SerialComm.get_state3_inputs_threaded,
                                args=(current_tilt_angle, current_friction_wheel_speed, input_queue, old_settings),
                                daemon=True
                            )
                            input_thread.start()
                    return True
                return False
            
            def check_input_completion():
                """Check if state 3 input is complete (shared across all modes)"""
                global waiting_for_inputs, current_tilt_angle, current_friction_wheel_speed
                
                if waiting_for_inputs:
                    try:
                        result_type, result_tilt, result_friction = input_queue.get_nowait()
                        current_tilt_angle = result_tilt
                        current_friction_wheel_speed = result_friction
                        waiting_for_inputs = False
                        print(f"Input complete. Now using state {current_state} with tilt angle {current_tilt_angle} and friction speed {current_friction_wheel_speed}")
                        return True
                    except queue.Empty:
                        pass
                return False
            
            def get_effective_params():
                """Get effective state and parameters based on current status"""
                if waiting_for_inputs and current_state == 3:
                    return previous_state, 0, 0  # Use previous state with default params
                else:
                    tilt_angle = current_tilt_angle if current_state == 3 else 0
                    friction_speed = current_friction_wheel_speed if current_state == 3 else 0
                    return current_state, tilt_angle, friction_speed
            
            if keyboard_active:
                print("\n--- Keyboard Controller Mode ---")
                print("w/a/s/d: Move forward/left/backward/right")
                print("q/e: Rotate left/right")
                print("0/1/2/3: Set robot state (State 3 allows tilt angle and friction speed input)")
                print("space: Stop motors")
                print("Ctrl+C: Exit program")
                print("\nNote: When selecting state 3, you'll be prompted to enter parameters.")
                print("Commands will continue with the previous state until inputs are complete.")
                print(f"\nKeyboard controller active. Current state: {current_state}")
                
                # Setup non-blocking input
                old_settings = SerialComm.setup_non_blocking_input()
                
                # Movement parameters
                speed = 100
                motor1, motor2 = 0, 0
                
                try:
                    while True:
                        # Check for keyboard input
                        ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
                        if ready_to_read:
                            try:
                                key = sys.stdin.read(1)
                                if key:
                                    if handle_state_input(key, old_settings):
                                        # State changed, restore non-blocking mode
                                        old_settings = SerialComm.setup_non_blocking_input()
                                    elif key == 'w':
                                        motor1, motor2 = speed, speed
                                        print(f"Moving forward: {motor1}, {motor2}")
                                    elif key == 's':
                                        motor1, motor2 = -speed, -speed
                                        print(f"Moving backward: {motor1}, {motor2}")
                                    elif key == 'a':
                                        motor1, motor2 = -speed, speed
                                        print(f"Turning left: {motor1}, {motor2}")
                                    elif key == 'd':
                                        motor1, motor2 = speed, -speed
                                        print(f"Turning right: {motor1}, {motor2}")
                                    elif key == 'q':
                                        motor1, motor2 = -speed//2, speed//2
                                        print(f"Rotating left: {motor1}, {motor2}")
                                    elif key == 'e':
                                        motor1, motor2 = speed//2, -speed//2
                                        print(f"Rotating right: {motor1}, {motor2}")
                                    elif key == ' ':
                                        motor1, motor2 = 0, 0
                                        print(f"Stopping: {motor1}, {motor2}")
                            except IOError:
                                pass
                        
                        # Check input completion
                        if check_input_completion():
                            old_settings = SerialComm.setup_non_blocking_input()
                        
                        # Send commands
                        effective_state, tilt_angle, friction_speed = get_effective_params()
                        
                        # Debug output for state 3 to verify tilt angle is being sent
                        if effective_state == 3 and (tilt_angle != 0 or friction_speed != 0):
                            print(f"\nSending State 3: tilt_angle={tilt_angle}, friction_speed={friction_speed}")
                        
                        success = serial_comm.send_command(effective_state, motor1, motor2, tilt_angle, friction_speed)
                        if not success:
                            print(f"\nFailed to send command: state={effective_state}, motors=({motor1},{motor2}), tilt={tilt_angle}, friction={friction_speed}")
                        
                        time.sleep(0.02)  # 50Hz update rate
                        
                except KeyboardInterrupt:
                    pass
                finally:
                    if input_thread and input_thread.is_alive():
                        input_thread.join(timeout=0.5)
                    SerialComm.restore_terminal_settings(old_settings)
            elif debug_active:
                print("\n--- Debug Mode ---")
                print("0/1/2/3: Set robot state (State 3 allows tilt angle and friction speed input)")
                print("Send custom messages at specified frequency")
                print("Enter message in format 'motor1,motor2' (e.g., '100,100')")
                print("You can change the message during runtime by pressing Enter and typing a new one")
                print("Ctrl+C: Exit program")
                print("\nNote: When selecting state 3, you'll be prompted to enter parameters.")
                print("Commands will continue with the previous state until inputs are complete.")
                
                # Setup non-blocking input
                old_settings = SerialComm.setup_non_blocking_input()
                
                # Get initial debug parameters
                SerialComm.restore_terminal_settings(old_settings)
                try:
                    message_input = input("Enter initial motor speeds 'motor1,motor2' (default '100,100'): ").strip() or "100,100"
                    motor1, motor2 = map(int, message_input.split(','))
                    frequency = float(input("Enter frequency in Hz (default 50): ") or "50")
                except ValueError as e:
                    print(f"Invalid input ({e}), using defaults: 100,100 at 50Hz")
                    motor1, motor2, frequency = 100, 100, 50.0
                
                send_interval = 1.0 / frequency
                print(f"\nSending commands with motors: {motor1},{motor2} at {frequency}Hz")
                print(f"Mode active. Current state: {current_state}")
                old_settings = SerialComm.setup_non_blocking_input()
                
                # Debug mode tracking
                message_count = 0
                start_time = time.time()
                last_status_time = 0
                status_interval = 1.0
                
                try:
                    last_send_time = 0
                    last_output_time = 0
                    output_interval = 1.0
                    
                    while True:
                        current_time = time.time()
                        
                        # Check for keyboard input (state changes and motor speed changes)
                        ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
                        if ready_to_read:
                            try:
                                key = sys.stdin.read(1)
                                if key:
                                    if handle_state_input(key, old_settings):
                                        # State changed, restore non-blocking mode
                                        old_settings = SerialComm.setup_non_blocking_input()
                                    elif key == '\n':
                                        # Change motor speeds
                                        SerialComm.restore_terminal_settings(old_settings)
                                        try:
                                            new_input = input("\nEnter new motor speeds 'motor1,motor2': ").strip()
                                            if new_input:
                                                motor1, motor2 = map(int, new_input.split(','))
                                                print(f"Motor speeds changed to: {motor1},{motor2}")
                                        except ValueError:
                                            print("Invalid format, keeping current values")
                                        old_settings = SerialComm.setup_non_blocking_input()
                            except IOError:
                                pass
                        
                        # Check input completion
                        if check_input_completion():
                            old_settings = SerialComm.setup_non_blocking_input()
                        
                        # Send commands
                        if current_time - last_send_time >= send_interval:
                            effective_state, tilt_angle, friction_speed = get_effective_params()
                            
                            success = serial_comm.send_command(effective_state, motor1, motor2, tilt_angle, friction_speed)
                            if success:
                                message_count += 1
                            else:
                                print("Failed to send message")
                            
                            last_send_time = current_time
                        
                        # Print status periodically
                        if current_time - last_status_time >= status_interval:
                            elapsed_time = current_time - start_time
                            actual_frequency = message_count / elapsed_time if elapsed_time > 0 else 0
                            effective_state, tilt_angle, friction_speed = get_effective_params()
                            
                            print(f"\r{' ' * 120}", end='')
                            input_status = f" | Using prev state ({previous_state})" if waiting_for_inputs and current_state == 3 else ""
                            print(f"\rSent {message_count} | Target: {frequency:.1f}Hz | Actual: {actual_frequency:.1f}Hz | "
                                  f"State: {effective_state}, Motors: {motor1},{motor2}, Tilt: {tilt_angle}, Friction: {friction_speed}{input_status} | Input: ", end='', flush=True)
                            last_status_time = current_time
                        
                        time.sleep(0.001)
                        
                except KeyboardInterrupt:
                    elapsed_time = time.time() - start_time
                    final_frequency = message_count / elapsed_time if elapsed_time > 0 else 0
                    print(f"\n\nDebug mode stopped. Sent {message_count} messages in {elapsed_time:.2f}s")
                    print(f"Average frequency: {final_frequency:.2f}Hz")
                finally:
                    if input_thread and input_thread.is_alive():
                        input_thread.join(timeout=0.5)
                    SerialComm.restore_terminal_settings(old_settings)
            else:
                # Non-keyboard modes with standardized control
                print(f"\n--- {'Send' if send_active and not receive_active else 'Receive' if receive_active and not send_active else 'Communication'} Mode ---")
                print("0/1/2/3: Set robot state (State 3 allows tilt angle and friction speed input)")
                print("Ctrl+C: Exit program")
                print("\nNote: When selecting state 3, you'll be prompted to enter parameters.")
                print("Commands will continue with the previous state until inputs are complete.")
                print(f"\nMode active. Current state: {current_state}")
                
                # Setup non-blocking input
                old_settings = SerialComm.setup_non_blocking_input()
                
                # Standard mode defaults
                motor1, motor2 = 100, 100
                frequency = 50.0
                send_interval = 1.0 / frequency
                
                try:
                    last_send_time = 0
                    last_output_time = 0
                    output_interval = 1.0
                    
                    while True:
                        current_time = time.time()
                        
                        # Check for keyboard input (state changes)
                        ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
                        if ready_to_read:
                            try:
                                key = sys.stdin.read(1)
                                if key:
                                    if handle_state_input(key, old_settings):
                                        # State changed, restore non-blocking mode
                                        old_settings = SerialComm.setup_non_blocking_input()
                            except IOError:
                                pass
                        
                        # Check input completion
                        if check_input_completion():
                            old_settings = SerialComm.setup_non_blocking_input()
                        
                        # Send commands
                        if send_active and current_time - last_send_time >= send_interval:
                            effective_state, tilt_angle, friction_speed = get_effective_params()
                            serial_comm.send_command(effective_state, motor1, motor2, tilt_angle, friction_speed)
                            last_send_time = current_time
                        
                        # Receive messages
                        if receive_active:
                            status = serial_comm.receive_status_message(timeout=0.001)
                            if status and current_time - last_output_time >= output_interval:
                                effective_state, tilt_angle, friction_speed = get_effective_params()
                                input_status = f" | Using prev state ({previous_state})" if waiting_for_inputs and current_state == 3 else ""
                                print(f"\r{' ' * 120}", end='')
                                print(f"\rReceived status: State={status['state']}, "
                                    f"Wheels=({status['wheel1_distance']:.2f},{status['wheel2_distance']:.2f}), "
                                    f"IMU=({status['imu_x']:.2f},{status['imu_y']:.2f},{status['imu_z']:.2f}), "
                                    f"Yaw={status['imu_yaw']:.2f} | "
                                    f"Current: State={effective_state}, Tilt={tilt_angle}, Friction={friction_speed}{input_status} | Input: ", end='', flush=True)
                                last_output_time = current_time
                        time.sleep(0.001)
                        
                except KeyboardInterrupt:
                    pass        
                finally:
                    if input_thread and input_thread.is_alive():
                        input_thread.join(timeout=0.5)
                    SerialComm.restore_terminal_settings(old_settings)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Sending stop command before exit")
        serial_comm.send_command(0, 0, 0, 0)
        serial_comm.disconnect()
        print("Test completed")