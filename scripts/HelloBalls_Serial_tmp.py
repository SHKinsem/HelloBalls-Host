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
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_ODD,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Set non-blocking mode
            self.ser.nonblocking = True
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
            self.ser.write(command.encode('ascii'))
            4# Remove flush() for non-blocking operation - let OS buffer handle transmission
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
            
            if keyboard_active:
                print("\n--- Keyboard Controller Mode ---")
                print("W: Forward    S: Backward")
                print("A: Turn Left  D: Turn Right")
                print("0/1/2/3: Set robot state (State 3 allows tilt angle input)")
                print("Space: Stop motors")
                print("Q: Exit program")
                print("\nNote: When selecting state 3, you'll be prompted to enter a tilt angle and friction wheel speed.")
                print("The robot will continue using the previous state until both inputs are complete.")
                print("The tilt angle and friction wheel speed will then be used for all movement commands in state 3.")
                print("Other states (0, 1, 2) will use tilt angle 0 and friction wheel speed 0.")
                print("\nControls active. Current state: 1")
                
                # Initialize controller state
                current_state = 1
                wheel_speed = 2000  # Default speed
                current_tilt_angle = 0  # Track current tilt angle for state 3
                current_friction_wheel_speed = 0  # Track current friction wheel speed for state 3
                last_command = None  # Track last command to avoid repeating
                current_key = None  # Current active key
                key_last_read_time = 0
                key_timeout = 0.1  # Consider key released if no new input in 100ms
                # Track key state separately from key input to maintain continuous movement
                active_movement_key = None  # Key that 's actually controlling movement
                
                # Tilt angle and friction wheel speed input handling
                waiting_for_inputs = False
                input_queue = queue.Queue()
                input_thread = None
                previous_state = current_state  # Track previous state for state 3 transition
                
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
                    
                    print("Returning to keyboard control mode...")
                
                # Keyboard detection settings - increase frequency
                keyboard_check_interval = 0.005  # 200Hz keyboard polling for better responsiveness
                last_keyboard_check_time = 0
                
                # Reduce terminal output
                output_interval = 1.0  # Only output status every 1 second
                last_output_time = 0
                
                # Save terminal settings
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    # Set terminal to raw mode
                    tty_settings = termios.tcgetattr(sys.stdin)
                    tty_settings[3] = tty_settings[3] & ~(termios.ECHO | termios.ICANON)
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, tty_settings)
                    
                    # Set non-blocking
                    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, os.O_NONBLOCK)
                    
                    running = True
                    last_serial_time = 0
                    serial_interval = 0.02  # 50Hz for serial communication
                    
                    while running:
                        current_time = time.time()
                        
                        # Check keyboard at higher frequency than serial commands
                        check_keyboard = current_time - last_keyboard_check_time >= keyboard_check_interval
                        
                        if check_keyboard:
                            last_keyboard_check_time = current_time
                            
                            # Non-blocking check for keypresses
                            ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
                            
                            # If there's input available, read it and update current_key
                            if ready_to_read:
                                try:
                                    key = sys.stdin.read(1)
                                    if key:  # Only update if we got a valid key
                                        if key == 'q':
                                            running = False
                                            current_key = None  # Clear current key
                                            active_movement_key = None  # Clear active movement
                                        elif key in ['w', 'a', 's', 'd', ' ']:
                                            # Movement keys - update both current key and active movement key
                                            current_key = key
                                            active_movement_key = key
                                            key_last_read_time = current_time
                                        elif key in ['0', '1', '2', '3']:
                                            # State keys - update current key only
                                            current_key = key
                                            key_last_read_time = current_time
                                        else:
                                            # Other non-movement keys
                                            current_key = key
                                            key_last_read_time = current_time
                                except IOError:
                                    # Handle errors from non-blocking read
                                    pass
                            
                            # Auto-release key if no new input has been received for a while
                            # This only affects the current_key, not active_movement_key
                            if current_key and current_key not in ['0', '1', '2', '3'] and current_time - key_last_read_time > key_timeout:
                                # Only print key released message occasionally and only if we're actually changing movement
                                if active_movement_key and current_time - last_output_time >= output_interval:
                                    # Update the same line without creating new lines
                                    print(f"\r{' ' * 80}", end='')  # Clear the line first
                                    print("\rKey released - stopping movement | Input: ", end='', flush=True)
                                    last_output_time = current_time
                                current_key = None
                                active_movement_key = None  # Stop movement when key is released
                        
                        # Process serial communications at 50Hz 
                        # (decoupled from keyboard input for more responsive controls)
                        send_serial = current_time - last_serial_time >= serial_interval
                        
                        if send_serial:
                            last_serial_time = current_time
                            
                            # Process the current active key
                            command = None
                            if current_key in ['0', '1', '2', '3']:
                                # State change keys
                                new_state = int(current_key)
                                if new_state != current_state:
                                    previous_state = current_state  # Save current state before changing
                                    current_state = new_state
                                    print(f"State changed to: {current_state}")
                                    
                                    # Special handling for state 3 - prompt for tilt angle and friction wheel speed in separate thread
                                    if current_state == 3 and not waiting_for_inputs:
                                        waiting_for_inputs = True
                                        print(f"Continuing with previous state ({previous_state}) while waiting for inputs...")
                                        # Start input thread to get tilt angle and friction wheel speed without blocking serial communication
                                        input_thread = threading.Thread(
                                            target=get_state3_inputs_threaded,
                                            args=(current_tilt_angle, current_friction_wheel_speed, input_queue, old_settings),
                                            daemon=True
                                        )
                                        input_thread.start()
                                        
                                current_key = None  # Reset after state change
                            
                            # Check for input completion (non-blocking)
                            if waiting_for_inputs:
                                try:
                                    result_type, result_tilt, result_friction = input_queue.get_nowait()
                                    current_tilt_angle = result_tilt
                                    current_friction_wheel_speed = result_friction
                                    waiting_for_inputs = False
                                    print(f"Input complete. Now using state {current_state} with tilt angle {current_tilt_angle} and friction speed {current_friction_wheel_speed}")
                                    
                                    # Restore raw mode for keyboard controls after input thread completes
                                    tty_settings = termios.tcgetattr(sys.stdin)
                                    tty_settings[3] = tty_settings[3] & ~(termios.ECHO | termios.ICANON)
                                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, tty_settings)
                                    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, os.O_NONBLOCK)
                                    
                                except queue.Empty:
                                    # No input ready yet, continue with normal operation
                                    pass
                            
                            # Always send a command to maintain 50Hz communication
                            # Determine which state and parameters to use
                            if waiting_for_inputs and current_state == 3:
                                # Use previous state while waiting for inputs
                                effective_state = previous_state
                                tilt_angle = 0  # Use 0 tilt angle for previous state
                                friction_speed = 0  # Use 0 friction speed for previous state
                            else:
                                # Use current state
                                effective_state = current_state
                                tilt_angle = current_tilt_angle if current_state == 3 else 0
                                friction_speed = current_friction_wheel_speed if current_state == 3 else 0
                            
                            if active_movement_key == ' ':  # Use active_movement_key for movement commands
                                # Stop motors but keep current state
                                command = (effective_state, 0, 0, tilt_angle, friction_speed)
                            elif active_movement_key == 'w':
                                # Forward
                                command = (effective_state, wheel_speed, wheel_speed, tilt_angle, friction_speed)
                            elif active_movement_key == 'a':
                                # Left turn
                                command = (effective_state, wheel_speed//2, -wheel_speed//2, tilt_angle, friction_speed)
                            elif active_movement_key == 'd':
                                # Right turn
                                command = (effective_state, -wheel_speed//2, wheel_speed//2, tilt_angle, friction_speed)
                            elif active_movement_key == 's':
                                # Backward
                                command = (effective_state, -wheel_speed, -wheel_speed, tilt_angle, friction_speed)
                            else:
                                # No movement key pressed - send 0 speed to maintain communication
                                command = (effective_state, 0, 0, tilt_angle, friction_speed)
                            
                            # Send command at 50Hz (always send to maintain communication)
                            if command:
                                serial_comm.send_command(*command)
                                
                                # Only print when command changes and not too equently
                                if command != last_command and current_time - last_output_time >= output_interval:
                                    last_command = command
                                    last_output_time = current_time
                                    
                                    # Print the current command
                                    if command[1] == 0 and command[2] == 0:
                                        action = "STOP"
                                    else:
                                        action = {
                                            (wheel_speed, wheel_speed): "FORWARD",
                                            (-wheel_speed, -wheel_speed): "BACKWARD",
                                            (wheel_speed//2, -wheel_speed//2): "LEFT TURN",
                                            (-wheel_speed//2, wheel_speed//2): "RIGHT TURN"
                                        }.get((command[1], command[2]), "CUSTOM")
                                    
                                    # Update the same line without creating new lines
                                    print(f"\r{' ' * 150}", end='')  # Clear the line first
                                    tilt_info = f" | Tilt: {command[3]}" if current_state == 3 and not waiting_for_inputs else ""
                                    friction_info = f" | Friction: {command[4]}" if current_state == 3 and not waiting_for_inputs else ""
                                    if waiting_for_inputs and current_state == 3:
                                        input_status = f" | Using prev state ({previous_state}) - waiting for inputs..."
                                    else:
                                        input_status = ""
                                    print(f"\rAction: {action} | Command: state={command[0]}, m1={command[1]}, m2={command[2]}, tilt={command[3]}, friction={command[4]}{tilt_info}{friction_info}{input_status} | Input: ", end='', flush=True)
                        
                        # If receiving is enabled along with keyboard, do it at the same frequency as sending
                        if receive_active:
                            status = serial_comm.receive_status_message(timeout=0.001)  # Very short timeout
                            if status and current_time - last_output_time >= output_interval:
                                # Update the same line without creating new lines
                                print(f"\r{' ' * 150}", end='')  # Clear the line first
                                print(f"\rReceived status: State={status['state']}, "
                                    f"Wheels=({status['wheel1_distance']:.2f},{status['wheel2_distance']:.2f}), "
                                    f"IMU=({status['imu_x']:.2f},{status['imu_y']:.2f},{status['imu_z']:.2f}), "
                                    f"Yaw={status['imu_yaw']:.2f} | Input: ", end='', flush=True)
                                last_output_time = current_time
                        
                        # Tiny sleep to prevent CPU hogging but maintain responsiveness
                        time.sleep(0.001)
                
                finally:
                    # Clean up any running input thread
                    if input_thread and input_thread.is_alive():
                        # Give the thread a short time to finish
                        input_thread.join(timeout=0.5)
                    
                    # Restore terminal settings
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    # Print a newline to ensure terminal prompt appears on new line
                    print()
            elif debug_active:
                print("\n--- Debug Mode ---")
                print("Send custom messages at specified frequency")
                print("Enter message in format 'state,motor1,motor2,tilt_angle' (e.g., '1,100,100,0')")
                print("You can change the message during runtime by pressing Enter and typing a new one")
                
                # Get initial message and frequency
                try:
                    message_input = input("Enter initial message (default '0,0,0,0'): ").strip() or "0,0,0,0"
                    parts = message_input.split(',')
                    if len(parts) != 4:
                        raise ValueError("Message must have exactly 4 comma-separated values")
                    state, motor1, motor2, tilt_angle = map(int, parts)
                    frequency = float(input("Enter frequency in Hz (default 50): ") or "50")
                except ValueError as e:
                    print(f"Invalid input ({e}), using defaults: 0,0,0,0 at 50Hz")
                    state, motor1, motor2, tilt_angle, frequency = 0, 0, 0, 0, 50.0
                
                send_interval = 1.0 / frequency
                print(f"\nSending messages: {state},{motor1},{motor2},{tilt_angle} at {frequency}Hz")
                print("Press Enter to change message, Ctrl+C to exit")
                
                # Set stdin to non-blocking mode for runtime message changes
                import fcntl
                import os
                old_flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
                fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
                
                last_send_time = 0
                message_count = 0
                start_time = time.time()
                last_status_time = 0
                status_interval = 1.0  # Print status every second
                
                try:
                    while True:
                        current_time = time.time()
                        
                        # Check for user input to change message
                        try:
                            # Non-blocking read from stdin
                            user_input = sys.stdin.readline().strip()
                            if user_input:
                                try:
                                    parts = user_input.split(',')
                                    if len(parts) == 4:
                                        state, motor1, motor2, tilt_angle = map(int, parts)
                                        print(f"Message changed to: {state},{motor1},{motor2},{tilt_angle}")
                                    elif len(parts) == 1 and user_input.replace('.', '').isdigit():
                                        # If single number, treat as frequency change
                                        frequency = float(user_input)
                                        send_interval = 1.0 / frequency
                                        print(f"Frequency changed to: {frequency}Hz")
                                    else:
                                        print("Invalid format. Use 'state,motor1,motor2,tilt_angle' or just frequency number")
                                except ValueError:
                                    print("Invalid values. Use integers for message, float for frequency")
                        except IOError:
                            # No input available, continue
                            pass
                        
                        # Send message at specified frequency
                        if current_time - last_send_time >= send_interval:
                            success = serial_comm.send_command(state, motor1, motor2, tilt_angle)
                            if success:
                                message_count += 1
                            else:
                                print("Failed to send message")
                            
                            last_send_time = current_time
                        
                        # Print status periodically
                        if current_time - last_status_time >= status_interval:
                            elapsed_time = current_time - start_time
                            actual_frequency = message_count / elapsed_time if elapsed_time > 0 else 0
                            # Update the same line without creating new lines
                            print(f"\r{' ' * 120}", end='')  # Clear the line first
                            print(f"\rSent {message_count} messages | "
                                  f"Target: {frequency:.1f}Hz | "
                                  f"Actual: {actual_frequency:.1f}Hz | "
                                  f"Current: {state},{motor1},{motor2},{tilt_angle} | Input: ", end='', flush=True)
                            last_status_time = current_time
                        
                        # Small sleep to prevent CPU hogging
                        time.sleep(0.001)
                        
                except KeyboardInterrupt:
                    elapsed_time = time.time() - start_time
                    final_frequency = message_count / elapsed_time if elapsed_time > 0 else 0
                    print(f"\n\nDebug mode stopped. Sent {message_count} messages in {elapsed_time:.2f}s")
                    print(f"Average frequency: {final_frequency:.2f}Hz")
                finally:
                    # Restore stdin flags
                    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, old_flags)
            else:
                print("Press Ctrl+C to exit")
                
                # Loop to send and/or receive messages (non-keyboard mode)
                last_send_time = 0
                send_interval = 0.02  # 50Hz
                
                while True:
                    current_time = time.time()
                    
                    # Send command at consistent 50Hz rate
                    if send_active and current_time - last_send_time >= send_interval:
                        serial_comm.send_command(1, 100, 100, 0)
                        last_send_time = current_time
                    
                    # Receive and process status message if enabled (do this more frequently)
                    if receive_active:
                        status = serial_comm.receive_status_message(timeout=0.001)  # Very short timeout
                        if status:
                            # Update the same line without creating new lines
                            print(f"\r{' ' * 120}", end='')  # Clear the line first
                            print(f"\rReceived status: State={status['state']}, "
                                f"Wheels=({status['wheel1_distance']:.2f},{status['wheel2_distance']:.2f}), "
                                f"IMU=({status['imu_x']:.2f},{status['imu_y']:.2f},{status['imu_z']:.2f}), "
                                f"Yaw={status['imu_yaw']:.2f} | Input: ", end='', flush=True)
                    
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Always disconnect properly
        print("Sending stop command before exit")
        serial_comm.send_command(0, 0, 0, 0)
        serial_comm.disconnect()
        print("Test completed")