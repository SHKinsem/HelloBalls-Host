import serial
import time
import glob
import sys
import serial.tools.list_ports
import os
import termios
import fcntl
import select

class SerialComm:
    """
    Class to handle serial communication with MCU for the HelloBalls project.
    Sends messages in the format "state,motor_speed_1,motor_speed_2".
    """
    
    def __init__(self, port=None, baud_rate=115200, timeout=1, auto_reconnect=True, reconnect_interval=2):
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
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
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
            
    def send_command(self, state, motor_speed_1, motor_speed_2):
        """
        Send a command to the MCU in the format "state,motor_speed_1,motor_speed_2".
        If auto_reconnect is enabled and the connection is lost, tries to reconnect.
        
        Args:
            state (int): State value (e.g., 0 for stop, 1 for run)
            motor_speed_1 (int): Speed value for motor 1
            motor_speed_2 (int): Speed value for motor 2
            
        Returns:
            bool: True if command was sent successfully, False otherwise
        """
        # Ensure connection is active
        if not self.ensure_connection():
            print("Not connected to serial port and reconnection failed")
            return False
            
        try:
            # Format the command as "state,motor_speed_1,motor_speed_2"
            command = f"{state},{motor_speed_1},{motor_speed_2}\n"
            self.ser.write(command.encode('ascii'))
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            self.connected = False  # Mark as disconnected to trigger reconnect on next attempt
            return False
    
    def read_response(self, timeout=1.0):
        """
        Read a response from the MCU.
        If auto_reconnect is enabled and the connection is lost, tries to reconnect.
        
        Args:
            timeout (float): Maximum time to wait for a response
            
        Returns:
            str: Response from the MCU or None if no response
        """
        # Ensure connection is active
        if not self.ensure_connection():
            print("Not connected to serial port and reconnection failed")
            return None
            
        try:
            # Check if data is available to read
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                if self.ser.in_waiting > 0:
                    return self.ser.readline().decode('ascii').strip()
                time.sleep(0.01)
            return None
        except Exception as e:
            print(f"Error reading response: {e}")
            self.connected = False  # Mark as disconnected to trigger reconnect on next attempt
            return None
            
    def receive_status_message(self, timeout=1.0):
        """
        Receive and parse a status message in the format:
        "MSG,state,wheel1_distance,wheel2_distance,imu_x,imu_y,imu_z,imu_yaw"
        
        This function will filter for messages that start with "MSG" and parse
        them into a structured dictionary.
        
        Args:
            timeout (float): Maximum time to wait for a valid message
            
        Returns:
            dict: Parsed message with the following keys:
                - state: Current state of the device
                - wheel1_distance: Distance traveled by wheel 1
                - wheel2_distance: Distance traveled by wheel 2
                - imu_x: IMU X value
                - imu_y: IMU Y value 
                - imu_z: IMU Z value
                - imu_yaw: IMU Yaw value
                Or None if no valid message received
        """
        # Ensure connection is active
        if not self.ensure_connection():
            print("Not connected to serial port and reconnection failed")
            return None
            
        try:
            # Keep trying to read until we get a valid message or timeout
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('ascii').strip()
                    
                    # Check if this is a status message (starts with MSG)
                    if line.startswith("MSG"):
                        parts = line.split(',')
                        
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
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing status message: {e}")
                                print(f"Raw message: {line}")
                                # Continue trying to read another message
                        else:
                            print(f"Invalid status message format: {line}")
                            # Continue trying to read another message
                time.sleep(0.01)
            
            # Timeout reached without finding a valid message
            return None
        except Exception as e:
            print(f"Error reading status message: {e}")
            self.connected = False  # Mark as disconnected to trigger reconnect on next attempt
            return None


# For testing purposes
if __name__ == "__main__":
    print("Available ports:", SerialComm.list_available_ports())
    auto_port = SerialComm.find_port()
    print(f"Auto-detected port: {auto_port}")
    
    # Create a SerialComm instance with auto-detection and auto-reconnect
    serial_comm = SerialComm(auto_reconnect=True)
    
    try:
        # Connect to the auto-detected serial port
        if serial_comm.connect():
            print("Connected successfully. Starting test...")
            
            # Send some test commands
            print("Sending stop command (0, 0, 0)")
            serial_comm.send_command(0, 0, 0)
            time.sleep(1)
            
            # Choose mode
            mode = input("Choose mode - [1] Send at 50Hz, [2] Receive status messages, [3] Both, [4] Keyboard controller: ")
            send_active = mode in ["1", "3"]
            receive_active = mode in ["2", "3"]
            keyboard_active = mode == "4"
            
            if keyboard_active:
                print("\n--- Keyboard Controller Mode ---")
                print("W: Forward    S: Backward")
                print("A: Turn Left  D: Turn Right")
                print("R: Toggle state between 1 and 2")
                print("Space: Stop (state 0)")
                print("Q: Exit program")
                print("\nControls active. Current state: 1")
                
                # Initialize controller state
                current_state = 1
                wheel_speed = 800  # Default speed
                last_command = None  # Track last command to avoid repeating
                current_key = None  # Current active key
                key_last_read_time = 0
                key_timeout = 0.1  # Consider key released if no new input in 100ms
                
                # Keyboard detection frequency settings
                keyboard_check_interval = 0.1  # 10Hz keyboard polling
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
                    while running:
                        start_time = time.time()
                        current_time = start_time
                        
                        # Check keyboard at specified frequency (10Hz)
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
                                        else:
                                            current_key = key
                                            key_last_read_time = current_time
                                except IOError:
                                    # Handle errors from non-blocking read
                                    pass
                            
                            # Auto-release key if no new input has been received for a while
                            if current_key and current_key not in ['r'] and current_time - key_last_read_time > key_timeout:
                                # Only print key released message occasionally
                                if current_time - last_output_time >= output_interval:
                                    print("Key released - stopping movement")
                                    last_output_time = current_time
                                current_key = None
                                # Send stop command when key is released
                                serial_comm.send_command(0, 0, 0)
                                last_command = (0, 0, 0)
                                continue
                        
                        # Process the current active key (this happens at full 50Hz)
                        command = None
                        if current_key == 'r':
                            # Toggle state between 1 and 2
                            current_state = 2 if current_state == 1 else 1
                            print(f"State toggled to: {current_state}")
                            current_key = None  # Reset after toggle
                        elif current_key == ' ':
                            # Stop command (state 0)
                            command = (0, 0, 0)
                        elif current_key == 'w':
                            # Forward
                            command = (current_state, wheel_speed, wheel_speed)
                        elif current_key == 's':
                            # Backward
                            command = (current_state, -wheel_speed, -wheel_speed)
                        elif current_key == 'a':
                            # Left turn
                            command = (current_state, wheel_speed, -wheel_speed)
                        elif current_key == 'd':
                            # Right turn
                            command = (current_state, -wheel_speed, wheel_speed)
                        
                        # Send command (every frame at 50Hz while key is held)
                        if command:
                            serial_comm.send_command(*command)
                            
                            # Only print when command changes and not too frequently
                            if command != last_command and current_time - last_output_time >= output_interval:
                                last_command = command
                                last_output_time = current_time
                                
                                # Print the current command
                                action = "STOP" if command[0] == 0 else {
                                    (current_state, wheel_speed, wheel_speed): "FORWARD",
                                    (current_state, -wheel_speed, -wheel_speed): "BACKWARD",
                                    (current_state, -wheel_speed, wheel_speed): "LEFT TURN",
                                    (current_state, wheel_speed, -wheel_speed): "RIGHT TURN"
                                }.get(command, "CUSTOM")
                                
                                print(f"Action: {action} | Command: state={command[0]}, m1={command[1]}, m2={command[2]}")
                        
                        # If receiving is enabled along with keyboard
                        if receive_active:
                            status = serial_comm.receive_status_message(timeout=0.01)
                            if status and current_time - last_output_time >= output_interval:
                                print(f"Received status: State={status['state']}, "
                                      f"Wheels=({status['wheel1_distance']:.2f},{status['wheel2_distance']:.2f}), "
                                      f"IMU=({status['imu_x']:.2f},{status['imu_y']:.2f},{status['imu_z']:.2f}), "
                                      f"Yaw={status['imu_yaw']:.2f}")
                                last_output_time = current_time
                        
                        # Calculate sleep time to maintain approximately 50Hz frequency
                        elapsed = time.time() - start_time
                        sleep_time = max(0, 0.02 - elapsed)  # 0.02s = 20ms = 50Hz
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                
                finally:
                    # Restore terminal settings
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            else:
                print("Press Ctrl+C to exit")
                
                # Loop to send and/or receive messages (non-keyboard mode)
                while True:
                    start_time = time.time()
                    
                    # Send command if enabled
                    if send_active:
                        serial_comm.send_command(1, 100, 100)
                    
                    # Receive and process status message if enabled
                    if receive_active:
                        status = serial_comm.receive_status_message(timeout=0.01)  # Short timeout to maintain timing
                        if status:
                            print(f"Received status: State={status['state']}, "
                                f"Wheels=({status['wheel1_distance']:.2f},{status['wheel2_distance']:.2f}), "
                                f"IMU=({status['imu_x']:.2f},{status['imu_y']:.2f},{status['imu_z']:.2f}), "
                                f"Yaw={status['imu_yaw']:.2f}")
                    
                    # Calculate sleep time to maintain approximately 50Hz frequency
                    if send_active:
                        elapsed = time.time() - start_time
                        sleep_time = max(0, 0.02 - elapsed)  # 0.02s = 20ms = 50Hz
                        
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    else:
                        # If we're not sending, still provide some delay to not overload CPU
                        time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Always disconnect properly
        print("Sending stop command before exit")
        serial_comm.send_command(0, 0, 0)
        serial_comm.disconnect()
        print("Test completed")