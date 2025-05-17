import serial
import time
import glob
import sys
import serial.tools.list_ports
import os
import termios
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
            # Flush to ensure immediate transmission
            self.ser.flush()
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
            print("Sending stop command (0, 0, 0)")
            serial_comm.send_command(0, 0, 0)
            time.sleep(0.1)  # Reduced sleep time
            
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
                # Track key state separately from key input to maintain continuous movement
                active_movement_key = None  # Key that 's actually controlling movement
                
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
                                        else:
                                            # Non-movement keys like 'r'
                                            current_key = key
                                            key_last_read_time = current_time
                                except IOError:
                                    # Handle errors from non-blocking read
                                    pass
                            
                            # Auto-release key if no new input has been received for a while
                            # This only affects the current_key, not active_movement_key
                            if current_key and current_key not in ['r'] and current_time - key_last_read_time > key_timeout:
                                # Only print key released message occasionally and only if we're actually changing movement
                                if active_movement_key and current_time - last_output_time >= output_interval:
                                    print("Key released - stopping movement")
                                    last_output_time = current_time
                                current_key = None
                                active_movement_key = None  # Stop movement when key is released
                                # Send stop command when movement key is released
                                serial_comm.send_command(0, 0, 0)
                                last_command = (0, 0, 0)
                        
                        # Process serial communications at 50Hz 
                        # (decoupled from keyboard input for more responsive controls)
                        send_serial = current_time - last_serial_time >= serial_interval
                        
                        if send_serial:
                            last_serial_time = current_time
                            
                            # Process the current active key
                            command = None
                            if current_key == 'r':
                                # Toggle state between 1 and 2
                                current_state = 2 if current_state == 1 else 1
                                print(f"State toggled to: {current_state}")
                                current_key = None  # Reset after toggle
                            elif active_movement_key == ' ':  # Use active_movement_key for movement commands
                                # Stop command (state 0)
                                command = (0, 0, 0)
                            elif active_movement_key == 'w':
                                # Forward
                                command = (current_state, wheel_speed, wheel_speed)
                            elif active_movement_key == 'a':
                                # Left turn
                                command = (current_state, wheel_speed//2, -wheel_speed//2)
                            elif active_movement_key == 'd':
                                # Right turn
                                command = (current_state, -wheel_speed//2, wheel_speed//2)
                            elif active_movement_key == 's':
                                # Backward
                                command = (current_state, -wheel_speed, -wheel_speed)
                            
                            # Send command at 50Hz while key is held
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
                            
                            # If receiving is enabled along with keyboard, do it at the same frequency as sending
                            if receive_active:
                                status = serial_comm.receive_status_message(timeout=0.001)  # Very short timeout
                                if status and current_time - last_output_time >= output_interval:
                                    print(f"Received status: State={status['state']}, "
                                        f"Wheels=({status['wheel1_distance']:.2f},{status['wheel2_distance']:.2f}), "
                                        f"IMU=({status['imu_x']:.2f},{status['imu_y']:.2f},{status['imu_z']:.2f}), "
                                        f"Yaw={status['imu_yaw']:.2f}")
                                    last_output_time = current_time
                        
                        # Tiny sleep to prevent CPU hogging but maintain responsiveness
                        time.sleep(0.001)
                
                finally:
                    # Restore terminal settings
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            else:
                print("Press Ctrl+C to exit")
                
                # Loop to send and/or receive messages (non-keyboard mode)
                last_send_time = 0
                send_interval = 0.02  # 50Hz
                
                while True:
                    current_time = time.time()
                    
                    # Send command at consistent 50Hz rate
                    if send_active and current_time - last_send_time >= send_interval:
                        serial_comm.send_command(1, 100, 100)
                        last_send_time = current_time
                    
                    # Receive and process status message if enabled (do this more frequently)
                    if receive_active:
                        status = serial_comm.receive_status_message(timeout=0.001)  # Very short timeout
                        if status:
                            print(f"Received status: State={status['state']}, "
                                f"Wheels=({status['wheel1_distance']:.2f},{status['wheel2_distance']:.2f}), "
                                f"IMU=({status['imu_x']:.2f},{status['imu_y']:.2f},{status['imu_z']:.2f}), "
                                f"Yaw={status['imu_yaw']:.2f}")
                    
                    # Small sleep to prevent CPU hogging
                    time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Always disconnect properly
        print("Sending stop command before exit")
        serial_comm.send_command(0, 0, 0)
        serial_comm.disconnect()
        print("Test completed")