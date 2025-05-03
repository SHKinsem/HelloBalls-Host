#include "include/serial_comm.h"
#include "include/config.h"
#include <cstring>
#include <stdio.h>
#include <string>

SerialComm::SerialComm() : serialPort(-1), isConnected(false) {
    lastReconnectAttempt = std::chrono::steady_clock::now() - std::chrono::seconds(RECONNECT_INTERVAL);
}

SerialComm::~SerialComm() {
    disconnect();
}

std::vector<std::string> SerialComm::findSerialPorts() {
    std::vector<std::string> ports;
    
    // Check common serial port locations
    for (int i = 0; i < MAX_SERIAL_PORTS; i++) {
        std::string portName = "/dev/ttyUSB" + std::to_string(i);
        if (access(portName.c_str(), F_OK) == 0) {
            ports.push_back(portName);
        }
        
        portName = "/dev/ttyACM" + std::to_string(i);
        if (access(portName.c_str(), F_OK) == 0) {
            ports.push_back(portName);
        }
    }
    
    return ports;
}

bool SerialComm::tryOpenPort(const std::string& portName) {
    int port = open(portName.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    
    if (port < 0) {
        return false;
    }
    
    // Configure port settings
    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    
    // Get current attributes
    if (tcgetattr(port, &tty) != 0) {
        close(port);
        return false;
    }
    
    // Set baud rate
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    
    // 8N1 mode (8 bits, no parity, 1 stop bit)
    tty.c_cflag &= ~PARENB;  // No parity
    tty.c_cflag &= ~CSTOPB;  // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;      // 8 bits
    
    // No flow control
    tty.c_cflag &= ~CRTSCTS;
    
    // Turn on READ and ignore control lines
    tty.c_cflag |= CREAD | CLOCAL;
    
    // Turn off software flow control
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    
    // Make raw
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_oflag &= ~OPOST;
    
    // Set blocking behavior with short timeout
    tty.c_cc[VMIN] = 0;     // Non-blocking
    tty.c_cc[VTIME] = 1;    // 0.1 second timeout
    
    // Apply settings
    if (tcsetattr(port, TCSANOW, &tty) != 0) {
        close(port);
        return false;
    }
    
    // If we reached here, port is configured and ready
    std::lock_guard<std::mutex> lock(portMutex);
    serialPort = port;
    currentPortName = portName;
    isConnected = true;
    
    std::cout << "Connected to serial port: " << portName << std::endl;
    return true;
}

bool SerialComm::connect(const std::string& portName) {
    disconnect();  // Close any existing connection first
    
    // If a specific port is specified, try only that one
    if (!portName.empty()) {
        return tryOpenPort(portName);
    }
    
    // Auto-discover port
    std::vector<std::string> ports = findSerialPorts();
    
    std::cout << "Searching for available serial ports..." << std::endl;
    
    // Try each available port
    for (const auto& port : ports) {
        std::cout << "Trying port: " << port << std::endl;
        if (tryOpenPort(port)) {
            return true;
        }
    }
    
    std::cout << "No available serial ports found." << std::endl;
    return false;
}

void SerialComm::disconnect() {
    std::lock_guard<std::mutex> lock(portMutex);
    
    if (serialPort >= 0) {
        close(serialPort);
        serialPort = -1;
    }
    
    isConnected = false;
    currentPortName = "";
}

bool SerialComm::sendMotorSpeeds(int leftSpeed, int rightSpeed) {
    std::lock_guard<std::mutex> lock(portMutex);
    
    if (!isConnected || serialPort < 0) {
        return false;
    }
    
    // Format command string: "0,speed1,speed2"
    char buffer[50];
    snprintf(buffer, sizeof(buffer), COMMAND_FORMAT, leftSpeed, rightSpeed);
    
    // Add newline character for proper parsing
    strcat(buffer, "\n");
    
    // Send the command
    ssize_t bytesWritten = write(serialPort, buffer, strlen(buffer));
    
    if (bytesWritten < 0) {
        std::cerr << "Failed to send command" << std::endl;
        isConnected = false;
        return false;
    }
    
    // Flush output buffer
    tcdrain(serialPort);
    
    return true;
}

bool SerialComm::checkConnection() {
    std::lock_guard<std::mutex> lock(portMutex);
    
    // If already connected, just return true
    if (isConnected) {
        return true;
    }
    
    // Check if enough time has passed since last reconnect attempt
    auto now = std::chrono::steady_clock::now();
    auto elapsedSecs = std::chrono::duration_cast<std::chrono::seconds>(now - lastReconnectAttempt).count();
    
    if (elapsedSecs >= RECONNECT_INTERVAL) {
        lastReconnectAttempt = now;
        lock.~lock_guard();  // Release the lock before calling connect
        
        // Try to reconnect
        if (!currentPortName.empty()) {
            // Try the same port first
            if (connect(currentPortName)) {
                return true;
            }
        }
        
        // If that fails, try auto-discovery
        return connect();
    }
    
    return false;
}