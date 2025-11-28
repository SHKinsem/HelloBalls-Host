#pragma once

#include <string>
#include <vector>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <dirent.h>
#include <filesystem>
#include <mutex>

class SerialComm {
private:
    int serialPort;               // Serial port file descriptor
    std::string currentPortName;  // Current port name
    bool isConnected;             // Connection status flag
    std::chrono::steady_clock::time_point lastReconnectAttempt;
    std::mutex portMutex;         // Mutex for thread-safe operations
    
    // Find available serial ports
    std::vector<std::string> findSerialPorts();
    
    // Try to open a specific port
    bool tryOpenPort(const std::string& portName);

public:
    // Constructor and destructor
    SerialComm();
    ~SerialComm();
    
    // Connect to a specific port or auto-discover
    bool connect(const std::string& portName = "");
    
    // Disconnect from the port
    void disconnect();
    
    // Send motor speeds command
    bool sendMotorSpeeds(int leftSpeed, int rightSpeed);
    
    // Check if connected and attempt reconnection if needed
    bool checkConnection();
    
    // Get connection status
    bool connected() const { return isConnected; }
    
    // Get current port name
    std::string getPortName() const { return currentPortName; }
};