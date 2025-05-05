# Ball Detector Project

This project provides a Python interface for a C++ ball detection system using YOLO (You Only Look Once) model. The C++ code is designed for real-time camera inference and is wrapped in Python for ease of use.

## Project Structure

```
ball-detector-py
├── cpp
│   ├── bindings.cpp        # Pybind11 bindings for the C++ code
│   ├── CMakeLists.txt      # CMake configuration for building bindings
│   └── include             # Symbolic links or copies of original headers
├── python
│   ├── __init__.py
│   ├── ball_detector.py    # Python wrapper for the C++ functionality
│   ├── camera_utils.py     # Camera handling utilities
│   └── visualization.py     # Tools for visualizing detection results
├── examples
│   ├── simple_detection.py
│   └── video_processing.py
├── setup.py                # Python package setup file
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- Pybind11
- OpenCV
- CMake

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ball-detector-py
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the C++ bindings:**
   - Navigate to the `cpp` directory:
     ```bash
     cd cpp
     ```
   - Create a build directory and compile:
     ```bash
     mkdir build
     cd build
     cmake ..
     make
     ```

4. **Run the examples:**
   - Navigate to the `examples` directory:
     ```bash
     cd ../../examples
     ```
   - Run a simple detection example:
     ```bash
     python simple_detection.py
     ```

## Usage

The `ball_detector.py` module provides a high-level interface for loading the C++ model and performing ball detection. You can use the provided examples to understand how to integrate the functionality into your own applications.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Apache License, Version 2.0. See the LICENSE file for more details.