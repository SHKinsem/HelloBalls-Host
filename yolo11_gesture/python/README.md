### Step 1: Install Required Packages

First, ensure you have the necessary packages installed. You will need `cmake`, `g++`, and `pybind11`. You can install them using the following commands:

```bash
sudo apt update
sudo apt install cmake g++ python3-dev
pip install pybind11
```

### Step 2: Create the Project Structure

Create the following directory structure for your project:

```
HelloBalls-Host/
├── CMakeLists.txt
├── build.sh
├── run_cpp.sh
└── yolo11_demo/
    ├── cpp/
    │   ├── CMakeLists.txt
    │   ├── main.cc
    │   └── bindings.cpp
```

### Step 3: Write the C++ Code

1. **main.cc**: This file contains your main C++ code. For demonstration, let's assume it has a simple function.

```cpp
// filepath: /home/sunrise/Documents/HelloBalls-Host/yolo11_demo/cpp/main.cc
#include <iostream>

extern "C" {
    void hello() {
        std::cout << "Hello from C++!" << std::endl;
    }
}
```

2. **bindings.cpp**: This file will contain the bindings for the C++ functions you want to expose to Python.

```cpp
// filepath: /home/sunrise/Documents/HelloBalls-Host/yolo11_demo/cpp/bindings.cpp
#include <pybind11/pybind11.h>

extern "C" {
    void hello();
}

namespace py = pybind11;

PYBIND11_MODULE(hello_module, m) {
    m.def("hello", &hello, "A function that prints Hello from C++");
}
```

### Step 4: Create CMakeLists.txt Files

1. **CMakeLists.txt** for the main project:

```cmake
# filepath: /home/sunrise/Documents/HelloBalls-Host/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(HelloBalls)

add_subdirectory(yolo11_demo/cpp)
```

2. **CMakeLists.txt** for the C++ code:

```cmake
# filepath: /home/sunrise/Documents/HelloBalls-Host/yolo11_demo/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(yolo11_demo)

find_package(pybind11 REQUIRED)

pybind11_add_module(hello_module bindings.cpp main.cc)
```

### Step 5: Build the Project

Now, you can build the project using the `build.sh` script. Make sure it is executable:

```bash
chmod +x build.sh
```

Run the build script:

```bash
./build.sh
```

### Step 6: Use the C++ API in Python

After building, you should have a shared library file named `hello_module.so` in the `yolo11_demo/cpp/build` directory. You can now use this module in Python.

1. Open a Python shell or create a new Python script:

```python
# test.py
import sys
import os

# Add the path to the directory containing the compiled module
sys.path.append('/home/sunrise/Documents/HelloBalls-Host/yolo11_demo/cpp/build')

import hello_module

hello_module.hello()
```

2. Run the Python script:

```bash
python3 test.py
```

You should see the output:

```
Hello from C++!
```

### Summary

You have successfully created a C++ API that can be used in Python on an Ubuntu 20.04 system. The steps included setting up the project structure, writing the C++ code, creating the necessary CMake files, building the project, and using the API in Python.