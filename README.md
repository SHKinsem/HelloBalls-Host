### Step 1: Install Required Packages

First, ensure you have the required packages installed. Open a terminal and run:

```bash
sudo apt update
sudo apt install python3-dev python3-pip cmake g++ pybind11-dev
```

### Step 2: Create the Project Structure

Create a directory structure for your project:

```bash
mkdir -p ~/Documents/HelloBalls-Host/yolo11_demo/cpp
cd ~/Documents/HelloBalls-Host/yolo11_demo/cpp
mkdir build
```

### Step 3: Write Your C++ Code

Create a file named `main.cc` in the `cpp` directory. This file will contain the C++ code you want to expose to Python.

```cpp
// filepath: ~/Documents/HelloBalls-Host/yolo11_demo/cpp/main.cc
#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(my_module, m) {
    m.def("add", &add, "A function that adds two numbers");
}
```

### Step 4: Create the CMakeLists.txt File

In the `cpp` directory, create a file named `CMakeLists.txt` to configure the build process.

```cmake
# filepath: ~/Documents/HelloBalls-Host/yolo11_demo/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.4)
project(my_module)

set(CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)

pybind11_add_module(my_module main.cc)
```

### Step 5: Build the C++ Module

Navigate to the `build` directory and run the following commands to build the C++ module:

```bash
cd ~/Documents/HelloBalls-Host/yolo11_demo/cpp/build
cmake ..
make
```

This will generate a shared library file named `my_module.so` in the `build` directory.

### Step 6: Use the C++ Module in Python

Now you can use the compiled C++ module in Python. Create a Python script in the `yolo11_demo` directory to test the module.

```python
# filepath: ~/Documents/HelloBalls-Host/yolo11_demo/test.py
import sys
import os

# Add the build directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'cpp/build'))

import my_module

result = my_module.add(3, 4)
print(f"The result of adding 3 and 4 is: {result}")
```

### Step 7: Run the Python Script

Finally, run the Python script to see if everything works correctly:

```bash
python3 ~/Documents/HelloBalls-Host/yolo11_demo/test.py
```

You should see the output:

```
The result of adding 3 and 4 is: 7
```

### Summary

You have successfully created a C++ API that can be used through Python on an Ubuntu 20.04 system. The steps included installing necessary packages, creating a project structure, writing C++ code, building the module, and testing it with a Python script. You can expand this setup by adding more functions to your C++ code and exposing them through the `PYBIND11_MODULE` macro.
