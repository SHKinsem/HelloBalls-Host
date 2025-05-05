cmake_minimum_required(VERSION 3.4)
project(ball_detector_cpp)

# Find packages
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# Define DNN paths (same as in the main C++ project)
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")

# Include directories
include_directories(
    ${DNN_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp/include
)

link_directories(${DNN_LIB_PATH})

# Add the pybind11 module
pybind11_add_module(ball_detector_cpp ball_detector_bindings.cpp
    ../../cpp/src/model.cc
    ../../cpp/src/utils.cc
    ../../cpp/src/preprocess.cc
    ../../cpp/src/postprocess.cc
)

# Link libraries
target_link_libraries(ball_detector_cpp PRIVATE
    ${OpenCV_LIBS}
    dnn
    pthread
    rt
    dl
)