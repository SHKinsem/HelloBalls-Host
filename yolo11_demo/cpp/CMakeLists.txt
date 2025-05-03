cmake_minimum_required(VERSION 2.8)

project(rdk_yolov11_detect)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -Werror -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -Werror -O3 ")

# Default to Release build if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Define DNN lib path
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")

# Include directories
include_directories(${DNN_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${DNN_LIB_PATH})

# Source files
set(SOURCES 
    main.cc
    src/model.cc
    src/utils.cc
    src/preprocess.cc
    src/postprocess.cc
    src/pid_controller.cc
    src/serial_comm.cc
)

# Build executable
add_executable(main ${SOURCES})

# Link libraries
target_link_libraries(main
                     ${OpenCV_LIBS}
                     dnn
                     pthread
                     rt
                     dl)