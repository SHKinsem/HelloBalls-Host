cmake_minimum_required(VERSION 3.10)

project(rdk_yolov11_detect)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Python and pybind11
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Set compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -O3 ")

# Enable link time optimization
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-flto" HAS_FLTO)
if(HAS_FLTO)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
endif()

# Default to Release build if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Define DNN lib path
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")

# Include directories - Adding the include directory properly
include_directories(
    ${DNN_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PYTHON_INCLUDE_DIRS}
)
link_directories(${DNN_LIB_PATH})

# Source files
set(SOURCES 
    main.cc
    src/model.cc
    src/utils.cc
    src/preprocess.cc
    src/postprocess.cc
    src/pid_controller.cc
)

# Build shared library for Python bindings
pybind11_add_module(yolo11_api 
    bindings/yolo11_api_bindings.cpp
    src/model.cc 
    src/pid_controller.cc 
    src/postprocess.cc 
    src/preprocess.cc 
    src/utils.cc
)

target_include_directories(yolo11_api PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo11_api PRIVATE ${OpenCV_LIBS} dnn)

# Build executable
add_executable(main ${SOURCES})

# Fix unused variable warning in main.cc
set_target_properties(main PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-error=unused-variable")

target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(main PRIVATE ${OpenCV_LIBS} dnn)

install(TARGETS main RUNTIME DESTINATION bin)