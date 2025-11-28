cmake_minimum_required(VERSION 3.10)
project(gesture_yolo_api LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# 你的头文件就在 /usr/include/dnn/hb_dnn.h
include_directories(/usr/include)        # 包含 dnn/hb_dnn.h 与 hb_sys.h

# 常见库目录（按需保留）
link_directories(/usr/lib /usr/lib/hbbpu /lib/aarch64-linux-gnu)

add_compile_definitions(PYBIND11_DETAILED_ERROR_MESSAGES)
pybind11_add_module(gesture_yolo_api gesture_yolo_api.cpp)

# DNN 运行库：大多数板子是 dnn + hbrt（S/J 系列）
target_link_libraries(gesture_yolo_api PRIVATE
    ${OpenCV_LIBS}
    dnn
    hbrt
)

# 如果链接报 -lhbrt 找不到，换成 hbrt_bernoulli_aarch64（取决于你的板子）
# target_link_libraries(gesture_yolo_api PRIVATE ${OpenCV_LIBS} dnn hbrt_bernoulli_aarch64)
