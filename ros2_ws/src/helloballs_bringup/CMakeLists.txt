cmake_minimum_required(VERSION 3.8)
project(helloballs_bringup)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(camera_publisher_node src/camera_publisher_node.cpp)
target_compile_features(camera_publisher_node PUBLIC cxx_std_17)
target_include_directories(camera_publisher_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_publisher_node ${OpenCV_LIBRARIES})
ament_target_dependencies(camera_publisher_node rclcpp sensor_msgs)

install(TARGETS camera_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
