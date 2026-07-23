cmake_minimum_required(VERSION 3.8)
project(helloballs_bringup)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(camera_publisher_node src/camera_publisher_node.cpp)
target_compile_features(camera_publisher_node PUBLIC cxx_std_17)
target_include_directories(camera_publisher_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_publisher_node ${OpenCV_LIBRARIES})
ament_target_dependencies(camera_publisher_node rclcpp sensor_msgs)

add_executable(grayscale_converter_node src/grayscale_converter_node.cpp)
target_compile_features(grayscale_converter_node PUBLIC cxx_std_17)
target_include_directories(grayscale_converter_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(grayscale_converter_node ${OpenCV_LIBRARIES})
ament_target_dependencies(grayscale_converter_node rclcpp sensor_msgs)

add_executable(court_line_localizer_node src/court_line_localizer_node.cpp)
target_compile_features(court_line_localizer_node PUBLIC cxx_std_17)
target_include_directories(court_line_localizer_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(court_line_localizer_node ${OpenCV_LIBRARIES})
ament_target_dependencies(
  court_line_localizer_node
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  visualization_msgs
)

add_executable(court_vio_fusion_node src/court_vio_fusion_node.cpp)
target_compile_features(court_vio_fusion_node PUBLIC cxx_std_17)
ament_target_dependencies(
  court_vio_fusion_node
  geometry_msgs
  nav_msgs
  rclcpp
  tf2_ros
)

install(TARGETS
  camera_publisher_node
  court_line_localizer_node
  court_vio_fusion_node
  grayscale_converter_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
