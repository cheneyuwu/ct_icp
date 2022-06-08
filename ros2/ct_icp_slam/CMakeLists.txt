cmake_minimum_required(VERSION 3.16)
project(ct_icp_slam)

# Compile as C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Turn on as many warnings as possible by default.
add_compile_options(-march=native -O3 -pthread -Wall -Wextra)

## Find dependencies
find_package(ament_cmake REQUIRED)

find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

set(EXT_INSTALL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake-build-${CMAKE_BUILD_TYPE}/external/install/${CMAKE_BUILD_TYPE})
include(../../cmake/external.cmake)
include(../../cmake/test.cmake)
find_package(ct_icp REQUIRED)

## C++ Libraries

## Executables
include_directories(
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(${PROJECT_NAME} src/slam.cpp)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp tf2 tf2_ros tf2_eigen nav_msgs
  pcl_conversions pcl_ros
  ct_icp
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
