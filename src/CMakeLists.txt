cmake_minimum_required(VERSION 3.14)
project(ct_icp)

set(PROJECT_VERSION 0.0.0)

option(WITH_VIZ3D "Search and install viz3d (disabled by default)" OFF)
option(WITH_PYTHON_BINDING "Install Python bindings" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-march=native -O3)  # needed by steam and lgmath

set(EXT_INSTALL_ROOT ${CMAKE_BINARY_DIR}/../external/install/${CMAKE_BUILD_TYPE})
include(../cmake/external.cmake)
include(../cmake/test.cmake)
find_package(OpenMP REQUIRED)
find_package(lgmath REQUIRED)
find_package(steam REQUIRED)

add_subdirectory(ct_icp)
if (WITH_PYTHON_BINDING)
  add_subdirectory(binding)
endif()

add_executable(slam slam.cpp)
target_link_libraries(slam PUBLIC ct_icp yaml-cpp TCLAP Ceres::ceres glog::glog Eigen3::Eigen OpenMP::OpenMP_CXX lgmath steam)
if (WITH_VIZ3D)
    target_link_libraries(slam PUBLIC viz3d)
    target_compile_definitions(slam PUBLIC CT_ICP_WITH_VIZ=1)
endif ()

set(CT_ICP_RUNTIME_DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
install(TARGETS slam DESTINATION ${CT_ICP_RUNTIME_DESTINATION})
