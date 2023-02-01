cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(playback)

add_definitions(-std=c++11)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

find_package(PCL COMPONENTS segmentation surface common io registration sample_consensus filters visualization REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

include_directories("01_lidar_module/src")

add_executable (environment src/environment.cpp src/render/render.cpp
                            src/processPointClouds.cpp)
target_link_libraries (environment ${PCL_LIBRARIES})