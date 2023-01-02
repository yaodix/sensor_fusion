cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(playback)

add_definitions(-std=c++11)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(PCL COMPONENTS segmentation surface common io registration sample_consensus filters visualization REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

include_directories("lidar_module/src")

# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})
# list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")


add_executable (environment src/environment.cpp src/render/render.cpp
                            src/processPointClouds.cpp)
target_link_libraries (environment ${PCL_LIBRARIES})