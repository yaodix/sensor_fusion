cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(ukf)

add_definitions(-std=c++11)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable (ukf 05_UPDATE_radar_measurement_prediction/main.cpp
                    05_UPDATE_radar_measurement_prediction/ukf.cpp)
