cmake_minimum_required(VERSION 3.5)
project(ukf)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp ukf.cpp ukf.h measurement_package.h ground_truth_package.h)
add_executable(ukf ${SOURCE_FILES})