cmake_minimum_required(VERSION 3.0)

project(evo)


add_definitions(-std=c++14)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)


add_executable(main src/main.cc)

target_link_libraries(main m)
