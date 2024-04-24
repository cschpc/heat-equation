cmake_minimum_required(VERSION 3.18)

project(heat_tested VERSION 1.0 DESCRIPTION "Heat equation w/ cmake, ctest and gtest" LANGUAGES CXX C)

set(common_dir "${CMAKE_SOURCE_DIR}/../common")
set(install_dir "${PROJECT_NAME}")

add_subdirectory(src)
