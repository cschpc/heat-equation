cmake_minimum_required(VERSION 3.18)

project(heattests VERSION 1.0 DESCRIPTION "Heat equation w/ cmake, ctest and gtest" LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(common_dir "${CMAKE_SOURCE_DIR}/../common")
set(install_dir "${PROJECT_NAME}")

add_subdirectory(src)

option(package_tests "Build tests" ON)
if(package_tests)
    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()
