cmake_minimum_required(VERSION 3.20)

project(Heat LANGUAGES CXX C)

find_package(Kokkos CONFIG REQUIRED)
find_package(MPI REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(default_build_type "Release")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    message(STATUS "Release build with optimizations: ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

include_directories(../../common)

add_executable(heat 
    matrix.hpp
    parallel.hpp
    heat.hpp
    functions.hpp
    ../../common/pngwriter.h
    heat.cpp
    main.cpp
    io.cpp
    setup.cpp
    core.cpp
    utilities.cpp
    ../../common/pngwriter.c
)

target_link_libraries(heat PRIVATE Kokkos::kokkos)
target_link_libraries(heat PRIVATE MPI::MPI_CXX)
