cmake_minimum_required(VERSION 3.18)

find_package(MPI REQUIRED)

set(testname ${PROJECT_NAME}_tests)
set(testdata testdata)

add_executable(
    ${testname}
    field_tests.cpp
    integration_tests.cpp
    matrix_tests.cpp
    utilities_tests.cpp
)

target_include_directories(${testname} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    ${common_dir}
    )

target_include_directories(${testname} SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/googletest-1.14.0/googletest/include"
    "${CMAKE_SOURCE_DIR}/include"
    ${MPI_C_INCLUDE_DIRS}
    )

target_link_libraries(
  ${testname}
  GTest::gtest_main
  ${PROJECT_NAME}
)

add_subdirectory(googletest-1.14.0)

include(GoogleTest)
gtest_discover_tests(${testname})

# Add symbolic link from tests/testdata in source directory to build directory
ADD_CUSTOM_TARGET(link_target ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/${testdata}"
    "${CMAKE_CURRENT_BINARY_DIR}/${testdata}"
    )
