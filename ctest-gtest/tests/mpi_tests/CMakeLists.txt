cmake_minimum_required(VERSION 3.18)

find_package(MPI REQUIRED)

set(testname ${PROJECT_NAME}_mpi_tests)

add_executable(
    ${testname}
    full_run_tests.cpp
)

target_include_directories(${testname} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    ${common_dir}
    )

target_include_directories(${testname} SYSTEM PRIVATE
    "${googletest_include}"
    "${CMAKE_SOURCE_DIR}/include"
    ${MPI_C_INCLUDE_DIRS}
    )

target_link_libraries(
  ${testname}
  GTest::gtest_main
  ${PROJECT_NAME}
)

include(GoogleTest)
gtest_discover_tests(${testname})

# Add symbolic link from tests/testdata in source directory to build directory
ADD_CUSTOM_TARGET(mpi_tests_symlink ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${data_src_dir}"
    "${CMAKE_CURRENT_BINARY_DIR}/${data_dirname}"
    )

# TODO: how to make ctest use mpirun?
