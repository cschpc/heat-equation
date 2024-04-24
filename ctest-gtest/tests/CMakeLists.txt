cmake_minimum_required(VERSION 3.18)

set(testname ${PROJECT_NAME}_tests)

add_executable(
    ${testname}
    test.cpp
)

target_include_directories(${testname} PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    )

target_include_directories(${testname} SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/googletest-1.14.0/googletest/include"
    )

target_link_libraries(
  ${testname}
  GTest::gtest_main
  ${lib}
)

add_subdirectory(googletest-1.14.0)

include(GoogleTest)
gtest_discover_tests(${testname})
