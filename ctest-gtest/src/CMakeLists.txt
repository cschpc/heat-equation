cmake_minimum_required(VERSION 3.18)

find_package(PNG REQUIRED)
find_package(MPI REQUIRED)

# TODO:
# - Library and executable
add_executable(heat_tested
    core.cpp
    heat.cpp
    io.cpp
    main.cpp 
    parallel.cpp
    setup.cpp
    utilities.cpp
    "${common_dir}/pngwriter.c"
    )

target_compile_options(heat_tested
    PRIVATE
    ${MPI_C_COMPILE_OPTIONS}
    )

target_compile_definitions(heat_tested
    PRIVATE
    ${MPI_C_COMPILE_DEFINITIONS}
    ${PND_DEFINITIONS}
    )

target_link_libraries(heat_tested
    PRIVATE
    ${PNG_LIBRARIES}
    ${MPI_C_LIBRARIES}
    )

target_include_directories(heat_tested
    SYSTEM
    PRIVATE
    ${PNG_INCLUDE_DIRS}
    ${MPI_C_INCLUDE_DIRS}
    ${common_dir}
    )

install(TARGETS heat_tested DESTINATION ${install_dir})
install(FILES "${common_dir}/bottle.dat" DESTINATION ${install_dir})
