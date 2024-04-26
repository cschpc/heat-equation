cmake_minimum_required(VERSION 3.18)

find_package(MPI REQUIRED)

set(lib ${PROJECT_NAME})
set(objs objs)
set(binary ${PROJECT_NAME}-bin)

add_library(${lib} STATIC)
add_library(${objs} OBJECT "")
add_executable(${binary})

target_sources(
    ${objs}
    PRIVATE
    core.cpp
    field.cpp
    input.cpp
    io.cpp
    lib.cpp
    parallel.cpp
    setup.cpp
    utilities.cpp
    "${common_dir}/pngwriter.c"
    )

target_sources(
    ${binary}
    PRIVATE
    main.cpp
    )

target_compile_options(${objs}
    PRIVATE
    ${MPI_C_COMPILE_OPTIONS}
    )

target_compile_definitions(${objs}
    PRIVATE
    ${MPI_C_COMPILE_DEFINITIONS}
    )

target_include_directories(${objs}
    SYSTEM
    PRIVATE
    ${MPI_C_INCLUDE_DIRS}
    ${dependencies}
    )

target_include_directories(${objs}
    PRIVATE
    ${common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_options(${lib}
    PRIVATE
    ${MPI_C_LINK_FLAGS}
    )

target_link_libraries(${lib}
    PRIVATE
    ${objs}
    ${MPI_C_LIBRARIES}
    )

target_link_libraries(${binary} PUBLIC ${lib})

install(
    TARGETS ${lib} ${binary}
    CONFIGURATIONS Release ReleaseWithDebInfo
    ARCHIVE DESTINATION ${install_dir}/lib
    LIBRARY DESTINATION ${install_dir}/lib
    RUNTIME DESTINATION ${install_dir}/bin
    PUBLIC_HEADER DESTINATION ${install_dir}/include
    )
install(FILES "${common_dir}/bottle.dat" DESTINATION ${install_dir}/bin)
