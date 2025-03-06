cmake_minimum_required(VERSION 3.18)

# Get MPI
find_package(MPI REQUIRED)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

# Set variables
set(staticlib ${PROJECT_NAME})
set(sharedlib ${PROJECT_NAME}-shared)
set(binary ${PROJECT_NAME}-bin)

# List all sources used by the libs
set(libsources 
    core.cpp
    field.cpp
    io.cpp
    lib.cpp
    parallel.cpp
    utilities.cpp
    "${common_dir}/pngwriter.c"
    )

# Build a static library, a shared library and an executable
# Don't use an object library so only shared library uses -fPIC
add_executable(${binary})
add_library(${staticlib} STATIC ${libsources})
add_library(${sharedlib} SHARED ${libsources})

# The executable only has main as a source, it links with the static library
target_sources(
    ${binary}
    PRIVATE
    main.cpp
    )

# Add some include directories with the SYSTEM flag.
# This means our own compile options don't trigger warnings/errors
# from those header files
target_include_directories(${staticlib}
    SYSTEM
    PRIVATE
    ${dependencies}
    )

target_include_directories(${sharedlib}
    SYSTEM
    PRIVATE
    ${dependencies}
    )


# Our own header files
target_include_directories(${staticlib}
    PRIVATE
    ${common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(${sharedlib}
    PRIVATE
    ${common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

# Same for the executable
target_include_directories(${binary}
    SYSTEM
    PRIVATE
    ${dependencies}
    )

target_include_directories(${binary}
    PRIVATE
    ${common_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

# The static and shared libraries link with MPI
target_link_libraries(${staticlib}
    PRIVATE
    MPI::MPI_CXX
    )

target_link_libraries(${sharedlib}
    PRIVATE
    MPI::MPI_CXX
    )

# The executable links with the static library
target_link_libraries(${binary}
    PUBLIC
    ${staticlib}
    MPI::MPI_CXX
    )

install(
    TARGETS ${staticlib} ${sharedlib} ${binary}
    CONFIGURATIONS Release ReleaseWithDebInfo
    ARCHIVE DESTINATION ${install_dir}/lib
    LIBRARY DESTINATION ${install_dir}/lib
    RUNTIME DESTINATION ${install_dir}/bin
    PUBLIC_HEADER DESTINATION ${install_dir}/include
    )

# Create a symlink from testdata directory to the install directory
install(DIRECTORY "${CMAKE_SOURCE_DIR}/tests/testdata" DESTINATION ${install_dir}/bin)
