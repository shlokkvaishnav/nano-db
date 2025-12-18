cmake_minimum_required(VERSION 3.10)
project(NanoDB VERSION 1.0.0 LANGUAGES CXX)

# Standard Configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # ensures your editor doesn't show red squigglies when you use AVX2 intrinsics or OpenMP directives.

# Performance Optimizations
if(MSVC)
    # Windows (MSVC): Optimize for speed (/O2), enable AVX2 SIMD, and OpenMP
    add_compile_options(/O2 /arch:AVX2 /openmp)
else()
    # Linux (GCC/Clang): Optimize for speed (-O3), native CPU architecture, AVX2, and threading
    add_compile_options(-O3 -march=native -mavx2 -mfma -Wall -Wextra -pthread)
endif()

# Dependencies
find_package(OpenMP REQUIRED) # Required for multi-threaded graph construction

# Include Directories
include_directories(include) # Expose the 'include' folder globally

#Source Files
set(SOURCES
    src/main.cpp
    src/core/distance.cpp
    src/storage/mmap_handler.cpp
)

# Build Executable
add_executable(nano_db ${SOURCES})

# Link OpenMP library to the target executable
if(OpenMP_CXX_FOUND)
    target_link_libraries(nano_db PRIVATE OpenMP::OpenMP_CXX)
endif()

# Testing
enable_testing()
add_subdirectory(tests) # Initialize test suite