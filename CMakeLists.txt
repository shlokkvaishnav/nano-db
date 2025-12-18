cmake_minimum_required(VERSION 3.10)
project(NanoDB VERSION 1.0.0 LANGUAGES CXX)

# Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags (AVX2)
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /openmp)
else()
    add_compile_options(-O3 -march=native -mavx2 -mfma -pthread)
endif()

find_package(OpenMP REQUIRED)

# Create the Core Library
# We split the logic into a library so both the Exe and Python can use it.
add_library(nano_core OBJECT
    src/core/distance.cpp
    src/storage/mmap_handler.cpp
    # Note: hnsw.hpp is header-only, so we don't list a .cpp here
)
target_include_directories(nano_core PUBLIC include)

# Build the C++ Executable (CLI)
add_executable(nano_db src/main.cpp)
target_link_libraries(nano_db PRIVATE nano_core OpenMP::OpenMP_CXX)

# Build the Python Module 
add_subdirectory(extern/pybind11) # Initialize pybind11

pybind11_add_module(nanodb_py src/python_bindings.cpp)
target_link_libraries(nanodb_py PRIVATE nano_core OpenMP::OpenMP_CXX)

# Rename output to just "nanodb" so we can "import nanodb"
set_target_properties(nanodb_py PROPERTIES OUTPUT_NAME "nanodb")