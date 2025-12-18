cmake_minimum_required(VERSION 3.10)
project(NanoDB VERSION 1.0.0 LANGUAGES CXX)

# --- 1. Standard Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Generates compile_commands.json for clangd/VSCode

# --- 2. Performance Optimizations (SIMD & O3) ---
# Detect system architecture (optional advanced step), but for now we hardcode for performance.
# -O3: Maximum optimization
# -march=native: Optimizes for the machine you are building on (enables AVX2 if available)
# -mavx2 -mfma: Explicitly enable AVX2 and Fused Multiply-Add
if(MSVC)
    # Windows/Visual Studio specific flags
    add_compile_options(/O2 /arch:AVX2 /openmp)
else()
    # GCC/Clang (Linux/Mac/MinGW)
    add_compile_options(-O3 -march=native -mavx2 -mfma -Wall -Wextra -pthread)
endif()

# --- 3. Dependencies ---
# OpenMP for multi-threading HNSW graph construction
find_package(OpenMP REQUIRED)

# --- 4. Include Directories ---
include_directories(include)

# --- 5. Source Files ---
# We manually list sources to avoid Globbing issues, keeping the build deterministic.
set(SOURCES
    src/main.cpp
    src/core/distance.cpp
    src/storage/mmap_handler.cpp
)

# --- 6. Build Executable ---
add_executable(nano_db ${SOURCES})

# Link OpenMP to the executable
if(OpenMP_CXX_FOUND)
    target_link_libraries(nano_db PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- 7. Testing (Optional Future Step) ---
enable_testing()
add_subdirectory(tests) # We will configure the tests/CMakeLists.txt later