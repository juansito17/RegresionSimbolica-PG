cmake_minimum_required(VERSION 3.10)
project(SymbolicRegressionGP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add source files
set(SOURCES
    src/main.cpp
    src/ExpressionTree.cpp
    src/Fitness.cpp
    src/GeneticOperators.cpp
    src/AdvancedFeatures.cpp
    src/GeneticAlgorithm.cpp
)

# Include directory
include_directories(src)

# Create executable
add_executable(SymbolicRegressionGP ${SOURCES})

# Link libraries
target_link_libraries(SymbolicRegressionGP
    PUBLIC # OpenMP necesita ser PUBLIC si afecta cabeceras o flags
        OpenMP::OpenMP_CXX
    PRIVATE # La librería matemática 'm' usualmente solo es necesaria internamente
        m
)

# Optional: Add optimization flags for release builds
# target_compile_options(SymbolicRegressionGP PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG>)
# target_compile_options(SymbolicRegressionGP PRIVATE $<$<CONFIG:Debug>:-g>)