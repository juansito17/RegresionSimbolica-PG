cmake_minimum_required(VERSION 3.18)

# Initially only enable CXX. We will enable CUDA if found.
project(SymbolicRegressionGP LANGUAGES CXX)

# Fix for "PTX was compiled with an unsupported toolchain" on Colab T4
# This forces generation of SASS (binary) for the current GPU, avoiding JIT issues.
set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP (Required for CPU parallelism)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# --- CUDA Detection ---
include(CheckLanguage)
check_language(CUDA)

set(SOURCES
    src/main.cpp
    src/ExpressionTree.cpp
    src/Fitness.cpp
    src/GeneticOperators.cpp
    src/AdvancedFeatures.cpp
    src/GeneticAlgorithm.cpp
)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    enable_language(CUDA)
    
    # Suppress warning about FindCUDA being deprecated (CMP0146)
    cmake_policy(SET CMP0146 OLD)
    find_package(CUDA REQUIRED)

    # Add .cu file to sources only if CUDA is present
    list(APPEND SOURCES src/FitnessGPU.cu)

    # Create Executable
    add_executable(SymbolicRegressionGP ${SOURCES})

    # Define the macro to enable GPU code paths in C++
    target_compile_definitions(SymbolicRegressionGP PUBLIC "USE_GPU_ACCELERATION_DEFINED_BY_CMAKE")

    # Include CUDA dirs
    target_include_directories(SymbolicRegressionGP PUBLIC ${CUDA_INCLUDE_DIRS})

    # Link CUDA libraries
    # Try modern target first, fallback to variables
    if(TARGET CUDA::cudart)
        target_link_libraries(SymbolicRegressionGP PUBLIC CUDA::cudart)
    else()
        target_link_libraries(SymbolicRegressionGP PUBLIC ${CUDA_LIBRARIES})
    endif()

    set(CMAKE_CUDA_PROPAGATE_HOST_FLAGS ON)
    
    message(STATUS "Build type: GPU Accelerated")
else()
    message(STATUS "CUDA compiler NOT found. Building for CPU only.")
    
    # Create Executable (without .cu file)
    add_executable(SymbolicRegressionGP ${SOURCES})
    
    message(STATUS "Build type: CPU Only")
endif()

# Common settings
target_include_directories(SymbolicRegressionGP PUBLIC src)

# Enlazar las librerías necesarias.
# Se usa la variable legacy ${CUDA_LIBRARIES} como método de compatibilidad
# ya que el target moderno CUDA::cudart no se está encontrando en este sistema.
target_link_libraries(SymbolicRegressionGP PUBLIC
    # Enlazar con OpenMP para C++
    OpenMP::OpenMP_CXX

    # Enlazar con las librerías de CUDA (método de compatibilidad)
    ${CUDA_LIBRARIES}
)

# Asegura que los flags del compilador de C++ (como /std:c++17) se pasen a nvcc.
set(CMAKE_CUDA_PROPAGATE_HOST_FLAGS ON)


# === OPTIMIZACIÓN: Flags de compilación agresivos para GCC/Clang (Colab) ===
# NOTA: Para MSVC+CUDA, dejamos que CMake maneje la optimización automáticamente
#       ya que los flags manuales causan conflictos con nvcc
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(SymbolicRegressionGP PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>              # Máxima optimización
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>    # Optimizar para CPU actual
        $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>      # Matemáticas rápidas
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>   # Desenrollar loops
        $<$<COMPILE_LANGUAGE:CXX>:-ftree-vectorize> # Forzar vectorización
    )
    # Link-time optimization (LTO) para Release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_property(TARGET SymbolicRegressionGP PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()
