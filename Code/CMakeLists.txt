# Usar una versión más moderna de CMake para un mejor soporte de CUDA
cmake_minimum_required(VERSION 3.18)

# Habilitar los lenguajes CXX (C++) y CUDA.
project(SymbolicRegressionGP LANGUAGES CXX CUDA)

# Establecer el estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Encontrar OpenMP para C++ y CUDA explícitamente
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Suppress warning about FindCUDA being deprecated (CMP0146)
cmake_policy(SET CMP0146 OLD)
find_package(CUDA REQUIRED) # Fuerza a CMake a definir las variables de CUDA

# Use Static Runtime to match static CUDA runtime and avoid LNK4098
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# AÑADIDO: Incluir directorios de CUDA explícitamente (método de compatibilidad)
# La variable ${CUDA_LIBRARIES} no añade los directorios de inclusión, así que lo hacemos manualmente.
include_directories(${CUDA_INCLUDE_DIRS})

# Usar el comando estándar add_executable. CMake sabe cómo manejar los archivos .cu
# porque CUDA fue habilitado como lenguaje en project().
add_executable(SymbolicRegressionGP
    src/main.cpp
    src/ExpressionTree.cpp
    src/Fitness.cpp
    src/GeneticOperators.cpp
    src/AdvancedFeatures.cpp
    src/GeneticAlgorithm.cpp
    src/FitnessGPU.cu
)

# Especificar los directorios de inclusión para nuestro ejecutable.
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

# Definir macros de preprocesador
target_compile_definitions(SymbolicRegressionGP PUBLIC "USE_GPU_ACCELERATION_DEFINED_BY_CMAKE")

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
