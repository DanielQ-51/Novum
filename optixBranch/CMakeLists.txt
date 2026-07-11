cmake_minimum_required(VERSION 3.24)

project(OptixExperiment LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

set(OPTIX_INCLUDE_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include")

# ==============================================================================
# 1. COMPILE THE DEVICE CODE TO PTX
# ==============================================================================

# Treat the megakernel as an "OBJECT" library and tell CMake to output PTX
add_library(optix_ptx_renderer OBJECT renderer.cu)
set_property(TARGET optix_ptx_renderer PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(optix_ptx_renderer PRIVATE
    ../include
    ../thirdparty
    ${OPTIX_INCLUDE_DIR}
)

target_compile_options(optix_ptx_renderer PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    
)


# --- PTX MODULE 2: ReSTIR Enhanced Shaders ---
add_library(optix_ptx_restir OBJECT restirPTenhancedShaders.cu)
set_property(TARGET optix_ptx_restir PROPERTY CUDA_PTX_COMPILATION ON)

target_include_directories(optix_ptx_restir PRIVATE
    ../include
    ../thirdparty
    ${OPTIX_INCLUDE_DIR}
)
target_compile_options(optix_ptx_restir PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

# ==============================================================================
# 2. COMPILE THE HOST CODE (EXECUTABLE)
# ==============================================================================

add_executable(renderer_optix
    main.cu
    restirPTenhanced_kernels.cu
    ../src/tinyexr_impl.cpp 
    ../thirdparty/tinyexr/miniz.c
    ../src/imageUtil.cu
    ../src/rng.cu
)

# Reach UP one directory to link your shared headers to the host code
target_include_directories(renderer_optix PRIVATE 
    ../include
    ../thirdparty
    ../thirdparty/tinyexr
    ${OPTIX_INCLUDE_DIR}
)

# ==============================================================================
# 3. FIXING THE ASSET_PATH MACRO
# ==============================================================================

# Get the absolute path to the parent directory (the root of NovumRender)
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH PARENT_DIR)

# Inject ROOT_DIR so your C++ ASSET_PATH macro works exactly like it did before
target_compile_definitions(renderer_optix PRIVATE 
    ROOT_DIR=\"${PARENT_DIR}\"
    PTX_DIR=\"$<TARGET_FILE_DIR:renderer_optix>\"
)


# ==============================================================================
# 4. LINKING LIBRARIES AND COPYING THE PTX
# ==============================================================================

# Link CUDA, the CUDA driver, and Windows API (Required by OptiX)
target_link_libraries(renderer_optix PRIVATE 
    CUDA::cuda_driver
    CUDA::cudart
    advapi32
)

# CMake buries .ptx files deep in build folders. 
# This automatically copies the generated PTX file and places it right next 
# to renderer_optix.exe so read_file_to_string("renderer.ptx") finds it instantly!
add_custom_command(TARGET renderer_optix POST_BUILD
    # Copy renderer.ptx
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_OBJECTS:optix_ptx_renderer>
    $<TARGET_FILE_DIR:renderer_optix>/renderer.ptx

    # Copy restirPTenhancedShaders.ptx
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_OBJECTS:optix_ptx_restir>
    $<TARGET_FILE_DIR:renderer_optix>/restirPTenhancedShaders.ptx
)