cmake_minimum_required(VERSION 3.24)

project(NovumRender LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

add_executable(NovumRender
    src/main.cu
    src/deviceCode.cu
    src/imageUtil.cu
    src/rng.cu
    src/fastIntegrators.cu
)

target_include_directories(NovumRender PRIVATE include)

target_include_directories(NovumRender PRIVATE third_party)

target_compile_definitions(NovumRender PRIVATE 
    ROOT_DIR=\"${CMAKE_SOURCE_DIR}\"
)
target_compile_options(NovumRender PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

target_link_libraries(NovumRender PRIVATE CUDA::curand)