#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <exception>
#include <set>
#include <iomanip>
#include "imageUtil.cuh"
#include "sceneContexts.cuh"
#include "objects.cuh"
#include "util.cuh"
#include <fstream>
#include <cuda_fp16.h>
#include <string>
#include <iomanip>

struct PipelineParams {
    Camera camera;
    ShadeContext shadeContext; // multiple pointers
    OptixTraversableHandle bvh_handle; // long long

    float4* accum_buffer;
    
    unsigned int w;
    unsigned int h;
    unsigned int frame_index;
    unsigned int max_depth;
};