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

#ifndef ROOT_DIR
#define ROOT_DIR "." // Fallback just in case the compiler flag fails
#endif

#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

