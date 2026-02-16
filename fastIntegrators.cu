#include "integratorUtilities.cuh"
#include "reflectors.cuh"
#include "deviceCode.cuh"
#include <chrono>
#include <iostream>
#include "imageUtil.cuh"
#include <cub/cub.cuh>

__device__ __constant__ bool SAMPLE_ENVIRONMENT = false;

__host__ void launch_wavefrontUnidirectional ()
{

}