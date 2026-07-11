#pragma once
#include <optix.h>
#include <optix_device.h>
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"


extern "C" __global__ void __raygen__restirCandidateGeneration();
extern "C" __global__ void __raygen__restirSpatialReuse();
extern "C" __global__ void __raygen__restirTemporalReuse();