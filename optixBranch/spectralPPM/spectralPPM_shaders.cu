#include <optix.h>
#include <optix_device.h>
#include "settings.cuh"
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"
#include "helpers.cuh"

#include "spectralPPM_Utils.cuh"

extern "C" {
    __constant__ PipelineParams allParams;
}

extern "C" __global__ void __raygen__spectralPPM_traceLight() {

}

extern "C" __global__ void __raygen__spectralPPM_traceEye() {

}
