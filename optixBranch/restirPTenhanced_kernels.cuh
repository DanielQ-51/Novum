#pragma once
#include "optixStructs.cuh"

__global__ void computeDualMV(
    GBuffer gbuffer,
    uint32_t w,
    uint32_t h
);

__global__ void computeDuplicationMapKernel(
    Reservoir lastFrameReservoir,
    uint8_t* __restrict__ duplication_map,
    uint32_t w,
    uint32_t h
);

__global__ void displayWinningReservoirs(PipelineParams params);

__global__ void initLinks(uint32_t* buffer, uint32_t dimension);

__global__ void shuffleLinks(uint32_t* bufferA, uint32_t* bufferB, uint32_t dimension, uint32_t iteration);

__global__ void resolvePairsPassA(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t dimension);

__global__ void resolvePairsPassB(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, uint32_t dimension);

__global__ void extractDeltasKernel(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, short2* outputTex, uint32_t dimension);