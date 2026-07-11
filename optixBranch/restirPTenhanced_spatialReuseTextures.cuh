#pragma once

#include "restirPTenhanced_kernels.cuh"

__host__ short2* allocateReuseTexture(unsigned int dimension, unsigned int iterations) {
    uint32_t* bufferA;
    uint32_t* bufferB;
    uint32_t* indexTableSlot0;
    uint32_t* indexTableSlot1;
    
    void* raw;
    cudaMalloc(&raw, dimension * dimension * sizeof(uint32_t) * 3);

    char* ptr = static_cast<char*>(raw);
    bufferA = reinterpret_cast<uint32_t*>(ptr); ptr += sizeof(uint32_t) * dimension * dimension;
    bufferB = reinterpret_cast<uint32_t*>(ptr); ptr += sizeof(uint32_t) * dimension * dimension;
    indexTableSlot0 = reinterpret_cast<uint32_t*>(ptr); ptr += sizeof(uint32_t) * dimension * dimension / 2;
    indexTableSlot1 = reinterpret_cast<uint32_t*>(ptr);

    short2* outTexture;
    cudaMalloc(&outTexture, dimension * dimension * sizeof(short2));

    int blockSize = 256;
    int numBlocks = (dimension * dimension + blockSize - 1) / blockSize;

    initLinks<<<numBlocks, blockSize>>>(bufferA, dimension);

    numBlocks /= 4;

    uint32_t* final_buffer = nullptr;
    
    for (int i = 0; i < iterations; i++) {
        uint32_t* in_buf = (i % 2 == 0) ? bufferA : bufferB;
        uint32_t* out_buf = (i % 2 == 0) ? bufferB : bufferA;
        shuffleLinks<<<numBlocks, blockSize>>>(in_buf, out_buf, dimension, i);

        final_buffer = out_buf;
    }

    numBlocks *= 4;

    cudaMemset(indexTableSlot0, 0xFF, sizeof(uint32_t) * dimension * dimension / 2);

    resolvePairsPassA<<<numBlocks, blockSize>>>(final_buffer, indexTableSlot0, dimension);
    resolvePairsPassB<<<numBlocks, blockSize>>>(final_buffer, indexTableSlot0, indexTableSlot1, dimension);

    extractDeltasKernel<<<numBlocks, blockSize>>>(final_buffer, indexTableSlot0, indexTableSlot1, outTexture, dimension);

    cudaFree(raw);

    return outTexture;
}