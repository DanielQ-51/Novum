#pragma once
#include "optixStructs.cuh"

__global__ void computeHashes(
    SpectralPhotonMap photons,
    int photonCount,
    uint32_t* d_hash_keys,
    uint32_t* d_indices,
    float3 sceneMin,
    float mergeRadius,
    int hashTableSize
);

__global__ void reorderPhotons(
    SpectralPhotonMap photons,
    SpectralPhotonMap photons_sorted,
    int photonCount,
    uint32_t* d_indices_out
);

__global__ void buildTable(
    uint32_t* d_hashes_sorted,
    uint32_t* d_cell_start,
    uint32_t* d_cell_end,
    int numPhotons,
    int hashTableSize
);

__host__ void buildHashGrid(
    SpectralPhotonMap photons,
    SpectralPhotonMap photons_sorted,
    int photonCount,
    uint32_t* d_hash_keys_in,
    uint32_t* d_hash_keys_out,
    uint32_t* d_indices_in,
    uint32_t* d_indices_out,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    uint32_t* d_cell_start,
    uint32_t* d_cell_end,
    float3 sceneMin,
    float mergeRadius,
    int hashTableSize,
    cudaStream_t stream
);
