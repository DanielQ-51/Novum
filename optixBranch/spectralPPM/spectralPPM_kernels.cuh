#pragma once
#include "optixStructs.cuh"

__global__ void computeHashes(
    SpectralPhotonMap photons,
    int photonCount,
    uint32_t* d_hash_keys,
    uint32_t* d_indices,
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
    float mergeRadius,
    int hashTableSize,
    cudaStream_t stream
);

// Debug: splat every stride-th photon into accum (overwrite mode, see SPECTRAL_PPM_DISPLAY_PHOTONS).
// Finite photons add 1 to each channel; a photon with non-finite radiance adds NaN, which the format kernel shows as magenta.
__global__ void paintPhotons(
    SpectralPhotonMap photons,
    int numPhotons,
    int stride,
    float4* __restrict__ accum,
    int w, int h,
    Camera camera
);
