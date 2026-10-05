#include "optixStructs.cuh"
#include "integratorUtilities.cuh"
#include "spectralPPM_kernels.cuh"

__global__ void computeHashes(
    SpectralPhotonMap photons,
    int photonCount,
    uint32_t* d_hash_keys,
    uint32_t* d_indices,
    float3 sceneMin,
    float mergeRadius,
    int hashTableSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= photonCount) return;

    float3 p = getPos_cs(photons, i);

    d_hash_keys[i] = ComputeGridHash(p, sceneMin, mergeRadius, hashTableSize);
    d_indices[i] = i;
}

__global__ void reorderPhotons(
    SpectralPhotonMap photons,
    SpectralPhotonMap photons_sorted,
    int photonCount,
    uint32_t* d_indices_out
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= photonCount) return;

    // packed fields are copied as-is, no unpacking
    copyPhoton(photons, getSortedIndex_cs(d_indices_out, i), photons_sorted, i);
}

__global__ void buildTable(
    uint32_t* d_hashes_sorted,
    uint32_t* d_cell_start,
    uint32_t* d_cell_end,
    int numPhotons,
    int hashTableSize
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPhotons) return;

    uint32_t hash = d_hashes_sorted[i];

    if (hash >= hashTableSize) {
        printf("Error: Thread %d found invalid hash %u (Limit: %d)\n", i, hash, hashTableSize);
        return;
    }

    if (i == 0 || d_hashes_sorted[i - 1] != hash) {
        d_cell_start[hash] = i;
    }

    if (i == numPhotons - 1 || d_hashes_sorted[i + 1] != hash) {
        d_cell_end[hash] = i + 1;
    }
}

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
)
{
    int blockSize = 256;
    int numBlocks = (photonCount + blockSize - 1) / blockSize;

    computeHashes<<<numBlocks, blockSize, 0, stream>>>(
        photons,
        photonCount,
        d_hash_keys_in,
        d_indices_in,
        sceneMin,
        mergeRadius,
        hashTableSize
    );

    //checkCudaErrors("compute hashes");

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_hash_keys_in, d_hash_keys_out, d_indices_in, d_indices_out, photonCount,
        0, sizeof(uint32_t) * 8, stream);

    //checkCudaErrors("radix sort");

    reorderPhotons<<<numBlocks, blockSize, 0, stream>>>(
        photons,
        photons_sorted,
        photonCount,
        d_indices_out
    );

    //checkCudaErrors("reorder photons");

    cudaMemsetAsync(d_cell_start, 0xFF, hashTableSize * sizeof(uint32_t), stream);
    cudaMemsetAsync(d_cell_end,   0xFF, hashTableSize * sizeof(uint32_t), stream);

    buildTable<<<numBlocks, blockSize, 0, stream>>>(
        d_hash_keys_out,
        d_cell_start,
        d_cell_end,
        photonCount,
        hashTableSize
    );

    //checkCudaErrors("build table");
}
