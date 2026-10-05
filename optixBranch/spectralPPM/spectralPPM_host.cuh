#pragma once

#include "spectralPPM_Utils.cuh"
#include "pbrt_spectral_utils.cuh"

void launch_spectral_PPM (
    OptixEngineState& engineState,
    CommonParams commonParams,
    uint32_t sampleCount
) {
    uint32_t maxPhotons = commonParams.w * commonParams.h * commonParams.max_depth;

    SpectralPhotonMap unsortedMap;
    void* unsortedMap_mem = allocateSpectralPhotonMap(unsortedMap, maxPhotons);

    SpectralPhotonMap sortedMap;
    void* sortedMap_mem = allocateSpectralPhotonMap(sortedMap, maxPhotons);
    
    SpectralParams specParams = {};

    cudaMalloc(&specParams.sRGBToSpectrumTable_Scale, sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Scale));
    cudaMemcpy(specParams.sRGBToSpectrumTable_Scale, PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Scale,
            sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Scale), cudaMemcpyHostToDevice);

    cudaMalloc(&specParams.sRGBToSpectrumTable_Data, sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Data));
    cudaMemcpy(specParams.sRGBToSpectrumTable_Data, PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Data,
            sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Data), cudaMemcpyHostToDevice);

    
    

    cudaFree(specParams.sRGBToSpectrumTable_Scale);
    cudaFree(specParams.sRGBToSpectrumTable_Data);

    cudaFree(unsortedMap_mem);
    cudaFree(sortedMap_mem);

}