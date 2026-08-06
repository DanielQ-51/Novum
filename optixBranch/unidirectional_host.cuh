#pragma once
#include <optix.h>
#include <optix_stubs.h>
#include "optixStructs.cuh"
#include <cstdio>
#include <algorithm>

__host__ void launch_unidirectional(
    OptixEngineState engineState,
    CommonParams commonParams,
    uint32_t sampleCount
) {
    if (sampleCount == 0) return;

    const uint32_t w = commonParams.w;
    const uint32_t h = commonParams.h;

    PipelineParams allParams = {};
    allParams.common = commonParams;

    CUdeviceptr d_params;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(PipelineParams));

    CUstream stream;
    cudaStreamCreate(&stream);

    // Start from a clean accumulator so the buffer holds exactly the sum of the
    // samples we fire here, regardless of prior state.
    cudaMemsetAsync(commonParams.accum_buffer, 0, (size_t)w * h * sizeof(float4), stream);

    // Pinned ring of launch params. Each entry is a full PipelineParams that
    // differs from the others only in frame_index (the RNG seed). Firing a wave
    // of launches that each memcpy from a DISTINCT pinned slot lets the burst run
    // un-synchronised with no host-buffer aliasing hazard: within a wave the host
    // never rewrites a slot while its async copy may still be in flight. d_params
    // is reused across launches in the wave -- that is safe because same-stream
    // ordering serialises memcpy(i) -> launch(i) -> memcpy(i+1).
    const uint32_t RING = 64;
    PipelineParams* h_ring = nullptr;
    cudaMallocHost(&h_ring, RING * sizeof(PipelineParams));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n", freeB / (1024.0 * 1024), totalB / (1024.0 * 1024));

    cudaEventRecord(start, stream);

    for (uint32_t base = 0; base < sampleCount; base += RING) {
        uint32_t waveCount = std::min(RING, sampleCount - base);

        // Prefill this wave's slots (no launches in flight are reading these yet:
        // the previous wave was synced before we got here).
        for (uint32_t i = 0; i < waveCount; i++) {
            h_ring[i] = allParams;
            h_ring[i].common.frame_index = base + i; // unique RNG seed per sample
        }

        // Fire the wave as an un-synced burst.
        for (uint32_t i = 0; i < waveCount; i++) {
            cudaMemcpyAsync(
                reinterpret_cast<void*>(d_params),
                &h_ring[i],
                sizeof(PipelineParams),
                cudaMemcpyHostToDevice,
                stream
            );

            optixLaunch(
                engineState.pipeline,
                stream,
                d_params,
                sizeof(PipelineParams),
                &engineState.sbt_unidirectional,
                w,   // Launch X
                h,   // Launch Y
                1    // Launch Z
            );
        }

        // Wave barrier: everything above must finish reading h_ring before we
        // overwrite it for the next wave. Doubles as the progress throttle.
        cudaStreamSynchronize(stream);
        printf("\rUnidirectional: %u / %u spp", base + waveCount, sampleCount);
        fflush(stdout);
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("\nUnidirectional PT: %u spp in %.2f ms (%.4f ms/spp)\n",
           sampleCount, ms, ms / sampleCount);

    cudaFreeHost(h_ring);
    cudaFree(reinterpret_cast<void*>(d_params));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}
