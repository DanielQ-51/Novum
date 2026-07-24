#pragma once

#include <cstdio>
#include <cmath>

#include "restirPTenhanced_kernels.cuh"
#include "settings.cuh"

#if VALIDATE_REUSE_TEXTURES == 1

// Runs countLinkIds/checkLinkCounts on the link buffer before deltas are
// extracted, so a broken generation stage is attributed to the shuffles rather
// than to the delta extraction. Called from allocateReuseTexture.
__host__ inline void validateLinkBuffer(const uint32_t* linkBuffer, unsigned int dimension) {
    const uint32_t numLinks = dimension * dimension / 2;

    uint32_t* d_counts = nullptr;
    ReuseTextureStats* d_stats = nullptr;
    cudaMalloc(&d_counts, numLinks * sizeof(uint32_t));
    cudaMalloc(&d_stats, sizeof(ReuseTextureStats));
    cudaMemset(d_counts, 0, numLinks * sizeof(uint32_t));
    cudaMemset(d_stats, 0, sizeof(ReuseTextureStats));

    dim3 block(32, 8);
    dim3 grid((dimension + block.x - 1) / block.x, (dimension + block.y - 1) / block.y);

    countLinkIds<<<grid, block>>>(linkBuffer, d_counts, dimension, d_stats);
    checkLinkCounts<<<(numLinks + 255) / 256, 256>>>(d_counts, numLinks, d_stats);

    cudaError_t err = cudaDeviceSynchronize();

    ReuseTextureStats s = {};
    cudaMemcpy(&s, d_stats, sizeof(ReuseTextureStats), cudaMemcpyDeviceToHost);

    printf("[reuseTex %ux%u] link id pass: %s  (%u ids not used exactly twice, %u out of range ids)%s\n",
           dimension, dimension,
           (err == cudaSuccess && s.brokenInvolution == 0 && s.outOfRange == 0) ? "PASS" : "FAIL",
           s.brokenInvolution, s.outOfRange,
           (err != cudaSuccess) ? cudaGetErrorString(err) : "");

    cudaFree(d_counts);
    cudaFree(d_stats);
}

#endif // VALIDATE_REUSE_TEXTURES == 1

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

    // Every one of these kernels indexes with blockIdx.y/threadIdx.y, so the launch
    // has to be 2D. A 1D launch pins y to 0 and only the first row gets written.
    const dim3 blockSize(32, 8);
    const dim3 gridFull((dimension + blockSize.x - 1) / blockSize.x,
                        (dimension + blockSize.y - 1) / blockSize.y);
    const dim3 gridHalf((dimension / 2 + blockSize.x - 1) / blockSize.x,
                        (dimension / 2 + blockSize.y - 1) / blockSize.y);

    initLinks<<<gridFull, blockSize>>>(bufferA, dimension);

    uint32_t* final_buffer = bufferA;

    for (int i = 0; i < iterations; i++) {
        uint32_t* in_buf = (i % 2 == 0) ? bufferA : bufferB;
        uint32_t* out_buf = (i % 2 == 0) ? bufferB : bufferA;
        shuffleLinks<<<gridHalf, blockSize>>>(in_buf, out_buf, dimension, i);

        final_buffer = out_buf;
    }

    cudaMemset(indexTableSlot0, 0xFF, sizeof(uint32_t) * dimension * dimension / 2);
    cudaMemset(indexTableSlot1, 0xFF, sizeof(uint32_t) * dimension * dimension / 2);

#if VALIDATE_REUSE_TEXTURES == 1
    validateLinkBuffer(final_buffer, dimension);
#endif

    resolvePairsPassA<<<gridFull, blockSize>>>(final_buffer, indexTableSlot0, dimension);
    resolvePairsPassB<<<gridFull, blockSize>>>(final_buffer, indexTableSlot0, indexTableSlot1, dimension);

    extractDeltasKernel<<<gridFull, blockSize>>>(final_buffer, indexTableSlot0, indexTableSlot1, outTexture, dimension);

    cudaFree(raw);

    return outTexture;
}

#if VALIDATE_REUSE_TEXTURES == 1
// ---------------------------------------------------------------------------
// Validation driver.
//
// Call once after the textures are built, e.g. from launch_restir:
//     validateReuseTextures(restirParams.reuseTextures,
//                           restirParams.reuseTextureSizes,
//                           NUM_REUSE_TEXTURES,
//                           commonParams.w, commonParams.h,
//                           8, 16);
//
// framesToTest sweeps frame_index so all four transpose/flip combinations of
// the per frame shuffle get exercised (a bug that only fires when both
// transpose and flip are on shows up in roughly 1 frame in 4).
//
// expectedIterations is the n_sigma passed to allocateReuseTexture, used only
// to print the sigma the paper's Eq. 3 predicts next to the measured one.
// ---------------------------------------------------------------------------
__host__ inline void validateReuseTextures(
    short2* const* textures,
    const uint32_t* sizes,
    uint32_t numTextures,
    uint32_t w,
    uint32_t h,
    uint32_t framesToTest,
    uint32_t expectedIterations
) {
    ReuseTextureStats* d_stats = nullptr;
    cudaMalloc(&d_stats, sizeof(ReuseTextureStats));

    ReuseTextureStats s = {};
    dim3 block(32, 8);

    printf("\n================ reuse texture validation ================\n");

    // ---- pass 1: the texture on its own ----------------------------------
    for (uint32_t t = 0; t < numTextures; ++t) {
        const uint32_t S = sizes[t];
        cudaMemset(d_stats, 0, sizeof(ReuseTextureStats));

        dim3 grid((S + block.x - 1) / block.x, (S + block.y - 1) / block.y);
        validateReuseTextureTexSpace<<<grid, block>>>(textures[t], S, t, d_stats);

        cudaError_t err = cudaDeviceSynchronize();
        cudaMemcpy(&s, d_stats, sizeof(ReuseTextureStats), cudaMemcpyDeviceToHost);

        const unsigned long long total = (unsigned long long)S * S;
        const bool pass = (err == cudaSuccess) && s.selfLinks == 0
                       && s.brokenInvolution == 0 && s.outOfRange == 0
                       && s.checked == total;

        printf("\n[tex %u  %ux%u] TEXTURE SPACE: %s\n", t, S, S, pass ? "PASS" : "FAIL");
        if (err != cudaSuccess) {
            printf("    cuda error: %s\n", cudaGetErrorString(err));
        }
        printf("    valid pairs      : %llu / %llu\n", s.checked, total);
        printf("    self links       : %u\n", s.selfLinks);
        printf("    broken inversions: %u\n", s.brokenInvolution);
        printf("    long links       : %u   (|component| > %u)\n", s.outOfRange, S / 2);

        if (s.checked > 0) {
            const double n     = (double)s.checked;
            const double meanX = (double)(long long)s.sumDx / n;
            const double meanY = (double)(long long)s.sumDy / n;
            const double sigma = sqrt((double)(long long)s.sumD2 / (2.0 * n));
            const double sigmaExpected = sqrt(2.0 * (double)expectedIterations - 1.0);

            printf("    mean delta       : (%.4f, %.4f)   (should be ~0)\n", meanX, meanY);
            printf("    sigma            : %.3f px   (Eq.3 predicts ~%.3f for n_sigma=%u)\n",
                   sigma, sigmaExpected, expectedIterations);
            printf("    max |component|  : %u\n", s.maxAbsDelta);
            printf("    |delta| histogram:\n");
            for (uint32_t b = 0; b < REUSE_VALIDATION_HIST_BINS; ++b) {
                if (s.hist[b] == 0) continue;
                printf("        %2u : %7u  (%5.2f%%)\n", b, s.hist[b], 100.0 * s.hist[b] / n);
            }
        }
    }

    // ---- pass 2: the screen wide test ------------------------------------
    dim3 screenGrid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    for (uint32_t t = 0; t < numTextures; ++t) {
        printf("\n[tex %u  %ux%u] SCREEN SPACE (%ux%u), frames 0..%u\n",
               t, sizes[t], sizes[t], w, h, framesToTest - 1);

        for (uint32_t frame = 0; frame < framesToTest; ++frame) {
            cudaMemset(d_stats, 0, sizeof(ReuseTextureStats));

            validateReuseTextureScreenSpace<<<screenGrid, block>>>(
                w, h, frame, t, sizes[t], textures[t], d_stats);

            cudaError_t err = cudaDeviceSynchronize();
            cudaMemcpy(&s, d_stats, sizeof(ReuseTextureStats), cudaMemcpyDeviceToHost);

            const unsigned long long total = (unsigned long long)w * h;
            const bool pass = (err == cudaSuccess) && s.selfLinks == 0
                           && s.brokenInvolution == 0 && s.outOfRange == 0
                           && (s.checked + s.offScreen == total);

            const double n     = s.checked > 0 ? (double)s.checked : 1.0;
            const double sigma = sqrt((double)(long long)s.sumD2 / (2.0 * n));

            printf("    frame %2u: %s  symmetric %llu (%.2f%%), off screen %u (%.2f%%), "
                   "asymmetric %u, self %u, oob %u, sigma %.3f\n",
                   frame, pass ? "PASS" : "FAIL",
                   s.checked, 100.0 * s.checked / (double)total,
                   s.offScreen, 100.0 * s.offScreen / (double)total,
                   s.brokenInvolution, s.selfLinks, s.outOfRange, sigma);

            if (err != cudaSuccess) {
                printf("        cuda error: %s\n", cudaGetErrorString(err));
            }
        }
    }

    printf("\n=========================================================\n\n");

    cudaFree(d_stats);
}

#endif // VALIDATE_REUSE_TEXTURES == 1