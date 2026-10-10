#pragma once

#include "spectralPPM_Utils.cuh"
#include "pbrt_spectral_utils.cuh"
#include "spectralPPM_kernels.cuh"
#include <cub/cub.cuh>

// TEMPORARY debug: 1 = skip the eye pass and overwrite accum with this iteration's photon map (see paintPhotons).
#ifndef SPECTRAL_PPM_DISPLAY_PHOTONS
#define SPECTRAL_PPM_DISPLAY_PHOTONS 0
#endif

void launch_spectral_PPM (
    OptixEngineState& engineState,
    CommonParams commonParams,
    uint32_t sampleCount,
    float sceneRadius,
    float3 sceneCenter,
    const RenderConfig& config
) {
    CUstream stream;
    cudaStreamCreate(&stream);

    CUdeviceptr d_params;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(PipelineParams));

    uint32_t maxPhotons = commonParams.w * commonParams.h * commonParams.max_depth / 3.0f;

    SpectralPhotonMap unsortedMap;
    void* unsortedMap_mem = allocateSpectralPhotonMap(unsortedMap, maxPhotons);

    SpectralPhotonMap sortedMap;
    void* sortedMap_mem = allocateSpectralPhotonMap(sortedMap, maxPhotons);
    
    SpectralParams specParams = {};

    specParams.sceneCenter = sceneCenter;
    specParams.sceneRadius = sceneRadius;

    cudaMalloc(&specParams.sRGBToSpectrumTable_Scale, sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Scale));
    cudaMemcpy(specParams.sRGBToSpectrumTable_Scale, PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Scale,
            sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Scale), cudaMemcpyHostToDevice);

    cudaMalloc(&specParams.sRGBToSpectrumTable_Data, sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Data));
    cudaMemcpy(specParams.sRGBToSpectrumTable_Data, PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Data,
            sizeof(PBRT_RGB2SPEC_TABLE::sRGBToSpectrumTable_Data), cudaMemcpyHostToDevice);

    specParams.maxPhotons = maxPhotons;
    specParams.photons = unsortedMap;
    specParams.photonsSorted = sortedMap;

    uint32_t* __restrict__ d_hash_keys_in;
    uint32_t* __restrict__ d_hash_keys_out;
    uint32_t* __restrict__ d_indices_in;
    uint32_t* __restrict__ d_indices_out;

    cudaMalloc(&d_hash_keys_in, maxPhotons * sizeof(uint32_t));
    cudaMalloc(&d_hash_keys_out, maxPhotons * sizeof(uint32_t));
    cudaMalloc(&d_indices_in, maxPhotons * sizeof(uint32_t));
    cudaMalloc(&d_indices_out, maxPhotons * sizeof(uint32_t));

    specParams.hashTableSize = GetNextPrime(maxPhotons * 2);

    cudaMalloc(&specParams.cellStart, specParams.hashTableSize * sizeof(uint32_t));
    cudaMalloc(&specParams.cellEnd, specParams.hashTableSize * sizeof(uint32_t));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_hash_keys_in, d_hash_keys_out, d_indices_in, d_indices_out, maxPhotons);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaMalloc(&specParams.photonCounter, sizeof(uint32_t));

    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, commonParams.w * commonParams.h * sizeof(float4));

    auto lastSaveTime = std::chrono::steady_clock::now();
    int saveIntervalSamples = 2000;
    Image image = Image(commonParams.w, commonParams.h);
    // Post-processing (exposure/tonemap/gamma) now happens on the GPU in
    // cleanFormatAndPostProcessImage, so saveImageBMP() must not re-apply it.
    image.postProcess = false;
    std::vector<float4> h_finalOutput(commonParams.w * commonParams.h);

    PipelineParams allParams = {};
    allParams.common = commonParams;
    allParams.spectralPPM = specParams;

    double sumY = 0.0, sumD65Y = 0.0;
    for (int i = 0; i < pbrtSpectralUtils::nCIESamples; ++i) {        // lambda = 360 + i nm
        // D65 is stored as (lambda, value) pairs every 5 nm starting at 300 nm, so lerp between pairs
        float p = (360.0f + i - 300.0f) / 5.0f;
        int j = (int)p < 105 ? (int)p : 105;
        float t = p - j;
        float d65 = (1.0f - t) * pbrtSpectralUtils::CIE_Illum_D6500[2 * j + 1] 
                            + t  * pbrtSpectralUtils::CIE_Illum_D6500[2 * j + 3];

        sumY    += pbrtSpectralUtils::CIE_Y[i];
        sumD65Y += d65 * pbrtSpectralUtils::CIE_Y[i];
    }
    float d65Norm = (float)(sumY / sumD65Y); 

    std::cout << "Begin Render with Spectral PPM" << std::endl;

    auto renderStartTime = std::chrono::steady_clock::now();

    float VCMMergeConstant = config.vcmMergeConst;
    float VCMInitialMergeRadiusMultiplier = config.vcmInitialMergeRadiusMultiplier;

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    uint32_t w = commonParams.w;
    uint32_t h = commonParams.h;

    for (int currSample = 0; currSample < sampleCount; currSample++)
    {
        allParams.spectralPPM.mergeRadius = calculateMergeRadius(sceneRadius * VCMInitialMergeRadiusMultiplier, VCMMergeConstant, currSample);
        allParams.spectralPPM.invMergeNorm = 1.0f / (allParams.spectralPPM.mergeRadius * allParams.spectralPPM.mergeRadius * (w * h * h_PI));
        allParams.common.frame_index = currSample;
        //printf("Current merge radius: %f\n", allParams.spectralPPM.mergeRadius);
        double u = 0.5 + currSample * 0.6180339887498949;   // golden-ratio sequence
        u -= floor(u);
        SampledWavelengths sampledWavelengths = sampleWavelengthsVisible(u);
        for (int i = 0; i < N; ++i)
            allParams.spectralPPM.wl.lambda[i] = sampledWavelengths.lambda[i];

        for (int i = 0; i < N; ++i) {
            float p = (allParams.spectralPPM.wl.lambda[i] - 300.0f) / 5.0f;
            int j = (int)p < 105 ? (int)p : 105;
            float t = p - j;
            allParams.spectralPPM.d65[i] = d65Norm * ((1.0f - t) * pbrtSpectralUtils::CIE_Illum_D6500[2 * j + 1]
                                                    +         t  * pbrtSpectralUtils::CIE_Illum_D6500[2 * j + 3]);
        }

        // Film weights: R = sum_i L.v[i] * wR[i]. Each wavelength's share of the pixel is
        // cmf(lambda_i) / (N * pdf_i * CIE_Y_integral), so a flat spectrum of 1 gives Y = 1,
        // then XYZ -> linear sRGB (D65) is folded in.
        const float xyzToSRGB[3][3] = {
            { 3.2404542f, -1.5371385f, -0.4985314f },
            {-0.9692660f,  1.8760108f,  0.0415560f },
            { 0.0556434f, -0.2040259f,  1.0572252f } };

        for (int i = 0; i < N; ++i) {
            float p = sampledWavelengths.lambda[i] - 360.0f; // CIE tables are 1 nm from 360
            int j = (int)p < pbrtSpectralUtils::nCIESamples - 2 ? (int)p : pbrtSpectralUtils::nCIESamples - 2;
            float t = p - j;
            float xbar = (1.0f - t) * pbrtSpectralUtils::CIE_X[j] + t * pbrtSpectralUtils::CIE_X[j + 1];
            float ybar = (1.0f - t) * pbrtSpectralUtils::CIE_Y[j] + t * pbrtSpectralUtils::CIE_Y[j + 1];
            float zbar = (1.0f - t) * pbrtSpectralUtils::CIE_Z[j] + t * pbrtSpectralUtils::CIE_Z[j + 1];

            float s = 1.0f / (N * sampledWavelengths.pdf[i] * pbrtSpectralUtils::CIE_Y_integral);
            float X = xbar * s, Y = ybar * s, Z = zbar * s;

            allParams.spectralPPM.wR[i] = xyzToSRGB[0][0] * X + xyzToSRGB[0][1] * Y + xyzToSRGB[0][2] * Z;
            allParams.spectralPPM.wG[i] = xyzToSRGB[1][0] * X + xyzToSRGB[1][1] * Y + xyzToSRGB[1][2] * Z;
            allParams.spectralPPM.wB[i] = xyzToSRGB[2][0] * X + xyzToSRGB[2][1] * Y + xyzToSRGB[2][2] * Z;
        }

        // hero-only paths: lambda_0 alone, i.e. the index-0 weights without the 1/N
        allParams.spectralPPM.heroW = make_float3(N * allParams.spectralPPM.wR[0],
                                                  N * allParams.spectralPPM.wG[0],
                                                  N * allParams.spectralPPM.wB[0]);

        cudaMemsetAsync(allParams.spectralPPM.photonCounter, 0, sizeof(uint32_t), stream);
        cudaMemcpyAsync(reinterpret_cast<void*>(d_params), &allParams, sizeof(PipelineParams),
                cudaMemcpyHostToDevice, stream);

        // trace light paths
        optixLaunch(engineState.pipeline, stream, d_params, sizeof(PipelineParams),
            &engineState.sbt_spectralPPMLight, w, h, 1);

        uint32_t photonCount;
        cudaMemcpyAsync(&photonCount, allParams.spectralPPM.photonCounter, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        photonCount = photonCount < allParams.spectralPPM.maxPhotons ? photonCount : allParams.spectralPPM.maxPhotons;
        if (photonCount > 0) {
            buildHashGrid(
                allParams.spectralPPM.photons, 
                allParams.spectralPPM.photonsSorted,
                photonCount,
                d_hash_keys_in,
                d_hash_keys_out,
                d_indices_in,
                d_indices_out,
                d_temp_storage,
                temp_storage_bytes,
                allParams.spectralPPM.cellStart,
                allParams.spectralPPM.cellEnd,
                allParams.spectralPPM.mergeRadius, 
                allParams.spectralPPM.hashTableSize, 
                stream
            );
        }
        

#if SPECTRAL_PPM_DISPLAY_PHOTONS
        cudaMemsetAsync(commonParams.accum_buffer, 0, (size_t)w * h * sizeof(float4), stream);
        if (photonCount > 0) {
            const int paintStride = 1;
            int numPainted = (photonCount + paintStride - 1) / paintStride;
            paintPhotons<<<(numPainted + 255) / 256, 256, 0, stream>>>(
                allParams.spectralPPM.photonsSorted, photonCount, paintStride, commonParams.accum_buffer, w, h, commonParams.camera);
        }
#else
        optixLaunch(engineState.pipeline, stream, d_params, sizeof(PipelineParams),
            &engineState.sbt_spectralPPMEye, w, h, 1);
#endif

        if (DO_PROGRESSIVERENDER)
            cudaDeviceSynchronize();

        if (currSample % saveIntervalSamples == 0 && DO_PROGRESSIVERENDER)
        {
            dim3 blockSize(16, 16);
            dim3 gridSize((w + 15) / 16, (h + 15) / 16);
            cleanFormatAndPostProcessImage<<<gridSize, blockSize>>>(
                commonParams.accum_buffer, nullptr, d_finalOutput, w, h, SPECTRAL_PPM_DISPLAY_PHOTONS ? 0 : currSample, config.exposure, image.use_fitted_aces,
                true // ignore negatives (proper output of spectral to rgb conversion)
            );

            cudaMemcpy(h_finalOutput.data(), d_finalOutput, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            #pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                int x = i % w;
                int y = i / w;
                image.setColor(x, y, h_finalOutput[i]);
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);

            std::string filename2 = ASSET_PATH("renders/spectral_ppm/render.bmp");
            image.saveImageBMP(filename2);

            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);

            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, sampleCount, avgTimeMs);
            fflush(stdout);
        }
    }

    printf("\n");
    cudaDeviceSynchronize();

    cudaFree(d_finalOutput);

    cudaFree(d_temp_storage);

    cudaFree(specParams.photonCounter);

    cudaFree(d_hash_keys_in);
    cudaFree(d_hash_keys_out);
    cudaFree(d_indices_in);
    cudaFree(d_indices_out);

    cudaFree(specParams.cellStart);
    cudaFree(specParams.cellEnd);

    cudaFree(specParams.sRGBToSpectrumTable_Scale);
    cudaFree(specParams.sRGBToSpectrumTable_Data);

    cudaFree(unsortedMap_mem);
    cudaFree(sortedMap_mem);

    cudaFree(reinterpret_cast<void*>(d_params));

    cudaStreamDestroy(stream);

}