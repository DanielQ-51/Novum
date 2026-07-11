#pragma once
#include <optix.h>
#include <optix_device.h>
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"

#include "restirPTenhancedShaders.cuh"
#include "restirPTobjects.cuh"
#include "hostSetup.cuh"
#include "animation.cuh"
#include "restirPTenhanced_kernels.cuh"
#include "restirPTenhanced_spatialReuseTextures.cuh"
#include "settings.cuh"

__host__ void launch_restir (
    OptixEngineState engineState,
    CommonParams commonParams,
    uint32_t frameCount
) {
    Reservoir reservoir1, reservoir2;
    GBuffer gbuffer1;
    GBuffer gbuffer2;

    void* r1Memory = allocateReservoir(reservoir1, commonParams.w * commonParams.h);
    void* r2Memory = allocateReservoir(reservoir2, commonParams.w * commonParams.h);
    void* gb1Memory = allocateGBuffer(gbuffer1, commonParams.w * commonParams.h);
    void* gb2Memory = allocateGBuffer(gbuffer2, commonParams.w * commonParams.h);

    short2* reuseTexture1 = allocateReuseTexture(254, 16);
    short2* reuseTexture2 = allocateReuseTexture(230, 16);
    short2* reuseTexture3 = allocateReuseTexture(210, 16);

    PipelineParams allParams = {};
    allParams.common = commonParams;

    RestirCommonParams restirParams = {};
    restirParams.reservoir = reservoir1;
    restirParams.lastFrameReservoir = reservoir2;
    restirParams.gbuffer = gbuffer1;
    restirParams.prevGbuffer = gbuffer2;
    restirParams.reuseTexture1 = reuseTexture1;
    restirParams.reuseTexture2 = reuseTexture2;
    restirParams.reuseTexture3 = reuseTexture3;

    allParams.restir = restirParams;

    CUdeviceptr d_params;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(PipelineParams));

    CUstream stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    TurntableCameraAnimation animation = TurntableCameraAnimation(f3(0.0f, 0.0f, -1.5f), 6.5f, -0.f, 90.0f, 0.0f);
    //TurntableCameraAnimation animation = TurntableCameraAnimation(f3(0.0f, 0.0f, -1.5f), 6.5f, -0.36f, 90.0f, 0.0f);
    //LinearCameraAnimation animation = LinearCameraAnimation(f3(commonParams.camera.cameraOrigin), f3(commonParams.camera.xRot, commonParams.camera.yRot, commonParams.camera.zRot), f3(0.005f, 0.0f, 0.0f) ,f3());
    animation.update(allParams.common.camera, 0);

    dim3 blockSize(32, 8);  
    dim3 gridSize((commonParams.w+31)/32, (commonParams.h+7)/8);
    
    Image image = Image(commonParams.w, commonParams.h);
    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, commonParams.w * commonParams.h * sizeof(float4));
    cudaMemset(d_finalOutput, 0, commonParams.w * commonParams.h * sizeof(float4));

    float4* d_overlay;
    cudaMalloc(&d_overlay, commonParams.w * commonParams.h * sizeof(float4));
    cudaMemset(d_overlay, 0, commonParams.w * commonParams.h * sizeof(float4));
    float4* host_colors = new float4[commonParams.w * commonParams.h];

    uint8_t* d_duplication_map;
    cudaMalloc(&d_duplication_map, commonParams.w * commonParams.h * sizeof(uint8_t));
    cudaMemset(d_duplication_map, 0, commonParams.w * commonParams.h * sizeof(uint8_t));

    allParams.restir.duplication_map = d_duplication_map;
    allParams.common.overlay_buffer = d_overlay;

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    cudaEventRecord(start, stream);

    for (uint32_t frame = 0; frame < frameCount; frame++) {
        allParams.common.frame_index = frame;
        cudaMemcpyAsync(
            reinterpret_cast<void*>(d_params), 
            &allParams, 
            sizeof(PipelineParams), 
            cudaMemcpyHostToDevice, 
            stream
        );

        // Generate candidates, fill allParams.restir.reservoir

        optixLaunch(
            engineState.pipeline,
            stream,
            d_params,
            sizeof(PipelineParams), 
            &engineState.sbt_restirCandidate,                  
            commonParams.w,                   // Launch X
            commonParams.h,                   // Launch Y
            1                       // Launch Z
        );

        computeDualMV<<<gridSize, blockSize, 0, stream>>>(
            allParams.restir.gbuffer, 
            commonParams.w, 
            commonParams.h
        );

        computeDuplicationMapKernel<<<gridSize, blockSize, 0, stream>>>(
            allParams.restir.lastFrameReservoir, 
            allParams.restir.duplication_map, 
            commonParams.w, 
            commonParams.h
        );
        
        if (frame > 0) {
            optixLaunch(
                engineState.pipeline,
                stream,
                d_params,
                sizeof(PipelineParams), 
                &engineState.sbt_restirTemporal,                  
                commonParams.w,                   // Launch X
                commonParams.h,                   // Launch Y
                1                       // Launch Z
            );
        }
        

        displayWinningReservoirs<<<gridSize, blockSize, 0, stream>>>(allParams);
        
#if SAVE_SEQUENCE == 1

#if ACCUMULATE_FRAMES == 1
        cleanAndFormatImage<<<gridSize, blockSize, 0, stream>>>(
            allParams.common.accum_buffer, allParams.common.overlay_buffer, d_finalOutput, commonParams.w, commonParams.h, frame
        );
#else
        cleanAndFormatImage<<<gridSize, blockSize, 0, stream>>>(
            allParams.common.accum_buffer, allParams.common.overlay_buffer, d_finalOutput, commonParams.w, commonParams.h, 1
        );
#endif

        cudaMemcpyAsync(host_colors, d_finalOutput, commonParams.w * commonParams.h * sizeof(float4), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        #pragma omp parallel for
        for (int i = 0; i < commonParams.w * commonParams.h; i++) {
            int x = i % commonParams.w;
            int y = i / commonParams.w;
            image.setColor(x, y, host_colors[i]);
        }
        std::stringstream ss;
        
        
#if DEBUG_VISUALIZE_TYPE == 1
        ss << "renders/restirDebug/render" << std::setfill('0') << std::setw(4) << frame << ".bmp";
#else if DEBUG_VISUALIZE_TYPE == 0
        ss << "renders/restir/render" << std::setfill('0') << std::setw(4) << frame << ".bmp";
#endif  

        std::string filename = ASSET_PATH(ss.str());
        
        std::string filename2 = ASSET_PATH("renders/restir/render.bmp");
        image.saveImageBMP(filename);
        image.saveImageBMP(filename2);
        cudaMemsetAsync(d_overlay, 0, commonParams.w * commonParams.h * sizeof(float4), stream);
#endif
        // Swap reservoirs (changes baked into vram at start of loop)
        Reservoir temp = allParams.restir.lastFrameReservoir;
        allParams.restir.lastFrameReservoir = allParams.restir.reservoir;
        allParams.restir.reservoir = temp;

        cudaMemsetAsync(allParams.restir.reservoir.F, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.W, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.initRandomSeed, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.pathFlags, 0, commonParams.w * commonParams.h * sizeof(float), stream);
        cudaMemsetAsync(allParams.restir.reservoir.rcVertexGeometry, 0, commonParams.w * commonParams.h * sizeof(float4), stream);
        cudaMemsetAsync(allParams.restir.reservoir.rcVertexCachedValues, 0, commonParams.w * commonParams.h * sizeof(float2), stream);

        GBuffer tempGB = allParams.restir.prevGbuffer;
        allParams.restir.prevGbuffer = allParams.restir.gbuffer;
        allParams.restir.gbuffer = tempGB;

        allParams.restir.lastFrameCamera = allParams.common.camera;
        animation.update(allParams.common.camera, frame + 1);
        cudaStreamSynchronize(stream);
    }



    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "ReSTIR PT took: " << milliseconds/frameCount 
        << " ms per frame, or " << 1.0f / (milliseconds * 0.001f/frameCount) << " frames per second."<< std::endl;



    cudaFree(reinterpret_cast<void*>(d_params));
    cudaFree(r1Memory);
    cudaFree(r2Memory);
    cudaFree(gb1Memory);
    cudaFree(gb2Memory);
    cudaFree(d_finalOutput);
    cudaFree(d_overlay);
    cudaFree(d_duplication_map);
    cudaFree(reuseTexture1);
    cudaFree(reuseTexture2);
    cudaFree(reuseTexture3);
}



