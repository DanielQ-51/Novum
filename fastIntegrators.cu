#include "integratorUtilities.cuh"
#include "reflectors.cuh"
#include "deviceCode.cuh"
#include <chrono>
#include <iostream>
#include "imageUtil.cuh"
#include <cub/cub.cuh>

__device__ __constant__ bool SAMPLE_ENVIRONMENT = false;

__device__ __constant__ bool BDPT_LIGHTTRACE;
__device__ __constant__ bool BDPT_NEE;
__device__ __constant__ bool BDPT_NAIVE;
__device__ __constant__ bool BDPT_CONNECTION;

__device__ __constant__ bool VCM_DOMERGE;

__device__ __constant__ bool DO_SPPM;

__device__ __constant__ bool BDPT_DRAWPATH;
__device__ __constant__ bool BDPT_DOMIS;
__device__ __constant__ bool BDPT_PAINTWEIGHT;

__device__ __constant__ float eta_vcm;
__device__ __constant__ float sceneRadius;
__device__ __constant__ float4 sceneCenter;
__device__ __constant__ float4 sceneMin;

__device__ __constant__ int w;
__device__ __constant__ int h;

__host__ void updateConstants(RenderConfig& config)
{
    cudaMemcpyToSymbol(BDPT_LIGHTTRACE, &config.bdptLightTrace, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_NAIVE, &config.bdptNaive, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_NEE, &config.bdptNee, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_CONNECTION, &config.bdptConnection, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_DRAWPATH, &config.bdptDrawPath, sizeof(bool));
    cudaMemcpyToSymbol(VCM_DOMERGE, &config.vcmDoMerge, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_DOMIS, &config.bdptDoMis, sizeof(bool));
    cudaMemcpyToSymbol(BDPT_PAINTWEIGHT, &config.bdptPaintWeight, sizeof(bool));
    cudaMemcpyToSymbol(SAMPLE_ENVIRONMENT, &config.sampleEnvironment, sizeof(bool));
    cudaMemcpyToSymbol(DO_SPPM, &config.doSPPM, sizeof(bool));
    return;
}

__global__ void generateInitialRays(
    cudaRNGState* rngStates,
    Camera camera
) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int rayID = y*w + x;

    cudaRNGState localState = rngStates[rayID];
    Ray r = camera.generateCameraRay(localState, x, y);
}

__host__ void launch_wavefrontUnidirectional(
    Camera camera, 
    const Material* __restrict__ materials, 
    const float4* __restrict__ textures, 
    const BVHnode* __restrict__ BVH, 
    const int* __restrict__ BVHindices, 
    const Vertices* __restrict__ vertices, 
    int vertNum, 
    const Triangle* __restrict__ scene, 
    int triNum, 
    const Triangle* __restrict__ lights, 
    int lightNum, int numSample, 
    int h_w, int h_h, 
    float4 h_sceneCenter, float h_sceneRadius, float4 h_sceneMin, 
    float4* __restrict__ colors, 
    float4* __restrict__ overlay, 
    bool postProcess
)
{
    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);

    cudaMemcpyToSymbol(sceneCenter, &(h_sceneCenter), sizeof(float4));
    cudaMemcpyToSymbol(sceneMin, &(h_sceneMin), sizeof(float4));
    cudaMemcpyToSymbol(sceneRadius, &(h_sceneRadius), sizeof(float));
    cudaMemcpyToSymbol(w, &(h_w), sizeof(int));
    cudaMemcpyToSymbol(h, &(h_h), sizeof(int));

    cudaRNGState* d_rngStates;
    cudaMalloc(&d_rngStates, w * h * sizeof(cudaRNGState));
    RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);

    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));
    
    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));
    
    auto lastSaveTime = std::chrono::steady_clock::now();
    int saveIntervalSamples = 30;
    Image image = Image(w, h);
    image.postProcess = postProcess;
    std::vector<float4> h_finalOutput(w * h);

    std::cout << "Begin Render with SPPM" << std::endl;

    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        
        if (DO_PROGRESSIVERENDER)
            cudaDeviceSynchronize();

        if (currSample % saveIntervalSamples == 0 && DO_PROGRESSIVERENDER) 
        {
            cleanAndFormatImage<<<gridSize, blockSize>>>(
                colors, overlay, d_finalOutput, w, h, currSample
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
            

            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (currSample + 1);
            
            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", currSample + 1, numSample, avgTimeMs);
            fflush(stdout);

            cudaMemset(overlay, 0, w * h * sizeof(float4));
        }
    }
    
    printf("\n"); // Move to a new line when the render loop finishes completely

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "RENDER ERROR: CUDA Error code: " << static_cast<int>(err) << std::endl;
        // only call this if the code isn't catastrophic
        if (err != cudaErrorAssert && err != cudaErrorUnknown)
            std::cerr << cudaGetErrorString(err) << std::endl;
    }
    else
        std::cout << "Render executed with no CUDA error" << std::endl;
}