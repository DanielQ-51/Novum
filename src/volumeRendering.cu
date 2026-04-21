#include "integratorUtilities.cuh"
#include "reflectors.cuh"
#include "volumeRendering.cuh"
#include <chrono>
#include <iostream>
#include "imageUtil.cuh"
#include "sceneContexts.cuh"
#include <cub/cub.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

__device__ __constant__ float sceneRadius;
__device__ __constant__ float4 sceneCenter;
__device__ __constant__ float4 sceneMin;

__device__ __constant__ int w;
__device__ __constant__ int h;

using leaf_t = nanovdb::LeafNode<float>;

__device__ void buildOrthonormalBasis(const nanovdb::Vec3f& n, nanovdb::Vec3f& b1, nanovdb::Vec3f& b2) {
    // A simple and robust way to generate two orthogonal vectors.
    // If n is pointing too close to the X-axis, use the Y-axis to cross, otherwise use X-axis.
    if (abs(n[0]) > 0.9f) {
        b1 = nanovdb::Vec3f(0.0f, 1.0f, 0.0f).cross(n);
    } else {
        b1 = nanovdb::Vec3f(1.0f, 0.0f, 0.0f).cross(n);
    }
    b1.normalize();
    b2 = n.cross(b1);
    b2.normalize();
}

__device__ nanovdb::Vec3f sample_HG(const nanovdb::Vec3f& incoming_dir, float g, float u1, float u2) {
    float cos_theta;
    
    if (abs(g) < 1e-3f) {
        // Isotropic edge case (g is effectively 0)
        cos_theta = 1.0f - 2.0f * u2;
    } else {
        // Anisotropic Henyey-Greenstein math
        float sqrTerm = (1.0f - g * g) / (1.0f - g + 2.0f * g * u2);
        cos_theta = (1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g);
    }

    float sin_theta = sqrt(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    
    // Azimuth angle (perfectly uniform around the ray)
    float phi = 2.0f * PI * u1;
    
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    // --- PHASE 2: Build the Local Vector ---
    // This vector assumes the incoming ray was pointing perfectly down the Z-axis (0, 0, 1)
    nanovdb::Vec3f local_dir(
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    );

    // --- PHASE 3: Rotate to World Space ---
    // Create an Orthonormal Basis (Tangent, Bi-tangent, Normal) around the incoming ray
    nanovdb::Vec3f tangent, bitangent;
    buildOrthonormalBasis(incoming_dir, tangent, bitangent);

    // Multiply the local vector by the basis to transform it into World Space
    nanovdb::Vec3f world_dir = 
        tangent * local_dir[0] + 
        bitangent * local_dir[1] + 
        incoming_dir * local_dir[2];

    world_dir.normalize(); // Ensure perfect unit length to prevent floating point drift
    
    return world_dir;
}

__global__ void render_volume(
    RNGState* rngStates,
    Camera camera,
    const SceneContext sceneContext,
    nanovdb::NanoGrid<float>* densityGrid,
    float4* __restrict__ colors,
    int maxDepth,
    int frameNum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);
    
    Ray r = camera.generateCameraRay(localState, x, y);
    
    nanovdb::Ray<float> worldRay(
        nanovdb::Vec3f(r.origin.x, r.origin.y, r.origin.z), 
        nanovdb::Vec3f(r.direction.x, r.direction.y, r.direction.z)
    );

    auto accessor = densityGrid->getAccessor();
    auto sampler = nanovdb::math::createSampler<1>(accessor);

    float4 throughput = f4(1.0f);
    float4 pixelColor = f4();
    float density_scale = 3.0f;

    // --- 1. BOUNCE LOOP ---
    for (int depth = 0; depth < maxDepth; depth++) {
        localState = load_rng(pixelIdx, frameNum, depth+1, rngStates);
        nanovdb::Ray<float> indexRay = worldRay.worldToIndexF(*densityGrid);
        nanovdb::math::TreeMarcher<leaf_t, nanovdb::Ray<float>, decltype(accessor)> marcher(accessor);

        if (!marcher.init(indexRay)) {
            nanovdb::Vec3f nv_dir = worldRay.dir();
            pixelColor += throughput * sampleSky(f4(nv_dir[0], nv_dir[1], nv_dir[2]));
            break;
        }

        const leaf_t* leaf = nullptr;
        float t0, t1;
        
        bool hit_particle = false;
        float t_hit = 0.0f;

        while (marcher.step(&leaf, t0, t1)) {
            float local_majorant = leaf->maximum() * density_scale;
            if (local_majorant <= 0.0f) continue;

            float t_current = t0;

            while (true) {
                float step = -log(rand(&localState)) / local_majorant;
                t_current += step;

                if (t_current >= t1) break; // Exited the node

                float true_density = sampler(indexRay(t_current)) * density_scale;

                if (rand(&localState) < (true_density / local_majorant)) {
                    hit_particle = true;
                    t_hit = t_current; // Save the exact hit distance
                    break;
                }
            }
            
            if (hit_particle) break; 
        }

        if (hit_particle) {
            throughput *= f4(0.9f, 0.9f, 0.9f);

            nanovdb::Vec3f index_hit_pos = indexRay(t_hit); // Use the saved t_hit
            nanovdb::Vec3f world_hit_pos = densityGrid->indexToWorldF(index_hit_pos);

            nanovdb::Vec3f new_world_dir = sample_HG(worldRay.dir(), 0.6f, rand(&localState), rand(&localState));

            worldRay = nanovdb::Ray<float>(world_hit_pos, new_world_dir);
            
        } else {
            nanovdb::Vec3f nv_dir = worldRay.dir();
            pixelColor += throughput * sampleSky(f4(nv_dir[0], nv_dir[1], nv_dir[2]));
            break; // Break the depth loop
        }
    }
    
    colors[pixelIdx] += pixelColor;
    save_rng(pixelIdx, &localState, rngStates);
}

/*
__global__ void render_volume_surface_integrated(
    RNGState* rngStates,
    Camera camera,
    const SceneContext sceneContext,
    const BVHContext bvhContext,
    float4* __restrict__ colors,
    int maxDepth,
    int frameNum
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;
    int pixelIdx = y*w + x;

    RNGState localState = load_rng(pixelIdx, frameNum, 0, rngStates);
    
    Ray r = camera.generateCameraRay(localState, x, y);

    float4 throughput = f4(1.0f);
    float4 pixelColor = f4();
    
    for (int depth = 0; depth < maxDepth; depth++) {
{
        localState = load_rng(pixelIdx, frameNum, depth+1, rngStates);

        float4 bary;
        float min_t_surface;
        float min_t_volume;
        int primID_surface;
        int primID_volume;
        BVHSceneIntersect_volume(r, bvhContext, bary, min_t_surface, min_t_volume, primID_surface, primID_volume);

        if (primID_surface == -1 && primID_volume == -1) { // skybox hit

        } else if (min_t_surface < min_t_volume) { // surface only hit
            
        } else { // volume first hit
            nanovdb::NanoGrid<float>* densityGrid = sceneContext.volumes[primID_volume].density_pointer;

            auto accessor = densityGrid->getAccessor();
            auto sampler = nanovdb::math::createSampler<1>(accessor);


        }
}

        auto accessor = densityGrid->getAccessor();
        auto sampler = nanovdb::math::createSampler<1>(accessor);

        nanovdb::Ray<float> indexRay = worldRay.worldToIndexF(*densityGrid);
        nanovdb::math::TreeMarcher<leaf_t, nanovdb::Ray<float>, decltype(accessor)> marcher(accessor);

        if (!marcher.init(indexRay)) {
            nanovdb::Vec3f nv_dir = worldRay.dir();
            pixelColor += throughput * sampleSky(f4(nv_dir[0], nv_dir[1], nv_dir[2]));
            break;
        }

        const leaf_t* leaf = nullptr;
        float t0, t1;
        
        bool hit_particle = false;
        float t_hit = 0.0f;

        while (marcher.step(&leaf, t0, t1)) {
            float local_majorant = leaf->maximum() * density_scale;
            if (local_majorant <= 0.0f) continue;

            float t_current = t0;

            while (true) {
                float step = -log(rand(&localState)) / local_majorant;
                t_current += step;

                if (t_current >= t1) break; // Exited the node

                float true_density = sampler(indexRay(t_current)) * density_scale;

                if (rand(&localState) < (true_density / local_majorant)) {
                    hit_particle = true;
                    t_hit = t_current; // Save the exact hit distance
                    break;
                }
            }
            
            if (hit_particle) break; 
        }

        if (hit_particle) {
            throughput *= f4(0.9f, 0.9f, 0.9f);

            nanovdb::Vec3f index_hit_pos = indexRay(t_hit); // Use the saved t_hit
            nanovdb::Vec3f world_hit_pos = densityGrid->indexToWorldF(index_hit_pos);

            nanovdb::Vec3f new_world_dir = sample_HG(worldRay.dir(), 0.6f, rand(&localState), rand(&localState));

            worldRay = nanovdb::Ray<float>(world_hit_pos, new_world_dir);
            
        } else {
            nanovdb::Vec3f nv_dir = worldRay.dir();
            pixelColor += throughput * sampleSky(f4(nv_dir[0], nv_dir[1], nv_dir[2]));
            break; // Break the depth loop
        }
    }
    
    colors[pixelIdx] += pixelColor;
    save_rng(pixelIdx, &localState, rngStates);
} */

__host__ void launch_simple_volume(
    Camera camera, 
    const SceneContext sceneContext,
    int numSample, int maxDepth,
    int h_w, int h_h, 
    float4 h_sceneCenter, float h_sceneRadius, float4 h_sceneMin, 
    float4* __restrict__ colors, 
    float4* __restrict__ overlay, 
    bool postProcess
)
{
    cudaMemcpyToSymbol(sceneCenter, &(h_sceneCenter), sizeof(float4));
    cudaMemcpyToSymbol(sceneMin, &(h_sceneMin), sizeof(float4));
    cudaMemcpyToSymbol(sceneRadius, &(h_sceneRadius), sizeof(float));
    cudaMemcpyToSymbol(w, &(h_w), sizeof(int));
    cudaMemcpyToSymbol(h, &(h_h), sizeof(int));

    dim3 blockSize(16, 16);  
    dim3 gridSize((h_w+15)/16, (h_h+15)/16);

    #if RNG_MODE == 3
        RNGState* d_rngStates = nullptr;
    #else
        RNGState* d_rngStates;
        cudaMalloc(&d_rngStates, w * h * sizeof(RNGState));
        RNGManager::launchInitRNG(d_rngStates, w, h, 5124123UL);
    #endif
    
    cudaDeviceSynchronize();

    auto handle = nanovdb::io::readGrid(ASSET_PATH("assets/vdb/nvdb/smoke2.nvdb"), "density");
    const nanovdb::NanoGrid<float>* hostGrid = handle.grid<float>();

    size_t gridSizeInBytes = handle.size();

    void* d_gridBuffer = nullptr;
    cudaMalloc(&d_gridBuffer, gridSizeInBytes);
    cudaMemcpy(d_gridBuffer, hostGrid, gridSizeInBytes, cudaMemcpyHostToDevice);

    float4* d_finalOutput;
    float4* d_overlay;
    cudaMalloc(&d_finalOutput, h_w * h_h * sizeof(float4));
    cudaMalloc(&d_overlay, h_w * h_h * sizeof(float4));
    cudaMemset(d_overlay, 0, h_w * h_h * sizeof(float4)); // Zero out the dummy overlay


    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("Free: %.2f MB of %.2f MB\n",
            freeB / (1024.0*1024),
            totalB / (1024.0*1024));

    // Image Object (CPU) & Saving logic from SPPM
    int saveIntervalSamples = 30; // Matches SPPM logic
    Image image = Image(h_w, h_h);
    image.postProcess = postProcess;
    std::vector<float4> h_finalOutput(h_w * h_h); 

    std::cout << "Running Kernels Unidirectional" << std::endl;
    
    // Start total timer
    auto renderStartTime = std::chrono::steady_clock::now();
    
    for (int currSample = 0; currSample < numSample; currSample++)
    {
        render_volume<<<gridSize, blockSize>>>(
            d_rngStates,
            camera,
            sceneContext,
            reinterpret_cast<nanovdb::NanoGrid<float>*>(d_gridBuffer),
            colors,
            maxDepth,
            currSample
        );
        cudaDeviceSynchronize();

        if ((currSample % saveIntervalSamples == 0 || currSample == numSample-1) && DO_PROGRESSIVERENDER) 
        {
            // Launch the formatting kernel to handle averaging, NaNs, and Infs on the GPU
            cleanAndFormatImage<<<gridSize, blockSize>>>(
                colors, d_overlay, d_finalOutput, h_w, h_h, currSample
            );

            // Copy the finalized buffer back to the host
            cudaMemcpy(h_finalOutput.data(), d_finalOutput, h_w * h_h * sizeof(float4), cudaMemcpyDeviceToHost);

            // Clean OpenMP loop simply maps the formatted colors to the image
            #pragma omp parallel for
            for (int i = 0; i < h_w * h_h; i++) 
            {
                int x = i % h_w;
                int y = i / h_w;
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

            // Reset the dummy overlay just like in the wavefront version
            cudaMemset(d_overlay, 0, h_w * h_h * sizeof(float4));
        }
    }
    
    printf("\n"); // Move to a new line when the render loop finishes completely
    cudaDeviceSynchronize();
    cudaFree(d_rngStates);
    cudaFree(d_gridBuffer);

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