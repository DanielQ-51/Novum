#include "rng.cuh"
#include <stdio.h>

// Entirely exclude global memory setup if we are purely stateless
#if RNG_MODE != 3

static curandDirectionVectors32_t* g_sobolVectors = nullptr;
static unsigned int* g_scrambleConstants = nullptr;

// 3. THE KERNEL
__global__ void initRNG_Kernel(RNGState* states, int width, int height, 
#if RNG_MODE == 2
    curandDirectionVectors32_t* directionVectors,
    unsigned int* scrambleConstants
#else
    unsigned long seed
#endif
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

#if RNG_MODE == 2 
    unsigned int stride = 50;
    unsigned int offset = idx * stride; 
    
    curand_init(directionVectors[0], scrambleConstants[0], offset, &states[idx]);
#else 
    curand_init(seed, idx, 0, &states[idx]);
#endif
}
#endif // RNG_MODE != 3

// 4. THE MANAGER IMPLEMENTATION
namespace RNGManager {

    void cleanup() {
        #if RNG_MODE == 2
        if (g_sobolVectors) {
            cudaFree(g_sobolVectors);
            cudaFree(g_scrambleConstants);
            g_sobolVectors = nullptr;
        }
        #endif
    }

    void launchInitRNG(RNGState* d_rngStates, int width, int height, unsigned long seed) {
        #if RNG_MODE == 3
        // Stateless PCG generates its state on the fly in kernel registers.
        // There is no VRAM buffer to initialize, so we do nothing.
        return; 
        #else
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        #if RNG_MODE == 2
            // Lazy initialization of the vectors
            if (g_sobolVectors == nullptr) {
                curandDirectionVectors32_t* hostVecs;
                curandGetDirectionVectors32(&hostVecs, CURAND_DIRECTION_VECTORS_32_JOEKUO6);
                cudaMalloc(&g_sobolVectors, sizeof(curandDirectionVectors32_t) * 20000);
                cudaMemcpy(g_sobolVectors, hostVecs, sizeof(curandDirectionVectors32_t) * 20000, cudaMemcpyHostToDevice);

                unsigned int* hostScramble;
                curandGetScrambleConstants32(&hostScramble);
                cudaMalloc(&g_scrambleConstants, sizeof(unsigned int) * 20000);
                cudaMemcpy(g_scrambleConstants, hostScramble, sizeof(unsigned int) * 20000, cudaMemcpyHostToDevice);
            }
            initRNG_Kernel<<<grid, block>>>(d_rngStates, width, height, g_sobolVectors, g_scrambleConstants);
        #else
            initRNG_Kernel<<<grid, block>>>(d_rngStates, width, height, seed);
        #endif

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("RNG Init Error: %s\n", cudaGetErrorString(err));
            #if RNG_MODE == 2
            g_scrambleConstants = nullptr;
            #endif
        }
        #endif // RNG_MODE == 3
    }
}