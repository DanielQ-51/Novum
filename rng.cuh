#pragma once
#include <curand_kernel.h>

// 1. CONFIGURATION
// Ideally, define this in your build settings or a central constants file.
#ifndef RNG_MODE
#define RNG_MODE 1 // 1 = Philox, 2 = Sobol
#endif

// 2. TYPE DEFINITIONS
#if RNG_MODE == 1
    typedef curandStatePhilox4_32_10_t cudaRNGState;
#elif RNG_MODE == 2
    typedef curandStateScrambledSobol32_t cudaRNGState;
#else
    typedef curandState cudaRNGState;
#endif

// 3. FUNCTION DECLARATIONS
namespace RNGManager {
    // Initializes the RNG states on the device
    void launchInitRNG(cudaRNGState* d_rngStates, int width, int height, unsigned long seed);
    
    // Frees any internal resources (like Sobol vectors)
    void cleanup();
}