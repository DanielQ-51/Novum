#pragma once
#include <curand_kernel.h>

// 1. CONFIGURATION
#ifndef RNG_MODE
#define RNG_MODE 3 // 1 = Philox, 2 = Sobol, 3 = Stateless PCG
#endif

__device__ inline uint32_t hash_pixel(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__device__ inline uint32_t hash_depth(uint32_t depth) {
    depth ^= depth >> 16;
    depth *= 0x85ebca6b;
    depth ^= depth >> 13;
    depth *= 0xc2b2ae35;
    depth ^= depth >> 16;
    return depth;
}

__device__ __forceinline__ uint32_t hash_uint32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

// 2. STATELESS PCG IMPLEMENTATION
#if RNG_MODE == 3
    __device__ inline uint32_t pcg_hash(uint32_t input) {
        uint32_t state = input * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    __device__ inline float uint_to_float(uint32_t x) {
        return __uint_as_float(0x3f800000 | (x >> 9)) - 1.0f;
    }

    struct StatelessRNG {
        uint32_t state;

        __device__ inline void init(uint32_t pixel_id, uint32_t frame, uint32_t depth) {
            uint32_t scrambled_id = hash_pixel(pixel_id); 
            uint32_t hashed_depth = hash_depth(depth);

            uint32_t seed = scrambled_id ^ (frame * 1973u) ^ hashed_depth;
            
            state = 0u;
            state = state * 747796405u + (seed | 1u);
            next_float();
        }

        __device__ inline void init(uint32_t seed) {
            state = seed;
        }

        /** Gets the current state to be stored in the ReSTIR reservoir.
         *  Must be called BEFORE any next_float() queries for this path!
         */
        __device__ inline uint32_t getSeed() {
            return state;
        }

        __device__ inline float next_float() {
            uint32_t old_state = state;
            
            state = old_state * 747796405u + 2891336453u; 
            
            uint32_t word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
            uint32_t result = (word >> 22u) ^ word;
            
            return __uint_as_float(0x3f800000 | (result >> 9)) - 1.0f;
        }
    };
#endif

#if RNG_MODE == 1
    typedef curandStatePhilox4_32_10_t RNGState;
#elif RNG_MODE == 2
    typedef curandStateScrambledSobol32_t RNGState;
#elif RNG_MODE == 3
    typedef StatelessRNG RNGState;
#else
    typedef curandState RNGState;
#endif

__device__ inline float rand(RNGState* state) {
#if RNG_MODE == 3
    return state->next_float();
#else
    return curand_uniform(state);
#endif
}

__device__ inline float4 rand4(RNGState* state) {
#if RNG_MODE == 3
    return make_float4(state->next_float(), state->next_float(), state->next_float(), state->next_float());
#else
    return make_float4(curand_uniform(state), curand_uniform(state), curand_uniform(state), curand_uniform(state));
#endif
}

namespace RNGManager {
    void launchInitRNG(RNGState* d_rngStates, int width, int height, unsigned long seed);
    void cleanup();
}

__device__ inline RNGState load_rng(int pixel_id, int frame_num, int depth, RNGState* global_states) {
    RNGState local_state;
    
#if RNG_MODE == 3
    local_state.init(pixel_id, frame_num, depth);
#else
    local_state = global_states[pixel_id];
#endif

    return local_state;
}
#if RNG_MODE == 3
__device__ inline RNGState load_rng(uint32_t seed) {
    RNGState local_state;
    
    local_state.init(seed);

    return local_state;
}
#endif
__device__ inline void save_rng(int pixel_id, RNGState* local_state, RNGState* global_states) {
#if RNG_MODE != 3
    global_states[pixel_id] = *local_state;
#endif
}