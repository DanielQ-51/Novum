#pragma once
#include <curand_kernel.h>

// 1. CONFIGURATION
#ifndef RNG_MODE
#define RNG_MODE 3 // 1 = Philox, 2 = Sobol, 3 = Stateless PCG
#endif

// 2. STATELESS PCG IMPLEMENTATION
#if RNG_MODE == 3
    __device__ inline unsigned int pcg_hash(unsigned int input) {
        unsigned int state = input * 747796405u + 2891336453u;
        unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    __device__ inline float uint_to_float(unsigned int x) {
        return __uint_as_float(0x3f800000 | (x >> 9)) - 1.0f;
    }

    struct StatelessRNG {
        unsigned int state;

        __device__ inline void init(unsigned int pixel_id, unsigned int frame, unsigned int depth) {
            unsigned int seed = pixel_id ^ (frame * 1973u) ^ (depth * 9277u);
            
            state = 0u;
            state = state * 747796405u + (seed | 1u);
            next_float();
        }

        __device__ inline void init(unsigned int seed) {
            state = seed;
        }

        /** Gets the current state to be stored in the ReSTIR reservoir.
         *  Must be called BEFORE any next_float() queries for this path!
         */
        __device__ inline unsigned int getSeed() {
            return state;
        }

        __device__ inline float next_float() {
            unsigned int old_state = state;
            
            state = old_state * 747796405u + 2891336453u; 
            
            unsigned int word = ((old_state >> ((old_state >> 28u) + 4u)) ^ old_state) * 277803737u;
            unsigned int result = (word >> 22u) ^ word;
            
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

__device__ inline void save_rng(int pixel_id, RNGState* local_state, RNGState* global_states) {
#if RNG_MODE != 3
    global_states[pixel_id] = *local_state;
#endif
}