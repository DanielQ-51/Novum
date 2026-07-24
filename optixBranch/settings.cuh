#pragma once

// Master switch for device side debug instrumentation.
//
// At 0, every debug printf, its format string, the overlay drawLine calls and
// the per pixel test that guards them are removed by the preprocessor. This
// matters more than it looks: the format strings live in the constant bank and
// each printf lowers to a vprintf call plus an argument buffer built on the
// local stack, so leaving them in costs constant memory, instruction cache and
// registers even on pixels that never print. Relying on the optimizer to drop
// them is not enough, since Debug builds compile with -O0 -G.
//
// Host side prints (timings, memory usage) are deliberately not gated.
#ifndef DEBUG_MODE
#define DEBUG_MODE 1
#endif

#ifndef SAVE_SEQUENCE
#define SAVE_SEQUENCE 1
#endif

#ifndef ACCUMULATE_FRAMES
#define ACCUMULATE_FRAMES 0
#endif

#ifndef DEBUG_VISUALIZE_TYPE
#define DEBUG_VISUALIZE_TYPE 0
#endif

#ifndef TEMPORAL_SKIP_REVERSE_SHIFT
#define TEMPORAL_SKIP_REVERSE_SHIFT 0
#endif

#ifndef CAMERA_MOVES
#define CAMERA_MOVES 0
#endif

#ifndef LERP_MCAP
#define LERP_MCAP 20.0f
#endif

#ifndef RECON_FOOTPRINT_C_CONSTANT
#define RECON_FOOTPRINT_C_CONSTANT 0.02f
#endif

#ifndef DEBUG_TEST_PIXEL_X
#define DEBUG_TEST_PIXEL_X 365
#endif

#ifndef DEBUG_TEST_PIXEL_Y
#define DEBUG_TEST_PIXEL_Y (800 - 200)
#endif

#ifndef NUM_REUSE_TEXTURES
#define NUM_REUSE_TEXTURES 3
#endif

#ifndef DO_SPATIAL_SHIFT
#define DO_SPATIAL_SHIFT 1
#endif

#ifndef USE_ENV_MAP
#define USE_ENV_MAP 1
#endif

// Runs the reuse texture self tests once at startup and prints a report.
// Costs a few ms, only meant to be on while debugging the pairing. At 0 the
// validation kernels are not compiled at all.
#ifndef VALIDATE_REUSE_TEXTURES
#define VALIDATE_REUSE_TEXTURES 1
#endif

// ---------------------------------------------------------------------------
// Debug instrumentation macros. Gated on DEBUG_MODE above.
//
// Use IS_DEBUG_PIXEL instead of comparing against DEBUG_TEST_PIXEL_X/Y by hand,
// and DEBUG_PRINTF / DEBUG_DRAWLINE / DEBUG_PRINT_PIXEL instead of calling
// printf / drawLine / printPixelData directly from device code, so a
// DEBUG_MODE 0 build carries none of it.
// ---------------------------------------------------------------------------
#if DEBUG_MODE == 1
    #define IS_DEBUG_PIXEL(px, py)  ((px) == DEBUG_TEST_PIXEL_X && (py) == DEBUG_TEST_PIXEL_Y)
    #define DEBUG_PRINTF(...)       printf(__VA_ARGS__)
    #define DEBUG_DRAWLINE(...)     drawLine(__VA_ARGS__)
    #define DEBUG_PRINT_PIXEL(...)  printPixelData(__VA_ARGS__)
#else
    #define IS_DEBUG_PIXEL(px, py)  (false)
    #define DEBUG_PRINTF(...)       ((void)0)
    #define DEBUG_DRAWLINE(...)     ((void)0)
    #define DEBUG_PRINT_PIXEL(...)  ((void)0)
#endif