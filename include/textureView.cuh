#pragma once

// -----------------------------------------------------------------------------
// TextureView — the device-facing half of the texture system.
//
// Materials store an int index into `handles`; -1 means "no texture".
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>

struct TextureView {
    const cudaTextureObject_t* handles; // device array, length == count
    int count;
};

// Returns linear-space RGBA in [0,1]; sRGB textures are decoded in hardware.
__device__ __forceinline__ float4 sampleTex(const TextureView& tv, int idx, float2 uv, float lod = 0.0f)
{
    return tex2DLod<float4>(tv.handles[idx], uv.x, uv.y, lod);
}
