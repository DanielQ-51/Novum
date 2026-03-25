#pragma once

#include "util.cuh"
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cuda_fp16.h>

// TODO consolidate uints depending on memory access patterns
struct RayQueue
{
    // x, y, z: Ray Origin
    // w: Octahedral Direction with 15 bit snorms, with delta flag packed
    float4* __restrict__ origin_plus_dir;

    // x: RGB9E5 Throughput
    // y: Packed State (26 bit pixel, 6 bit depth)
    // z: Medium Stack
    // w: Last Scattering PDF (float safely bitcast to uint)
    uint4* __restrict__ payload;
    
    __device__ __forceinline__ void getAll(
        int idx, 
        float4& outOrigin, 
        float4& outDir, 
        bool& outPrevDelta,
        float4& outThroughput,
        int& outPixelIdx,
        int& outDepth,
        unsigned int& outStack,
        float& outLastPDF
    ) const {
        // 1. One 128-bit load for Geometry
        float4 o_d = __ldg(&origin_plus_dir[idx]);
        outOrigin = make_float4(o_d.x, o_d.y, o_d.z, 1.0f);
        
        // Extract the raw bits from the float .w channel, then unpack
        unsigned int rawDir = __float_as_uint(o_d.w);
        outDir = unpackOctFlags(rawDir, &outPrevDelta, nullptr);

        // 2. One 128-bit load for Payload
        uint4 data = __ldg(&payload[idx]);
        
        outThroughput = fromRGB9E5(data.x);
        outPixelIdx   = data.y & 0x03FFFFFF;
        outDepth      = data.y >> 26;
        outStack      = data.z;
        
        // Extract the raw bits from the uint .w channel, interpret as float
        outLastPDF    = __uint_as_float(data.w);
    }

    __device__ __forceinline__ void setAll(
        int idx, 
        float4 inOrigin, 
        float4 inDir, 
        bool inPrevDelta,
        float4 inThroughput,
        int inPixelIdx,
        int inDepth,
        unsigned int inStack,
        float inLastPDF
    ) {
        // 1. Pack and write Geometry
        float4 o_d;
        o_d.x = inOrigin.x;
        o_d.y = inOrigin.y;
        o_d.z = inOrigin.z;
        o_d.w = __uint_as_float(packOctFlags(inDir, inPrevDelta, false));
        
        origin_plus_dir[idx] = o_d;

        // 2. Pack and write Payload
        uint4 data;
        data.x = toRGB9E5(inThroughput);
        data.y = (inPixelIdx & 0x03FFFFFF) | ((inDepth & 0x3F) << 26);
        data.z = inStack;
        data.w = __float_as_uint(inLastPDF);
        
        payload[idx] = data;
    }

    __device__ __forceinline__ Ray getRay(int idx) const {
        float4 o_d = __ldg(&origin_plus_dir[idx]);
        float4 dir = unpackOctFlags(__float_as_uint(o_d.w), nullptr, nullptr);
        
        // Ensure the w component of the origin is safe for matrix math if needed
        return Ray(make_float4(o_d.x, o_d.y, o_d.z, 1.0f), dir);
    }

    __device__ __forceinline__ float4 getOrigin(int idx) const {
        float4 o_d = __ldg(&origin_plus_dir[idx]);
        return make_float4(o_d.x, o_d.y, o_d.z, 1.0f);
    }

    __device__ __forceinline__ float4 getDirection(int idx) const {
        float4 o_d = __ldg(&origin_plus_dir[idx]);
        return unpackOctFlags(__float_as_uint(o_d.w), nullptr, nullptr);
    }

    __device__ __forceinline__ void getDirectionFlags(int idx, float4& dir, bool& delta) const {
        float4 o_d = __ldg(&origin_plus_dir[idx]);
        dir = unpackOctFlags(__float_as_uint(o_d.w), &delta, nullptr);
    }

    __device__ __forceinline__ float4 getThroughput(int idx) const {
        return fromRGB9E5(__ldg(&payload[idx]).x);
    }

    __device__ __forceinline__ void getPackedState(int idx, int& pixelIdx, int& depth) const {
        unsigned int state = __ldg(&payload[idx]).y;
        pixelIdx = state & 0x03FFFFFF;
        depth    = state >> 26;
    }

    __device__ __forceinline__ unsigned int getStackRaw(int idx) const {
        return __ldg(&payload[idx]).z;
    }

    __device__ __forceinline__ float getLastPDF(int idx) const {
        return __uint_as_float(__ldg(&payload[idx]).w);
    }
};

static __device__ __forceinline__ float4 getDirectionFromPacked(float4 packedRay) {
    return unpackOctFlags(__float_as_uint(packedRay.w), nullptr, nullptr);
}

static __device__ __forceinline__ int getCurrentMediumID(unsigned int stack) {
    return (int)(stack & 0xF);
}

static __device__ __forceinline__ void pushMedium(unsigned int& stack, int mediumID) {
    stack = (stack << 4) | (mediumID & 0xF);
}

static __device__ __forceinline__ void popMedium(unsigned int& stack) {
    stack >>= 4;
}

struct HitBuffer {
    float4* __restrict__ data;

    __device__ __forceinline__ void setMiss(int idx) {
        data[idx] = f4(1e30f, 0, 0, -1);
    }

    __device__ __forceinline__ void setHit(int idx, float t, float u, float v, int triID) {
        data[idx] = f4(t, u, v, __int_as_float(triID));
    }

    __device__ __forceinline__ float getT(int idx) const { return __ldg(&data[idx].x); }
    __device__ __forceinline__ float2 getUV(int idx) const { 
        float4 h = __ldg(&data[idx]);
        return make_float2(h.y, h.z); 
    }
    __device__ __forceinline__ int getID(int idx) const { 
        return __float_as_int(__ldg(&data[idx].w)); 
    }
    
    __device__ __forceinline__ bool isHit(int idx) const {
        return getT(idx) < 1e30f;
    }

    __device__ __forceinline__ void getAllInfo(int idx, float& t, float& u, float& v, int& triID) const {
        float4 h = __ldg(&data[idx]);
        
        t = h.x;
        u = h.y;
        v = h.z;
        triID = __float_as_int(h.w);
    }
};

struct ShadowQueue {
    // used for shadow ray anyhit
    float4* __restrict__ origin_plus_dist; 
    unsigned int* __restrict__ direction;   

    // shading payload
    uint2* __restrict__ payload;           


    // does NOT indicate a miss; indicates that this shadow ray doesnt exist
    __device__ __forceinline__ void setIgnore (
        int idx
    ) {
        origin_plus_dist[idx].w = -1.0f;
    }

    __device__ __forceinline__ void setShadowRay(
        int idx, float4 o, float4 dir, float maxT, float4 L, int pIdx
    ) {
        origin_plus_dist[idx] = make_float4(o.x, o.y, o.z, maxT);
        direction[idx] = packOct(dir);
        
        payload[idx] = make_uint2(toRGB9E5(L), pIdx);
    }

    __device__ __forceinline__ void setAnyHitResultAlphaTest(
        int idx, float4 throughputScale, bool unoccluded
    ) {
        if (unoccluded) {
            origin_plus_dist[idx] = throughputScale;
        } else {
            direction[idx] = 0xFFFFFFFF;
        }
    }

    __device__ __forceinline__ void setAnyHitResultNoAlphaTest(
        int idx, bool unoccluded
    ) {
        if (!unoccluded) {
            direction[idx] = 0xFFFFFFFF;
        }
    }

    __device__ __forceinline__ void getAnyHitData(
        int idx, Ray& r, float& maxT
    ) const {
        float4 o_d = __ldg(&origin_plus_dist[idx]);
        r.origin = o_d;
        maxT = o_d.w;
        
        r.direction = unpackOct(__ldg(&direction[idx]));
    }

    __device__ __forceinline__ bool getAccumulateData(
        int idx, float4& L, int& index
    ) const {
        // Fetch ONLY the 4-byte direction to check the sentinel flag
        unsigned int dir_flag = __ldg(&direction[idx]);
        
        // 0xFFFFFFFF is our sentinel for a "Blocked" ray
        if (dir_flag == 0xFFFFFFFF) return false;

        uint2 data = __ldg(&payload[idx]);
        L = fromRGB9E5(data.x);
        index = data.y;
        
        return true;
    }
};

__device__ __forceinline__ unsigned int expandBits2D(unsigned int v) {
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

__device__ __forceinline__ unsigned int getMorton18_2D(unsigned int u, unsigned int v) {
    return (expandBits2D(u) << 1) | expandBits2D(v);
}

__device__ __forceinline__ unsigned int expandBits3D(unsigned int v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

__device__ __forceinline__ unsigned int getMorton12_3D(unsigned int x, unsigned int y, unsigned int z) {
    return (expandBits3D(x) << 2) | (expandBits3D(y) << 1) | expandBits3D(z);
}

__device__ __forceinline__ unsigned int generateMortonSortKey(float4 origin, float4 dir, float4 sceneMin, float invDiameter) {
    float l1norm = fabsf(dir.x) + fabsf(dir.y) + fabsf(dir.z);
    float invL1 = (l1norm > 0.0f) ? (1.0f / l1norm) : 0.0f;
    
    float2 res;
    res.x = dir.x * invL1;
    res.y = dir.y * invL1;

    if (dir.z < 0.0f) {
        float tempX = res.x;
        float tempY = res.y;
        res.x = (1.0f - fabsf(tempY)) * (tempX >= 0.0f ? 1.0f : -1.0f);
        res.y = (1.0f - fabsf(tempX)) * (tempY >= 0.0f ? 1.0f : -1.0f);
    }

    unsigned int u = (unsigned int)fminf(fmaxf((res.x * 0.5f + 0.5f) * 511.0f, 0.0f), 511.0f);
    unsigned int v = (unsigned int)fminf(fmaxf((res.y * 0.5f + 0.5f) * 511.0f, 0.0f), 511.0f);
    unsigned int dirKey = getMorton18_2D(u, v); // 18 bits total

    unsigned int px = (unsigned int)fminf(fmaxf((origin.x - sceneMin.x) * invDiameter * 15.0f, 0.0f), 15.0f);
    unsigned int py = (unsigned int)fminf(fmaxf((origin.y - sceneMin.y) * invDiameter * 15.0f, 0.0f), 15.0f);
    unsigned int pz = (unsigned int)fminf(fmaxf((origin.z - sceneMin.z) * invDiameter * 15.0f, 0.0f), 15.0f);
    unsigned int posKey = getMorton12_3D(px, py, pz); // 12 bits total

    return (posKey << 18) | dirKey;
}

__device__ __forceinline__ unsigned int generateMaterialSortKey(int materialID, int textureIndex) {
    unsigned int mat16 = ((unsigned int)materialID) & 0xFFFFu;
    unsigned int tex16 = ((unsigned int)textureIndex) & 0xFFFFu;

    return (mat16 << 16) | tex16;
}