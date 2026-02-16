#pragma once

/*
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <curand_kernel.h>
#include <sstream>
#include <string>
#include <algorithm>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <set>
#include <iomanip>  
#include <chrono>
#include "imageUtil.cuh" 
*/

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <sstream>

__device__ __constant__ float EPSILON = 0.00001f;
__device__ __constant__ float RAY_EPSILON = 0.0001f;
__device__ __constant__ float PI = 3.141592f;
__device__ __constant__ float SKY_RADIUS = 100.0f;
__device__ __constant__ float MAX_FIREFLY_LUM = 15.0f;
__device__ __constant__ float MERGE_MAX_FIREFLY_LUM = 5.0f;
__device__ __constant__ float MERGE_ROUGHNESS_BOUND = 0.0f;

constexpr bool DO_PROGRESSIVERENDER = true;

typedef curandStatePhilox4_32_10_t cudaRNGState;

inline __host__ __device__ __forceinline__ float4 f4(float x, float y, float z, float w = 0.0f) {
    return make_float4(x, y, z, w);
}

inline __host__ __device__ __forceinline__ float4 f4() {return make_float4(0,0,0,0);}

inline __host__ __device__ __forceinline__ float4 f4(float a) {return make_float4(a,a,a,0);}

inline __host__ __device__ __forceinline__ float2 f2(float x, float y) {return make_float2(x, y);}

inline __host__ __device__ __forceinline__ float2 f2() {return make_float2(0,0);}

inline __host__ __device__ __forceinline__ float2 f2(float a) {return make_float2(a,a);}

inline __host__ __device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ __forceinline__ float4 operator-(const float4 &a, const float4 &b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator*(const float4 &a, float t) {
    return make_float4(a.x * t, a.y * t, a.z * t, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator*(float t, const float4 &a) {
    return a * t;
}

inline __host__ __device__ __forceinline__ float2 operator*(const float2 &a, float t) {
    return make_float2(a.x * t, a.y * t);
}

inline __host__ __device__ __forceinline__ float2 operator*(float t, const float2 &a) {
    return a * t;
}

inline __host__ __device__ __forceinline__ float4 operator/(const float4 &a, float t) {
    return make_float4(a.x / t, a.y / t, a.z / t, 0.0f);
}

inline __host__ __device__ __forceinline__ float4& operator+=(float4 &a, const float4 &b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float4& operator*=(float4 &a, float t) {
    a.x *= t; a.y *= t; a.z *= t;
    return a;
}

inline __host__ __device__  __forceinline__ float4& operator*=(float4& a, const float4& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

inline __host__ __device__ __forceinline__ float4& operator/=(float4 &a, float t) {
    a.x /= t; a.y /= t; a.z /= t;
    return a;
}

inline __host__ __device__ __forceinline__ float4 operator*(const float4& a, const float4& b)
{
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator/(const float4& a, const float4& b)
{
    return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 operator-(const float4& v)
{
    return make_float4(-v.x, -v.y, -v.z, -v.w);
}

inline __host__ __device__ __forceinline__ float dot(const float4 &a, const float4 &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ __forceinline__ float length(const float4 &v) {
    return sqrtf(dot(v, v));
}

inline __host__ __device__ __forceinline__ float lengthSquared(const float4 &v) {
    return dot(v, v);
}

inline __host__ __device__ __forceinline__ float4 normalize(const float4 &v) {
    float invLen = rsqrtf(dot(v, v)); // 1 / sqrt(dot(v,v))
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, 0.0f);
}

inline __host__ __device__ __forceinline__ float4 cross3(const float4 &a, const float4 &b) {
    return make_float4(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x,
        0.0f
    );
}

inline __host__ __device__ __forceinline__ float clamp(float x, float minVal, float maxVal) {
    if (x < minVal) return minVal;
    if (x > maxVal) return maxVal;
    return x;
}

inline __host__ __device__ __forceinline__ float4 clampf4(float4 v, float minVal, float maxVal) {
    return make_float4(
        clamp(v.x, minVal, maxVal),
        clamp(v.y, minVal, maxVal),
        clamp(v.z, minVal, maxVal),
        0.0f
    );
}

inline __host__ std::ostream& operator<< (std::ostream& out, const float4& v )
{
    out << "<" << v.x << " " << v.y << " " << v.z << " " << v.w << ">";
    return out; 
}

inline __device__ __forceinline__ void toWorld(const float4& wo_local, const float4& n, float4& wo_world)
{
    float4 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f4(-n.y, n.x, 0.0f));
    else
        t = normalize(f4(0.0f, -n.z, n.y));

    b = cross3(n, t);
    wo_world = wo_local.x * t + wo_local.y * b + wo_local.z * n;
}

inline __device__ __forceinline__ void toLocal(const float4& wo_world, const float4& n, float4& wo_local)
{
    float4 t, b;
    if (fabs(n.x) > fabs(n.z))
        t = normalize(f4(-n.y, n.x, 0.0f));
    else
        t = normalize(f4(0.0f, -n.z, n.y));

    b = cross3(n, t);
    wo_local = f4(dot(wo_world, t), dot(wo_world, b), dot(wo_world, n), 0.0f);
}

// Component-wise min of two float4s
inline __host__ __device__ __forceinline__ float4 fminf4(const float4 &a, const float4 &b) {
    return f4(
        fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z)
    );
}

// Component-wise max of two float4s
inline __host__ __device__ __forceinline__ float4 fmaxf4(const float4 &a, const float4 &b) {
    return f4(
        fmaxf(a.x, b.x),
        fmaxf(a.y, b.y),
        fmaxf(a.z, b.z)
    );
}

inline __host__ __device__ __forceinline__ float getFloat4Component(const float4 &v, int i) {
    switch(i) {
        case 0: return v.x;
        case 1: return v.y;
        case 2: return v.z;
        case 3: return v.w;
        default: return 0.0f; // or handle error
    }
}

inline __host__ __device__ __forceinline__ void setFloat4Component(float4 &v, int i, float value) {
    switch(i) {
        case 0: v.x = value; break;
        case 1: v.y = value; break;
        case 2: v.z = value; break;
        case 3: v.w = value; break;
    }
}

inline __host__ float surfaceArea(const float4& min, const float4& max)
{
    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;
    return 2.0f * (dx * dy + dx * dz + dy * dz);
}

__host__ __device__ __forceinline__ float4 sqrtf4(const float4& v) {
    return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), 0.0f);
}

__host__ __device__ __forceinline__ float4 rotateX(const float4& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f4(
        v.x,
        v.y * c - v.z * s,
        v.y * s + v.z * c
    );
}

__host__ __device__ __forceinline__ float4 rotateY(const float4& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f4(
        v.x * c + v.z * s,
        v.y,
        -v.x * s + v.z * c
    );
}

__host__ __device__ __forceinline__ float4 rotateZ(const float4& v, float angle)
{
    float c = cosf(angle), s = sinf(angle);
    return f4(
        v.x * c - v.y * s,
        v.x * s + v.y * c,
        v.z
    );
}

__device__ __forceinline__ float4 sampleSphere(cudaRNGState& localState, float R)
{
    float u = curand_uniform(&localState);
    float v = curand_uniform(&localState);

    float z = 1.0f - 2.0f * u;          // cos(theta) uniformly distributed
    float r = sqrtf(max(0.f, 1.f - z*z));

    float phi = 2.0f * 3.141592f * v;

    float x = r * cosf(phi);
    float y = r * sinf(phi);

    return f4(R * x, R * y, R * z);
}

__device__ inline float luminance(float4 c)
{
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__host__ inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return str;
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

__host__ inline float4 parseVec3(const std::string& val) {
    float4 v;
    std::stringstream ss(val);
    ss >> v.x >> v.y >> v.z;
    return v;
}

__host__ inline bool parseBool(const std::string& val) {
    std::string v = val;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "true");
}

__device__ __forceinline__ unsigned int toRGB9E5(float4 c)
{
    float maxRGB = fmaxf(c.x, fmaxf(c.y, c.z));
    int expShared = maxRGB <= 1.0e-9f ? -16 : (int)floorf(log2f(maxRGB));
    expShared = max(expShared, -16);
    expShared = min(expShared, 15);
    
    float denom = powf(2.0f, (float)(expShared - 9 + 15));
    int maxVal = (int)floorf(maxRGB / denom + 0.5f);
    
    if (maxVal > 511) { expShared++; denom *= 2.0f; }
    
    int r = (int)floorf(c.x / denom + 0.5f);
    int g = (int)floorf(c.y / denom + 0.5f);
    int b = (int)floorf(c.z / denom + 0.5f);
    
    return (unsigned int)((expShared + 15) << 27 | (b << 18) | (g << 9) | r);
}

__device__ __forceinline__ float4 fromRGB9E5(unsigned int packed)
{
    int exponent = (packed >> 27) - 15 - 9;
    float scale = powf(2.0f, (float)exponent);
    
    float r = (float)(packed & 0x1FF) * scale;
    float g = (float)((packed >> 9) & 0x1FF) * scale;
    float b = (float)((packed >> 18) & 0x1FF) * scale;
    
    return f4(r, g, b, 1.0f); // Default alpha to 1
}

// Encodes a unit vector into 32 bits (2x 16-bit snorm) with minimal error.
__device__ __forceinline__ float signNotZero(float k) { return (k >= 0.0f) ? 1.0f : -1.0f; }

__device__ __forceinline__ unsigned int packOct(float4 v)
{
    float l1norm = fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
    float invL1 = (l1norm > 0.0f) ? (1.0f / l1norm) : 0.0f;
    
    float2 res;
    res.x = v.x * invL1;
    res.y = v.y * invL1;

    // 2. Reflect folds if in lower hemisphere
    if (v.z < 0.0f) {
        float tempX = res.x;
        float tempY = res.y;
        res.x = (1.0f - fabsf(tempY)) * signNotZero(tempX);
        res.y = (1.0f - fabsf(tempX)) * signNotZero(tempY);
    }

    // 3. SNORM Quantization (The Fix)
    // We map [-1, 1] -> [-32767, 32767]
    // We use __float2int_rn to Round-to-Nearest, reducing error further.
    int16_t x = (int16_t)__float2int_rn(fminf(fmaxf(res.x, -1.0f), 1.0f) * 32767.0f);
    int16_t y = (int16_t)__float2int_rn(fminf(fmaxf(res.y, -1.0f), 1.0f) * 32767.0f);

    // Pack two int16s into one uint32
    // We cast to uint16_t first to ensure bits are preserved correctly during shift
    return ((uint32_t)(uint16_t)x) | (((uint32_t)(uint16_t)y) << 16);
}

__device__ __forceinline__ float4 unpackOct(uint32_t p) {
    int16_t packedX = (int16_t)(p & 0xFFFF);
    int16_t packedY = (int16_t)(p >> 16);

    // 2. Convert back to float [-1, 1]
    // Using max(-1.0f, ...) prevents -32768 (if it occurred) from breaking things
    float x = fmaxf(-1.0f, packedX / 32767.0f);
    float y = fmaxf(-1.0f, packedY / 32767.0f);

    // 3. Map back to Sphere
    float3 v;
    v.x = x;
    v.y = y;
    v.z = 1.0f - (fabsf(x) + fabsf(y));

    float t = fmaxf(-v.z, 0.0f);
    v.x += (v.x >= 0.0f) ? -t : t;
    v.y += (v.y >= 0.0f) ? -t : t;

    float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, 0.0f);
}

__device__ __forceinline__ uint32_t packOctFlags(float4 v, bool flag1, bool flag2)
{
    // 1. Octahedral Projection
    float l1norm = fabsf(v.x) + fabsf(v.y) + fabsf(v.z);
    float invL1 = (l1norm > 0.0f) ? (1.0f / l1norm) : 0.0f;

    float2 res;
    res.x = v.x * invL1;
    res.y = v.y * invL1;

    // 2. Reflect folds if in lower hemisphere
    if (v.z < 0.0f) {
        float tempX = res.x;
        float tempY = res.y;
        res.x = (1.0f - fabsf(tempY)) * signNotZero(tempX);
        res.y = (1.0f - fabsf(tempX)) * signNotZero(tempY);
    }

    // 3. SNORM Quantization (15-bit)
    // Map [-1, 1] -> [-16383, 16383]
    // We use 16383.0f because that is the max value for a signed 15-bit integer (2^14 - 1).
    int32_t x = __float2int_rn(fminf(fmaxf(res.x, -1.0f), 1.0f) * 16383.0f);
    int32_t y = __float2int_rn(fminf(fmaxf(res.y, -1.0f), 1.0f) * 16383.0f);

    // 4. Bit Packing
    // Mask x and y to ensure they fit in 15 bits (0x7FFF), then shift.
    uint32_t packedX = (uint32_t)(x & 0x7FFF);
    uint32_t packedY = (uint32_t)(y & 0x7FFF);
    
    // Shift flags to positions 30 and 31
    uint32_t packedF1 = (uint32_t)flag1 << 30;
    uint32_t packedF2 = (uint32_t)flag2 << 31;

    return packedX | (packedY << 15) | packedF1 | packedF2;
}

/**
 * Unpacks the 3D vector and writes the flags to the provided pointers.
 */
__device__ __forceinline__ float4 unpackOctFlags(uint32_t p, bool* flag1, bool* flag2) {
    // 1. Extract Flags
    if (flag1) *flag1 = (p >> 30) & 1;
    if (flag2) *flag2 = (p >> 31) & 1;

    // 2. Unpack SNORMs with Sign Extension
    // We need to interpret the 15-bit values as signed integers.
    // The "Shift Left then Shift Right" trick propagates the sign bit.
    
    // X is bottom 15 bits. Shift left 17 (32-15), then arithmetic shift right 17.
    int32_t packedX = ((int32_t)(p << 17)) >> 17;
    
    // Y is bits 15-29. Shift left 2 (remove flags), then arithmetic shift right 17 (remove X).
    int32_t packedY = ((int32_t)(p << 2)) >> 17;

    // 3. Convert back to float [-1, 1]
    float x = fmaxf(-1.0f, packedX / 16383.0f);
    float y = fmaxf(-1.0f, packedY / 16383.0f);

    // 4. Map back to Sphere (Standard Octahedral Decode)
    float3 v;
    v.x = x;
    v.y = y;
    v.z = 1.0f - (fabsf(x) + fabsf(y));

    float t = fmaxf(-v.z, 0.0f);
    v.x += (v.x >= 0.0f) ? -t : t;
    v.y += (v.y >= 0.0f) ? -t : t;

    float invLen = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float4(v.x * invLen, v.y * invLen, v.z * invLen, 0.0f);
}

__host__ inline bool IsPrime(int n) {
    if (n <= 1) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

__host__ inline int GetNextPrime(int n) {
    if (n <= 2) return 2;
    if (n % 2 == 0) n++;

    while (!IsPrime(n)) {
        n += 2;
    }
    return n;
}

__host__ inline float calculateMergeRadius(float initialRadius, float alpha, int currSample)
{
    return initialRadius * sqrtf( 1.0f / powf(currSample + 1, alpha));
}
