#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "util.cuh"
#include "objects.cuh"
#include <curand_kernel.h>



/**
 * Thin GBuffer for ReSTIR purposes.
 */
struct GBuffer {
    // Used for path shift rejection
    float* __restrict__ depth;
    uint32_t* __restrict__ normal;
    
    __device__ __forceinline__ inline void setPayload(
        int idx,
        float d,
        float3 n
    ) {
        normal[idx] = packOctF3(n);
        depth[idx] = d;
    }

    __device__ __forceinline__ inline float getDepth(
        int idx
    ) {
        return __ldg(&depth[idx]);
    }

    __device__ __forceinline__ inline float3 getNorm(
        int idx
    ) {
        return unpackOctF3(__ldg(&normal[idx]));
    }
};

struct ReSTIRRayQueue
{
    // x, y, z: Ray Origin
    // w: Octahedral Direction with 15 bit snorms, with delta flag and found xk packed
    float4* __restrict__ origin_plus_dir;

    // x: RGB9E5 Throughput
    // y: Packed State (26 bit pixel, 6 bit depth)
    // z: running path PDF
    // w: Last Scattering PDF (float safely bitcast to uint)
    uint4* __restrict__ payload;
    
    __device__ __forceinline__ void getAll(
        int idx, 
        float4& outOrigin, 
        float4& outDir, 
        bool& outPrevDelta,
        bool& outFoundXk,
        float4& outThroughput,
        int& outPixelIdx,
        int& outDepth,
        uint32_t& outTotalPDF,
        float& outLastPDF
    ) const {
        // 1. One 128-bit load for Geometry
        float4 o_d = __ldg(&origin_plus_dir[idx]);
        outOrigin = make_float4(o_d.x, o_d.y, o_d.z, 1.0f);
        
        // Extract the raw bits from the float .w channel, then unpack
        uint32_t rawDir = __float_as_uint(o_d.w);
        outDir = unpackOctFlags(rawDir, &outPrevDelta, &outFoundXk);

        // 2. One 128-bit load for Payload
        uint4 data = __ldg(&payload[idx]);
        
        outThroughput = fromRGB9E5(data.x);
        outPixelIdx   = data.y & 0x03FFFFFF;
        outDepth      = data.y >> 26;
        outTotalPDF   = data.z;
        
        // Extract the raw bits from the uint .w channel, interpret as float
        outLastPDF    = __uint_as_float(data.w);

        //outFootprint = __half2float(__ldg(&rayFootprint[idx]));
    }

    __device__ __forceinline__ void setAll(
        int idx, 
        float4 inOrigin, 
        float4 inDir, 
        bool inPrevDelta,
        bool inFoundXk,
        float4 inThroughput,
        int inPixelIdx,
        int inDepth,
        float inTotalPDF,
        float inLastPDF
    ) {
        // 1. Pack and write Geometry
        float4 o_d;
        o_d.x = inOrigin.x;
        o_d.y = inOrigin.y;
        o_d.z = inOrigin.z;
        o_d.w = __uint_as_float(packOctFlags(inDir, inPrevDelta, inFoundXk));
        
        origin_plus_dir[idx] = o_d;

        // 2. Pack and write Payload
        uint4 data;
        data.x = toRGB9E5(inThroughput);
        data.y = (inPixelIdx & 0x03FFFFFF) | ((inDepth & 0x3F) << 26);
        data.z = inTotalPDF;
        data.w = __float_as_uint(inLastPDF);
        
        payload[idx] = data;
        //rayFootprint[idx] = __float2half(inFootprint);
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
        uint32_t state = __ldg(&payload[idx]).y;
        pixelIdx = state & 0x03FFFFFF;
        depth    = state >> 26;
    }

    __device__ __forceinline__ float getLastPDF(int idx) const {
        return __uint_as_float(__ldg(&payload[idx]).w);
    }
};


struct ReSTIRShadowQueue {
    // used for shadow ray anyhit
    float4* __restrict__ origin_plus_dist; 
    uint32_t* __restrict__ direction;   
    uint32_t* __restrict__ pixelIdx;           


    // does NOT indicate a miss; indicates that this shadow ray doesnt exist
    __device__ __forceinline__ void setIgnore (
        int idx
    ) {
        origin_plus_dist[idx].w = -1.0f;
    }

    __device__ __forceinline__ void setShadowRay(
        int idx, float4 o, float4 dir, float maxT, int pIdx
    ) {
        origin_plus_dist[idx] = make_float4(o.x, o.y, o.z, maxT);
        direction[idx] = packOct(dir);
        
        pixelIdx[idx] = pIdx;
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
        int idx, int& index
    ) const {
        // Fetch ONLY the 4-byte direction to check the sentinel flag
        uint32_t dir_flag = __ldg(&direction[idx]);
        
        // 0xFFFFFFFF is our sentinel for a "Blocked" ray
        if (dir_flag == 0xFFFFFFFF) return false;

        index = __ldg(&pixelIdx[idx]); 
        return true;
    }
};

/**
 * ReSTIR PT reservoir, generalized for both direct and indirect lighting.
 * 
 * Minor changes for engine compatibility from the 2026 ReSTIR PT enhanced paper.
 * 
 * This one is the Canonical one, kept for reference. In practice, novum will use one more
 * optimized for its own limitations, which cuts it down to like 48 bytes or so by omitting
 * things relevant for lobe index path space extension and instance ID.
 */
struct CanonicalReservoir {
    float4* shadingData; // first channel is W, last three are F
    
    /** x: initial random seed
     *  y: reconnect vertex random seed (actually unused for single lobe version)
     *  z: pathFlags : 8 bit M, 8 bit index of rc vertex, 8 bit length, 8 bit for technique 
     *  w: rc vertex Wi (pointing to/from k+1)
     */ 
    uint4* pathIdentity;


    /** x: rc instance ID (Novum current doesnt have instance ID, so this is unused)
     *  y: rc vertex triangle index
     *  z: rc vertex barycentrics, as two 16 bit snorms, just like octahedral
     *  w: rc vertex incoming radiance (RGB9e5)
     */
    uint4* rcVertexGeometry;

    /** 
     *  x: the product 𝑝 (𝜔𝑘 ), 𝐺 (𝑥𝑘−1, 𝑥𝑘 ), 𝑝 (𝜔𝑘+1 )
     *  y: the NEE light pdf
     */
    float2* cachedValues;
};

/**
 * ReSTIR PT reservoir, generalized for both direct and indirect lighting.
 * This one is a candidate gen reservoir, which has additional parts to help with the candidate generation
 * 
 * We don't need two candidate gen reservoirs, so the second one will either be of a different type,
 * or simply just have its unused parts unallocated
 * 
 * Large changes from the 2026 ReSTIR PT enhanced paper to fit Novum's implementation
 * 
 * This version is not updated, see the updated one below.
 */
struct Reservoir1 {

    // W (Unbiased Contribution Weight)
    float* __restrict__ W;

    //(8-bit M | 8-bit rcIndex | 8-bit length | 8-bit technique 0 for bsdf 1 for nee).
    uint32_t* __restrict__ pathFlags;

    // yzw are all updated at similar times, so vectorized store/load
    uint4* __restrict__ shiftState;
    // .x -> initRandomSeed 
    //       (Grouped here so Decorrelation pass reads this chunk first).
    // .y -> F (Full RGB Integrand) via rgb9e5 
    // .z -> rcVertexRadiance 
    //       (RGB9e5 encoded).
    // .w -> __float_as_uint(neeLightPdf)
    //       (The cached light selection PDF).
    
    // vectorized load since these are load/stored at the same time
    uint4* __restrict__ rcGeometry;
    // .x -> rcVertexPrimitiveIndex
    //       (Used to fetch vertices/normals for shift target).
    // .z -> rcVertexBarycentrics
    //       (Two 16-bit unorms).
    // .y -> rcVertexWi 
    //       (Octahedral encoded incoming direction).
    // .w -> __float_as_uint(cachedJacobianProduct) 
    //       (The p(w_k) * G(x_k-1, x_k) * p(w_k+1) product).

    /**
     * This section is only touched during the candidate generation phase. Its not technically part of the reservoir,
     * but is kept here for convenience
     */

    // used instead of cachedJacobianProduct when k=d ie. the reconnection vertex is the one from which the nee branches
    // This is not touched after streaming the candidate generation.
    // float* __restrict__ modifiedMachedJacobianProduct;

    float* __restrict__ footprint;
    float* __restrict__ weightSum;
    uint32_t* __restrict__ suffixThroughput;


    uint4* __restrict__ backupRCGeometry;
    // .x -> rcVertexPrimitiveIndex
    //       (Used to fetch vertices/normals for shift target).
    // .z -> rcVertexBarycentrics
    //       (Two 16-bit unorms).
    // .y -> rcVertexWi 
    //       (Octahedral encoded incoming direction).
    // .w -> __float_as_uint(cachedJacobianProduct) 
    //       (The p(w_k) * G(x_k-1, x_k) * p(w_k+1) product).

    __device__ __forceinline__ float getFootprint(int idx) const {
        return __ldcs(&footprint[idx]);
    }

    __device__ __forceinline__ void setFootprint(int idx, float fp) {
        __stcs(&footprint[idx], fp);
    }

    __device__ __forceinline__ float getWeightSum(int idx) const {
        return __ldcs(&weightSum[idx]);
    }

    __device__ __forceinline__ void setWeightSum(int idx, float ws) {
        __stcs(&weightSum[idx], ws);
    }

    __device__ __forceinline__ float4 getSuffixThroughput(int idx) const {
        return __ldcs(&fromRGB9E5(suffixThroughput[idx]));
    }

    __device__ __forceinline__ void setSuffixThroughput(int idx, float4 st) {
        __stcs(&suffixThroughput[idx], toRGB9E5(st));
    }

    __device__ __forceinline__ void setRCGeometry(
        int idx,
        uint32_t rcPrimIdx, 
        float3 rcVertexWi, 
        float2 rcVertexBarycentrics, 
        float cachedJacobianProduct
    ) {
        half2 baryHalf = __floats2half2_rn(rcVertexBarycentrics.x, rcVertexBarycentrics.y);
        rcGeometry[idx] = make_uint4(
            rcPrimIdx, 
            packOct(f4(rcVertexWi.x, rcVertexWi.y, rcVertexWi.z)),
            reinterpret_cast<const uint32_t&>(baryHalf),
            __float_as_uint(cachedJacobianProduct)
        );
    }

    __device__ __forceinline__ void setShiftState(
        int idx,
        uint32_t seed,
        float3 F,
        float3 rcVertexRadiance,
        float neePdf
    ) {
        shiftState[idx] = make_uint4(
            seed,
            toRGB9E5(make_float4(F.x, F.y, F.z, 0.0f)),
            toRGB9E5(make_float4(rcVertexRadiance.x, rcVertexRadiance.y, rcVertexRadiance.z, 0.0f)),
            __float_as_uint(neePdf)
        );
    }

    __device__ __forceinline__ void initialize(
        int idx, 
        float W_val, 
        float3 F, 
        uint32_t seed, 
        uint32_t M, uint32_t rcIndex, uint32_t length, uint32_t technique, 
        float jacobianProduct, 
        float neePdf,
        uint32_t rcPrimIdx, 
        float3 rcVertexWi, 
        float2 rcVertexBarycentrics, 
        float3 rcVertexRadiance
    ) {
        // Unbiased Contribution Weight
        W[idx] = W_val;

        // Bitwise packing for state flags
        pathFlags[idx] = (M & 0xFF) | 
                         ((rcIndex & 0xFF) << 8) | 
                         ((length & 0xFF) << 16) | 
                         ((technique & 0xFF) << 24);

        // Core shift state mapping
        setShiftState(idx, seed, F, rcVertexRadiance, neePdf);

        // Reconnection geometry and cached Jacobian mapping
        setRCGeometry(idx, rcPrimIdx, rcVertexWi, rcVertexBarycentrics, jacobianProduct);
    }

    __device__ __forceinline__ void killWeight(int idx) {
        W[idx] = 0.0f;
    }
};

struct CandidateReservoirs {
    float* __restrict__ W; 
    // .x -> W (Unbiased Contribution Weight)

    uint4* __restrict__ payload;
    // .x -> F (Full RGB Integrand) via rgb9e5
    // .y -> pathFlags 
    //       (16-bit padding | 8-bit length | 8-bit technique 0 for bsdf 1 for nee).
    // .z -> __float_as_uint(cachedJacobianProduct) 
    //       (The p(w_k) * G(x_k-1, x_k) * p(w_k+1) product).
    // .w -> __float_as_uint(neeLightPdf)
    //       (The cached light selection PDF).

    __device__ __forceinline__ void initialize(
        uint32_t idx,
        float w,
        float3 f,
        uint32_t length,
        uint32_t technique,
        float jacobianProduct,
        float neePDF
    ) {
        __stcs(&W[idx], w);

        uint4 temp = make_uint4(
            toRGB9E5(f4(f)),
            (length << 8 | technique),
            __float_as_uint(jacobianProduct),
            __float_as_uint(neePDF)
        );

        __stcs(&payload[idx], temp);
    }

    __device__ __forceinline__ void getAll(
        uint32_t idx,
        float& w,
        float3& f,
        uint32_t& length,
        uint32_t& technique,
        float& jacobianProduct,
        float& neePDF
    ) {
        uint4 temp = __ldcs(&payload[idx]);

        f = f3(fromRGB9E5(temp.x));
        length = (temp.y >> 8);
        technique = (0x000000FF) & temp.y;
        jacobianProduct = __uint_as_float(temp.z);
        neePDF = __uint_as_float(temp.w);

        w = __uint_as_float(__ldcs(&W[idx]));
    }
};


struct UnoptimizedReservoir {
    float* __restrict__ W;
    float3* __restrict__ F;
    
    uint32_t* __restrict__ initRandomSeed;
    uint32_t* __restrict__ rcVertexRandomSeed;

    //(8-bit M | 8-bit rcIndex | 8-bit length | 8-bit technique 0 for bsdf 1 for nee).
    uint32_t* __restrict__ pathFlags;

    uint32_t* __restrict__ rcVertexInstanceID;
    uint32_t* __restrict__ rcVertexPrimitiveIndex;

    uint32_t* __restrict__ rcVertexBarycentrics;
    uint32_t* __restrict__ rcVertexWi;

    float3* __restrict__ rcVertexRadiance;

    // Cached jacobian, and hen nee pdf
    float2* __restrict__ rcVertexCachedValues;
};

struct LessUnoptimizedReservoir {
    float* __restrict__ W;
    uint32_t* __restrict__ F;
    
    uint32_t* __restrict__ initRandomSeed;
    uint32_t* __restrict__ rcVertexRandomSeed;

    //(8-bit M | 8-bit rcIndex | 8-bit length | 8-bit technique 0 for bsdf 1 for nee).
    uint32_t* __restrict__ pathFlags;

    uint32_t* __restrict__ rcVertexInstanceID;
    uint32_t* __restrict__ rcVertexPrimitiveIndex;

    uint32_t* __restrict__ rcVertexBarycentrics;
    uint32_t* __restrict__ rcVertexWi;

    uint32_t* __restrict__ rcVertexRadiance;

    // Cached jacobian, and hen nee pdf
    float2* __restrict__ rcVertexCachedValues;
};
