#pragma once
#include "objects.cuh"
#include "util.cuh"
#include "sceneContexts.cuh"
#include "optixStructs.cuh"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#ifndef USE_SER
#define USE_SER 1
#endif

struct SurfaceHit {
    bool isHit;
    float t;
    int primId;
    float2 barycentrics;
};

__device__ __forceinline__ SurfaceHit traceClosest(
    const PipelineParams& params,
    Ray r,
    float tmax = 999999999.0f) 
{
#if USE_SER == 0
    unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    optixTrace(
        params.bvh_handle,
        f3(r.origin), f3(r.direction),
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        p0, p1, p2, p3, p4
    );

    SurfaceHit hit;
    hit.isHit = (p0 == 1); 
    
    if (hit.isHit) {
        hit.t = __uint_as_float(p1);
        hit.barycentrics.x = __uint_as_float(p2);
        hit.barycentrics.y = __uint_as_float(p3);
        hit.primId = p4;
    }
    
    return hit;
#else 
    unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;

    optixTraverse(
        params.bvh_handle,
        f3(r.origin), f3(r.direction),
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        p0, p1, p2, p3, p4
    );
    SurfaceHit hit;

    unsigned int reorderKey;
    if (optixHitObjectIsHit()) {
        hit.isHit = true;
        hit.primId = optixHitObjectGetPrimitiveIndex();

        // For most levels of scene complexity and material complexity, fetching the material ID is not really worth it,
        // since most of the materials are simple anyways
        //hit.primId = optixHitObjectGetPrimitiveIndex();
        //const Triangle& tri = params.shadeContext.scene[hit.primId];
        //reorderKey = tri.materialID;

        reorderKey = 0u;
    } else {
        reorderKey = 0xFFFFFFFF;
        hit.isHit = false;
    }

    optixReorder(reorderKey, 1);

    if (hit.isHit) {
        optixInvoke(p0, p1, p2, p3, p4);

        if (hit.isHit) {
            hit.t = __uint_as_float(p1);
            hit.barycentrics.x = __uint_as_float(p2);
            hit.barycentrics.y = __uint_as_float(p3);
        }
    }
    return hit;

#endif
}

__device__ __forceinline__ bool traceVisibility(
    const PipelineParams& params, 
    Ray r,
    float targetDistance) 
{
    optixTraverse(
        params.bvh_handle,
        f3(r.origin), f3(r.direction),
        EPSILON, targetDistance, 0.0f, 
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT, 
        0, 1, 0
    );

    return optixHitObjectIsHit();
}