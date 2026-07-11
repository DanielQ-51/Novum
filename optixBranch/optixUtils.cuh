#pragma once
#include "objects.cuh"
#include "util.cuh"
#include "sceneContexts.cuh"
#include "optixStructs.cuh"
#include "helpers.cuh"
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

__device__ __forceinline__ SurfaceHit traceClosestSER(
    const CommonParams& params,
    const Ray& r,
    float tmax = 999999999.0f
) {
    uint32_t p0 = 0, p1 = 0;
    optixTraverse(
        params.bvh_handle,
        f3(r.origin), f3(r.direction),
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        p0, p1
    );

    SurfaceHit hit;
    hit.isHit = optixHitObjectIsHit(); 
    if (hit.isHit) {
        hit.t = optixHitObjectGetRayTmax();
        hit.primId = optixHitObjectGetPrimitiveIndex();

        // although this is better than invoking, this is still not optimal.
        hit.barycentrics = getBarycentrics(params.shadeContext, hit.primId, r);
    }

    uint32_t reorderKey;

    if (hit.isHit) {
        hit.primId = optixHitObjectGetPrimitiveIndex();

        // For most levels of scene complexity and material complexity, fetching the material ID is not really worth it,
        // since most of the materials are simple anyways
        //hit.primId = optixHitObjectGetPrimitiveIndex();
        //const Triangle& tri = params.shadeContext.scene[hit.primId];
        //reorderKey = tri.materialID;

        reorderKey = 1;
    } else {
        reorderKey = 0;
    }

    optixReorder(reorderKey, 1);

    return hit;
}
__device__ __forceinline__ SurfaceHit traceClosestNoSER(
    const CommonParams& params,
    Ray r,
    float tmax = 999999999.0f) 
{
    uint32_t p0 = 0, p1 = 0;
    optixTraverse(
        params.bvh_handle,
        f3(r.origin), f3(r.direction),
        EPSILON, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        p0, p1
    );

    SurfaceHit hit;
    hit.isHit = optixHitObjectIsHit(); 
    if (hit.isHit) {
        hit.t = optixHitObjectGetRayTmax();
        hit.primId = optixHitObjectGetPrimitiveIndex();
    }

    // Temporary fix, since for some reason fetching barycentrics directly from hitobject does not work,
    // we defer fetching barycentrics to the closest hit shader. This is unoptimal as it forces a context
    // change to the closest hit shader. Ideally we would be able skip the sbt entirely
    optixInvoke(p0, p1);
    hit.barycentrics.x = __uint_as_float(p0);
    hit.barycentrics.y = __uint_as_float(p1);

    return hit;
}

__device__ __forceinline__ SurfaceHit traceClosest(
    const CommonParams& params,
    Ray r,
    float tmax = 999999999.0f) 
{
#if USE_SER == 1
    return traceClosestSER(params, r, tmax);
#else 
    return traceClosestNoSER(params, r, tmax);
#endif
}

__device__ __forceinline__ bool traceVisibility(
    const CommonParams& params, 
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