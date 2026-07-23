#pragma once

#include "reflectors.cuh"
#include "sceneContexts.cuh"
#include <cub/cub.cuh>
#include <random>
#include <ctime>

const __device__ bool ALPHA_TEST = false;

__device__ inline bool triangleIntersect(
    const Vertices* __restrict__ verts,
    const Triangle* __restrict__ tri,
    const Ray& r,
    float3& barycentric,
    float& tval
)
{
    float3 tria = f3(__ldg(&verts->positions[tri->aInd]));
    float3 trib = f3(__ldg(&verts->positions[tri->bInd]));
    float3 tric = f3(__ldg(&verts->positions[tri->cInd]));
    float3 e1 = trib - tria;
    float3 e2 = tric - tria;

    float3 h = cross(r.direction, e2);
    float a = dot(h, e1);

    if (fabsf(a) < EPSILON)
        return false;

    float f = 1.0/a;

    float3 s = r.origin-tria;
    float u = f * dot(s, h);
    float3 q = cross(s, e1);
    float v = f * dot(r.direction, q);
    float t = f * dot(e2, q);


    if (((u >= 0) && (v >= 0) && (u + v <= 1)) && t > 0.0f)
    {
        barycentric = f3(u, v, 1.0f-u-v);
        tval = t;
        return true;
    }
    else
    {
        barycentric = f3();
        return false;
    }
}

__device__ inline bool aabbIntersect(
    const Ray& r,
    float3 minCorner,
    float3 maxCorner,
    float& tmin,
    float& tmax
)
{
    tmin = -1e30f; // initialize to -infinity
    tmax = 1e30f;  // initialize to +infinity

    // Compute inverse ray direction once
    float3 invDir = make_float3(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    );

    // X axis
    float tx1 = (minCorner.x - r.origin.x) * invDir.x;
    float tx2 = (maxCorner.x - r.origin.x) * invDir.x;
    float tx_min = fminf(tx1, tx2);
    float tx_max = fmaxf(tx1, tx2);
    tmin = fmaxf(tmin, tx_min);
    tmax = fminf(tmax, tx_max);

    // Y axis
    float ty1 = (minCorner.y - r.origin.y) * invDir.y;
    float ty2 = (maxCorner.y - r.origin.y) * invDir.y;
    float ty_min = fminf(ty1, ty2);
    float ty_max = fmaxf(ty1, ty2);
    tmin = fmaxf(tmin, ty_min);
    tmax = fminf(tmax, ty_max);

    // Z axis
    float tz1 = (minCorner.z - r.origin.z) * invDir.z;
    float tz2 = (maxCorner.z - r.origin.z) * invDir.z;
    float tz_min = fminf(tz1, tz2);
    float tz_max = fmaxf(tz1, tz2);
    tmin = fmaxf(tmin, tz_min);
    tmax = fminf(tmax, tz_max);
    
    return (tmax >= tmin) && (tmax > 0.0f);
}

__device__ inline bool aabbIntersect(const Ray& r, float3 minCorner, float3 maxCorner, float3 invDir, float& tmin, float& tmax)
{
    tmin = -1e30f;
    tmax = 1e30f;

    // X axis
    float tx1 = (minCorner.x - r.origin.x) * invDir.x;
    float tx2 = (maxCorner.x - r.origin.x) * invDir.x;
    float tx_min = fminf(tx1, tx2);
    float tx_max = fmaxf(tx1, tx2);
    tmin = fmaxf(tmin, tx_min);
    tmax = fminf(tmax, tx_max);

    // Y axis
    float ty1 = (minCorner.y - r.origin.y) * invDir.y;
    float ty2 = (maxCorner.y - r.origin.y) * invDir.y;
    float ty_min = fminf(ty1, ty2);
    float ty_max = fmaxf(ty1, ty2);
    tmin = fmaxf(tmin, ty_min);
    tmax = fminf(tmax, ty_max);

    // Z axis
    float tz1 = (minCorner.z - r.origin.z) * invDir.z;
    float tz2 = (maxCorner.z - r.origin.z) * invDir.z;
    float tz_min = fminf(tz1, tz2);
    float tz_max = fmaxf(tz1, tz2);
    tmin = fmaxf(tmin, tz_min);
    tmax = fminf(tmax, tz_max);
    
    return (tmax >= tmin) && (tmax > 0.0f);
}


/**
 * Lightweight volume closest hit function. It is the responsibility of the caller to retrieve additional data. 
 * This function only returns the bare minimum.
 **/
__device__ inline void BVHSceneIntersect_volume(
    const Ray& r, 
    const BVHContext bvhContext,
    float3& bary,
    float& min_t, // distance to nearest surface
    float& min_vol_t, // distance to nearest volume aabb
    int& volume_ID, // ID of the intersected volume, -1 for when closest hit is a surface
    int& surface_primID // ID of the intersected surface, -1 when it misses a surface
)
{
    min_t = 1e30f;
    min_vol_t = 1e30f;
    surface_primID = -1;
    volume_ID = -1;
    bary = f3(-1.0f);

    int nodeStack[32];
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    float3 invDir = make_float3(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    );

    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        const BVHnode& node = bvhContext.BVH[currentIndex];

        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int2 idx = __ldg(&bvhContext.BVHindices[i]);
                
                if (idx.x == TYPE_VOLUME) {
                    const Volume& vol = bvhContext.volumes[idx.y];
                    float tminV, tmaxV;
                    bool hitVol = aabbIntersect(r, f3(vol.aabbMIN), f3(vol.aabbMAX), invDir, tminV, tmaxV);

                    if (hitVol && tmaxV > 0.0f)
                    {
                        // If the ray starts INSIDE the cloud, tminV is negative, clamp to 0.0f.
                        float entry_t = fmaxf(0.0f, tminV);
                        
                        // CHANGE 2: Compare entry_t to min_vol_t to properly handle overlapping/inside volumes
                        if (entry_t < min_t && entry_t < min_vol_t)
                        {
                            min_vol_t = entry_t;
                            volume_ID = idx.y;
                        }
                    }
                    
                } else if (idx.x == TYPE_TRIANGLE) {
                    const Triangle* tri = &bvhContext.scene[idx.y];
                    float3 barycentric;
                    float t;
                    bool hitTri = triangleIntersect(bvhContext.vertices, tri, r, barycentric, t);

                    if (hitTri && (t < min_t))
                    {
                        min_t = t; // Update the closest-hit distance
                        surface_primID = idx.y;
                        bary = barycentric;
                    }
                }
                
            }
        }
        else
        {
            if (node.left >= 0 || node.right >= 0)
            {
                float tminL, tmaxL, tminR, tmaxR;
                bool hitLeft = false, hitRight = false;

                // Test left child if it exists
                if (node.left >= 0) {
                    hitLeft = aabbIntersect(r, f3(bvhContext.BVH[node.left].aabbMIN), f3(bvhContext.BVH[node.left].aabbMAX), invDir, tminL, tmaxL);
                    if (tminL > min_t) hitLeft = false; // Cull!
                }

                // Test right child if it exists
                if (node.right >= 0) {
                    hitRight = aabbIntersect(r, f3(bvhContext.BVH[node.right].aabbMIN), f3(bvhContext.BVH[node.right].aabbMAX), invDir, tminR, tmaxR);
                    if (tminR > min_t) hitRight = false; // Cull!
                }

                if (hitLeft && hitRight)
                {
                    if (tminL < tminR)
                    {
                        nodeStack[stackTop++] = node.right; // farther
                        nodeStack[stackTop++] = node.left;  // nearer
                    }
                    else
                    {
                        nodeStack[stackTop++] = node.left;  // farther
                        nodeStack[stackTop++] = node.right; // nearer
                    }
                }
                else if (hitLeft)
                {
                    nodeStack[stackTop++] = node.left;
                }
                else if (hitRight)
                {
                    nodeStack[stackTop++] = node.right;
                }
            }
        }
    }

    if (min_t < min_vol_t) {
        volume_ID = -1;
    }
}

struct VolumeInterval {
    int volume_ID;
    float t_min;
    float t_max;
};

/**
 * Lightweight shadow ray traversal. 
 * Returns TRUE if fully occluded by an opaque surface.
 * Otherwise, returns FALSE and populates an array of volume intersections.
 **/
__device__ inline bool BVHShadow_volume(
    const Ray& r, 
    const BVHContext& bvhContext,
    float max_t, // Distance to the light source
    VolumeInterval* volHits, // Fixed size array allocated by caller
    int max_vol_hits, // Capacity of the array (e.g., 4 or 8)
    int& num_volHits // Out: Number of volumes intersected
)
{
    num_volHits = 0;

    int nodeStack[32];
    int stackTop = 0;
    nodeStack[stackTop++] = 0;

    float3 invDir = make_float3(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    );

    while (stackTop > 0)
    {
        int currentIndex = nodeStack[--stackTop];
        const BVHnode& node = bvhContext.BVH[currentIndex];

        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int2 idx = __ldg(&bvhContext.BVHindices[i]);
                
                if (idx.x == TYPE_TRIANGLE) {
                    const Triangle* tri = &bvhContext.scene[idx.y];
                    float3 barycentric;
                    float t;
                    bool hitTri = triangleIntersect(bvhContext.vertices, tri, r, barycentric, t);

                    // EARLY OUT: If we hit a surface closer than the light, we are in shadow.
                    if (hitTri && t < max_t)
                    {
                        return true; 
                    }
                } 
                else if (idx.x == TYPE_VOLUME) {
                    const Volume& vol = bvhContext.volumes[idx.y];
                    float tminV, tmaxV;
                    bool hitVol = aabbIntersect(r, f3(vol.aabbMIN), f3(vol.aabbMAX), invDir, tminV, tmaxV);

                    // Extract intersection segment, clamped to ray bounds [0, max_t]
                    if (hitVol && tminV < max_t && tmaxV > 0.0f)
                    {
                        if (num_volHits < max_vol_hits) {
                            volHits[num_volHits].volume_ID = idx.y;
                            volHits[num_volHits].t_min = fmaxf(0.0f, tminV);
                            volHits[num_volHits].t_max = fminf(max_t, tmaxV);
                            num_volHits++;
                        }
                    }
                }
            }
        }
        else
        {
            if (node.left >= 0 || node.right >= 0)
            {
                float tminL, tmaxL, tminR, tmaxR;
                bool hitLeft = false, hitRight = false;

                // Test left child. CULL using max_t instead of min_t
                if (node.left >= 0) {
                    hitLeft = aabbIntersect(r, f3(bvhContext.BVH[node.left].aabbMIN), f3(bvhContext.BVH[node.left].aabbMAX), invDir, tminL, tmaxL);
                    if (tminL > max_t) hitLeft = false; 
                }

                // Test right child. CULL using max_t instead of min_t
                if (node.right >= 0) {
                    hitRight = aabbIntersect(r, f3(bvhContext.BVH[node.right].aabbMIN), f3(bvhContext.BVH[node.right].aabbMAX), invDir, tminR, tmaxR);
                    if (tminR > max_t) hitRight = false; 
                }

                if (hitLeft && hitRight)
                {
                    if (tminL < tminR) {
                        nodeStack[stackTop++] = node.right; 
                        nodeStack[stackTop++] = node.left;  
                    } else {
                        nodeStack[stackTop++] = node.left;  
                        nodeStack[stackTop++] = node.right; 
                    }
                }
                else if (hitLeft) { nodeStack[stackTop++] = node.left; }
                else if (hitRight) { nodeStack[stackTop++] = node.right; }
            }
        }
    }

    // If we reach here, no opaque surface blocked the ray.
    return false; 
}

/**
 * Closest hit function, returns a full intersect object. Only supports triangles.
 */
__device__ inline void BVHSceneIntersect(
    const Ray& r, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ verts, 
    const Triangle* __restrict__ scene, 
    Intersection& intersect, 
    float max_t = 999999.0f, 
    int skipTri = -1)
{
    intersect.valid = false;
    float min_t = 1e30f;
    int bestTriInd = -1;
    float3 bestBarycentric = f3(-1.0f);

    int nodeStack[32];
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    float3 invDir = make_float3(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    );

    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        const BVHnode& node = BVH[currentIndex];

        // 2. If it's a leaf node, check its triangles
        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int2 idx = __ldg(&BVHindices[i]);
                if (idx.x == TYPE_TRIANGLE)
                {
                    if (idx.y == skipTri) continue;
                    const Triangle* tri = &scene[idx.y];
                    float3 barycentric;
                    float t;
                    bool hitTri = triangleIntersect(verts, tri, r, barycentric, t);

                    if (hitTri && (t < min_t) && (t < max_t))
                    {
                        min_t = t; // Update the closest-hit distance
                        bestTriInd = idx.y;
                        bestBarycentric = barycentric;
                    }
                }
                
            }
        }
        // 3. If it's an internal node, push its children onto the stack
        else
        {
            if (node.left >= 0 || node.right >= 0)
            {
                float tminL, tmaxL, tminR, tmaxR;
                bool hitLeft = false, hitRight = false;

                // Test left child if it exists
                if (node.left >= 0) {
                    hitLeft = aabbIntersect(r, f3(BVH[node.left].aabbMIN), f3(BVH[node.left].aabbMAX), invDir, tminL, tmaxL);
                    if (tminL > min_t) hitLeft = false; // Cull!
                }

                // Test right child if it exists
                if (node.right >= 0) {
                    hitRight = aabbIntersect(r, f3(BVH[node.right].aabbMIN), f3(BVH[node.right].aabbMAX), invDir, tminR, tmaxR);
                    if (tminR > min_t) hitRight = false; // Cull!
                }

                // If both children were hit, push the farther one first
                if (hitLeft && hitRight)
                {
                    if (tminL < tminR)
                    {
                        nodeStack[stackTop++] = node.right; // farther
                        nodeStack[stackTop++] = node.left;  // nearer
                    }
                    else
                    {
                        nodeStack[stackTop++] = node.left;  // farther
                        nodeStack[stackTop++] = node.right; // nearer
                    }
                }
                else if (hitLeft)
                {
                    nodeStack[stackTop++] = node.left;
                }
                else if (hitRight)
                {
                    nodeStack[stackTop++] = node.right;
                }
            }
        }
    }

    if (bestTriInd != -1)
    {
        const Triangle* tri = &scene[bestTriInd];
        intersect.point = r.at(min_t);
        intersect.normal = normalize(
                            f3(__ldg(&verts->normals[tri->naInd])) * bestBarycentric.z +
                            f3(__ldg(&verts->normals[tri->nbInd])) * bestBarycentric.x +
                            f3(__ldg(&verts->normals[tri->ncInd])) * bestBarycentric.y);

        intersect.uv = __ldg(&verts->uvs[tri->uvaInd]) * bestBarycentric.z + 
            __ldg(&verts->uvs[tri->uvbInd]) * bestBarycentric.x + 
            __ldg(&verts->uvs[tri->uvcInd]) * bestBarycentric.y;
        if (dot(intersect.normal, r.direction) > 0.0f) 
        {
            intersect.normal = -intersect.normal;
            intersect.backface = true;
        }
        else 
        {
            intersect.backface = false;
        }
            
        intersect.materialID = tri->materialID;
        intersect.emission = f3(tri->emission);
        intersect.valid = true;
        intersect.triIDX = bestTriInd;

        intersect.dist = min_t;
    }
}

__device__ inline void BVHSceneIntersect_lightweight(
    const Ray& r, 
    const BVHContext bvhContext,
    float3& bary,
    float& min_t,
    int& triID
)
{
    min_t = 1e30f;
    triID = -1;
    bary = f3(-1.0f);

    int nodeStack[32];
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    float3 invDir = make_float3(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    );

    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        const BVHnode& node = bvhContext.BVH[currentIndex];

        // 2. If it's a leaf node, check its triangles
        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int2 idx = __ldg(&bvhContext.BVHindices[i]);
                const Triangle* tri = &bvhContext.scene[idx.y];
                float3 barycentric;
                float t;
                bool hitTri = triangleIntersect(bvhContext.vertices, tri, r, barycentric, t);

                if (hitTri && (t < min_t))
                {
                    min_t = t; // Update the closest-hit distance
                    triID = idx.y;
                    bary = barycentric;
                }
            }
        }
        // 3. If it's an internal node, push its children onto the stack
        else
        {
            if (node.left >= 0 || node.right >= 0)
            {
                float tminL, tmaxL, tminR, tmaxR;
                bool hitLeft = false, hitRight = false;

                // Test left child if it exists
                if (node.left >= 0)
                    hitLeft = aabbIntersect(r, f3(bvhContext.BVH[node.left].aabbMIN), f3(bvhContext.BVH[node.left].aabbMAX), invDir, tminL, tmaxL);

                // Test right child if it exists
                if (node.right >= 0)
                    hitRight = aabbIntersect(r, f3(bvhContext.BVH[node.right].aabbMIN), f3(bvhContext.BVH[node.right].aabbMAX), invDir, tminR, tmaxR);

                // If both children were hit, push the farther one first
                if (hitLeft && hitRight)
                {
                    if (tminL < tminR)
                    {
                        nodeStack[stackTop++] = node.right; // farther
                        nodeStack[stackTop++] = node.left;  // nearer
                    }
                    else
                    {
                        nodeStack[stackTop++] = node.left;  // farther
                        nodeStack[stackTop++] = node.right; // nearer
                    }
                }
                else if (hitLeft)
                {
                    nodeStack[stackTop++] = node.left;
                }
                else if (hitRight)
                {
                    nodeStack[stackTop++] = node.right;
                }
            }
        }
    }
}

__device__ inline void BVHShadowRay(
    const Ray& r, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ verts, 
    const Triangle* __restrict__ scene, 
    const Material* __restrict__ materials,
    float3& throughputScale,
    float max_t, 
    int skip_tri = -1
)
{
    int nodeStack[32];
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    float3 invDir = make_float3(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    );

    throughputScale = f3(1.0f);
    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        const BVHnode& node = BVH[currentIndex];

        // 2. If it's a leaf node, check its triangles
        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int2 idx = __ldg(&BVHindices[i]);
                const Triangle* tri = &scene[idx.y];
                float3 barycentric;
                float t;
                bool hitTri = triangleIntersect(verts, tri, r, barycentric, t);

                if (idx.y == skip_tri)
                    continue;

                if (hitTri && (t < max_t))
                {
                    // 1. THE FAST PATH: If we aren't alpha testing, ANY hit is a hard shadow.
                    if (!ALPHA_TEST)
                    {
                        throughputScale = f3(0.0f);
                        return; 
                    }

                    // 2. THE SLOW PATH: Alpha testing is on, so we must evaluate the material.
                    int matID = tri->materialID; 
                    
                    if (materials[matID].type == MAT_LEAF)
                    {
                        // We hit a leaf. Don't stop. Just darken the ray.
                        float3 transColor = f3(materials[matID].albedo);
                        float transmission = materials[matID].transmission;
                        
                        float3 n = f3(verts->normals[tri->naInd]) * barycentric.z +
                                f3(verts->normals[tri->nbInd]) * barycentric.x +
                                f3(verts->normals[tri->ncInd]) * barycentric.y;
                        
                        float cosTheta = fabsf(dot(r.direction, normalize(n)));
                        float F = schlick_fresnel(cosTheta, 1.0f, materials[matID].ior);
                        
                        throughputScale *= transColor * transmission * (1.0f - F);

                        if (fmaxf(throughputScale.x, fmaxf(throughputScale.y, throughputScale.z)) < 0.01f)
                        {
                            throughputScale = f3(0.0f);
                            return;
                        }
                    }
                    else 
                    {
                        // It's a solid object, block the shadow.
                        throughputScale = f3(0.0f);
                        return;
                    }
                }
            }
        }
        else
        {
            if (node.left >= 0 || node.right >= 0)
            {
                float tminL, tmaxL, tminR, tmaxR;
                bool hitLeft = false, hitRight = false;

                // Test left child if it exists
                if (node.left >= 0)
                    hitLeft = aabbIntersect(r, f3(BVH[node.left].aabbMIN), f3(BVH[node.left].aabbMAX), invDir, tminL, tmaxL);

                // Test right child if it exists
                if (node.right >= 0)
                    hitRight = aabbIntersect(r, f3(BVH[node.right].aabbMIN), f3(BVH[node.right].aabbMAX), invDir, tminR, tmaxR);

                // If both children were hit, push the farther one first
                if (hitLeft && hitRight)
                {
                    if (tminL < tminR)
                    {
                        nodeStack[stackTop++] = node.right; // farther
                        nodeStack[stackTop++] = node.left;  // nearer
                    }
                    else
                    {
                        nodeStack[stackTop++] = node.left;  // farther
                        nodeStack[stackTop++] = node.right; // nearer
                    }
                }
                else if (hitLeft)
                {
                    nodeStack[stackTop++] = node.left;
                }
                else if (hitRight)
                {
                    nodeStack[stackTop++] = node.right;
                }
            }
        }
    }
}

__device__ inline void BVHShadowRay_NoAlpha(
    const Ray& r, 
    const BVHnode* __restrict__ BVH, 
    const int2* __restrict__ BVHindices, 
    const Vertices* __restrict__ verts, 
    const Triangle* __restrict__ scene,
    float3& throughputScale,
    float max_t
)
{
    int nodeStack[32];
    int stackTop = 0;
    nodeStack[stackTop++] = 0;

    float3 invDir = make_float3(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z
    );

    throughputScale = f3(1.0f);

    while (stackTop > 0)
    {
        int currentIndex = nodeStack[--stackTop];
        const BVHnode& node = BVH[currentIndex];

        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int2 idx = __ldg(&BVHindices[i]);
                const Triangle* tri = &scene[idx.y];
                
                float3 barycentric;
                float t;
                bool hitTri = triangleIntersect(verts, tri, r, barycentric, t);

                if (hitTri && (t < max_t))
                {
                    throughputScale = f3(0.0f);
                    return; 
                }
            }
        }
        else
        {
            float tminL, tmaxL, tminR, tmaxR;
            bool hitLeft = false, hitRight = false;

            if (node.left >= 0)
                hitLeft = aabbIntersect(r, f3(BVH[node.left].aabbMIN), f3(BVH[node.left].aabbMAX), invDir, tminL, tmaxL);

            if (node.right >= 0)
                hitRight = aabbIntersect(r, f3(BVH[node.right].aabbMIN), f3(BVH[node.right].aabbMAX), invDir, tminR, tmaxR);


            if (hitLeft && hitRight)
            {
                if (tminL < tminR)
                {
                    nodeStack[stackTop++] = node.right; // farther
                    nodeStack[stackTop++] = node.left;  // nearer
                }
                else
                {
                    nodeStack[stackTop++] = node.left;  // farther
                    nodeStack[stackTop++] = node.right; // nearer
                }
            }
            else if (hitLeft)
            {
                nodeStack[stackTop++] = node.left;
            }
            else if (hitRight)
            {
                nodeStack[stackTop++] = node.right;
            }
        }
    }
}

__device__ inline void sceneIntersection(const Ray& r, Vertices* verts, Triangle* scene, int triNum, 
    Intersection& intersect)
{
    intersect.valid = false;
    float min_t = 3.402823466e+38f;
    
    for (int i = 0; i < triNum; i++)
    {
        Triangle* tri = &scene[i];
        float3 barycentric;
        float t;
        bool hitTri = triangleIntersect(verts, tri, r, barycentric, t); // returns true if it hits the tri
        if (hitTri && (t < min_t))
        {
            min_t = t; // Update the closest-hit distance
            intersect.point = r.at(t);
            intersect.normal = normalize(f3(verts->normals[tri->naInd]) * barycentric.z +
                                f3(verts->normals[tri->nbInd]) * barycentric.x +
                                f3(verts->normals[tri->ncInd]) * barycentric.y);

            intersect.uv = verts->uvs[tri->uvaInd] * barycentric.z + 
                verts->uvs[tri->uvbInd] * barycentric.x + 
                verts->uvs[tri->uvcInd] * barycentric.y;
            if (dot(intersect.normal, r.direction) > 0.0f) 
            {
                intersect.normal = -intersect.normal;
                intersect.backface = true;
            }
            else 
            {
                intersect.backface = false;
            }
                
            intersect.materialID = tri->materialID;
            intersect.emission = f3(tri->emission);
            intersect.valid = true;
            //intersect.tri = *tri; // could be bad for performance
            intersect.triIDX = i;

            intersect.dist = t;
        }
    }
}

__device__ inline int3 GetGridIndex(float3 p, float3 sceneMin, float cellSize) {
    return make_int3(
        floorf((p.x - sceneMin.x) / cellSize),
        floorf((p.y - sceneMin.y) / cellSize),
        floorf((p.z - sceneMin.z) / cellSize)
    );
}

__device__ inline uint32_t ComputeGridHash(float3 pos, float3 sceneMin, float mergeRadius, int hashTableSize) {
    int3 gridPos;
    gridPos.x = floorf((pos.x - sceneMin.x) / mergeRadius);
    gridPos.y = floorf((pos.y - sceneMin.y) / mergeRadius);
    gridPos.z = floorf((pos.z - sceneMin.z) / mergeRadius);

    gridPos.x = gridPos.x * 73856093;
    gridPos.y = gridPos.y * 19349663;
    gridPos.z = gridPos.z * 83492791;
    
    uint32_t combined = (uint32_t)(gridPos.x ^ gridPos.y ^ gridPos.z);
    uint32_t hash = combined % hashTableSize;
    return hash;
}

__device__ inline uint32_t HashGridIndex(int3 gridPos, int hashTableSize) {
    const uint32_t p1 = 73856093;
    const uint32_t p2 = 19349663;
    const uint32_t p3 = 83492791;

    uint32_t n = (p1 * gridPos.x) ^ (p2 * gridPos.y) ^ (p3 * gridPos.z);
    return n % hashTableSize;
}

__device__ inline void removeMaterialFromStack(int* stack, int* stackTop, int materialID)
{
    int i_found = -1;
    for (int i = (*stackTop) - 1; i > 0; i--)
    {
        if (stack[i] == materialID)
        {
            i_found = i;
            break;
        }
    }

    if (i_found != -1)
    {
        for (int i = i_found; i < (*stackTop) - 1; i++)
        {
            stack[i] = stack[i + 1];
        }
        (*stackTop)--;
    }
}

__device__ inline float3 sampleSky(float3 direction)
{
    float3 unit_dir = normalize(direction);

    // Maps Y from [-1, 1] to [0, 1] for the gradient
    float t = 0.5f * (unit_dir.y + 1.0f);

    // Reduced multipliers to prevent extreme clipping
    // Horizon is usually brighter than the zenith, but 3.0 is very high
    float3 c_horizon = 4.5f * f3(0.8f, 0.3f, 0.1f);
    float3 c_zenith  = f3(0.3f, 0.2f, 0.9f);

    float3 sky_color = (1.0f - t) * c_horizon + t * c_zenith;

    // Sun Calculation
    //float3 sun_dir = normalize(f3(-0.45f, 0.05f, 0.866f));
    float3 sun_dir = normalize(f3(-0.65f, 0.05f, -0.866f));
    float sun_focus = 800.0f;     // Higher = smaller, sharper sun
    float sun_intensity = 10.0f;  // Sun should be much brighter than sky
    float3 sun_base = f3(1.0f, 0.8f, 0.2f);

    float sun_factor = pow(max(0.0f, dot(unit_dir, sun_dir)), sun_focus);
    float3 sun_final = sun_base * sun_intensity * sun_factor;

    // IMPORTANT: Add the sun to the sky!
    return sky_color + sun_final;
}

__host__ inline void checkCudaErrors(const char * name)
{
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("!!! At %s Kernel Launch Failed: %s !!!\n", name, cudaGetErrorString(launchErr));
    }

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("!!! At %s Kernel Execution Crashed: %s !!!\n", name, cudaGetErrorString(syncErr));
    }
}

std::vector<float3> inline generateRandomProbes(int count, float3 sceneCenter, float sceneRadius)
{
    std::vector<float3> probes;
    probes.reserve(count);

    // Initialize random number generator
    static std::mt19937 rng(std::time(nullptr)); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    while (probes.size() < count) 
    {
        // Generate a random point in a unit cube [-1, 1]
        float u = dist(rng);
        float v = dist(rng);
        float w = dist(rng);

        // Rejection sampling: Only keep points inside the unit sphere
        // to avoid "corner bias"
        if ((u * u + v * v + w * w) <= 1.0f) 
        {
            float3 p;
            p.x = sceneCenter.x + (u * sceneRadius);
            p.y = sceneCenter.y + (v * sceneRadius);
            p.z = sceneCenter.z + (w * sceneRadius);

            probes.push_back(p);
        }
    }

    return probes;
}