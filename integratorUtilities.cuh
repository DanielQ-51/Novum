#pragma once

#include "reflectors.cuh"
#include <cub/cub.cuh>
#include <random>
#include <ctime>

__device__ inline bool triangleIntersect(Vertices* verts, Triangle* tri, const Ray& r, float4& barycentric, float& tval)
{
    float4 tria = verts->positions[tri->aInd];
    float4 trib = verts->positions[tri->bInd];
    float4 tric = verts->positions[tri->cInd];
    float4 e1 = trib - tria;
    float4 e2 = tric - tria;

    float4 h = cross3(r.direction, e2);
    float a = dot(h, e1);

    if (fabsf(a) < 1e-12f) 
        return false;
    
    float f = 1.0/a;

    float4 s = r.origin-tria;
    float u = f * dot(s, h);
    float4 q = cross3(s, e1);
    float v = f * dot(r.direction, q);
    float t = f * dot(e2, q);


    if (((u >= 0) && (v >= 0) && (u + v <= 1)) && t > 0.0f)
    {
        barycentric = f4(u, v, 1.0f-u-v);
        tval = t;
        return true;
    }
    else
    {
        barycentric = f4();
        return false;
    }
}

__device__ inline bool aabbIntersect(const Ray& r, float4 minCorner, float4 maxCorner, float& tmin, float& tmax)
{
    tmin = -1e30f; // initialize to -infinity
    tmax = 1e30f;  // initialize to +infinity

    // Compute inverse ray direction once
    float4 invDir = make_float4(
        1.0f / r.direction.x,
        1.0f / r.direction.y,
        1.0f / r.direction.z,
        0.0f
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

__device__ inline void BVHSceneIntersect(const Ray& r, BVHnode* BVH, int* BVHindices, Vertices* verts, Triangle* scene, Intersection& intersect, float max_t = 999999.0f, int skipTri = -1)
{
    intersect.valid = false;
    float min_t = 3.402823466e+38f;

    int nodeStack[128];
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        BVHnode& node = BVH[currentIndex];

        // 2. If it's a leaf node, check its triangles
        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int idx = BVHindices[i];
                if (idx == skipTri) continue;
                Triangle* tri = &scene[idx];
                float4 barycentric;
                float t;
                bool hitTri = triangleIntersect(verts, tri, r, barycentric, t);

                if (hitTri && (t < min_t) && (t < max_t))
                {
                    min_t = t; // Update the closest-hit distance
                    intersect.point = r.at(t);
                    intersect.color = verts->colors[tri->aInd] * barycentric.z + 
                                        verts->colors[tri->bInd] * barycentric.x + 
                                        verts->colors[tri->cInd] * barycentric.y;
                    intersect.normal = normalize(verts->normals[tri->naInd] * barycentric.z + 
                                        verts->normals[tri->nbInd] * barycentric.x + 
                                        verts->normals[tri->ncInd] * barycentric.y);

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
                    intersect.emission = tri->emission;
                    intersect.valid = true;
                    //intersect.tri = *tri; // could be bad for performance
                    intersect.triIDX = idx;

                    intersect.dist = t;
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
                    hitLeft = aabbIntersect(r, BVH[node.left].aabbMIN, BVH[node.left].aabbMAX, tminL, tmaxL);

                // Test right child if it exists
                if (node.right >= 0)
                    hitRight = aabbIntersect(r, BVH[node.right].aabbMIN, BVH[node.right].aabbMAX, tminR, tmaxR);

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

__device__ inline void BVHShadowRay(const Ray& r, BVHnode* BVH, int* BVHindices, Vertices* verts, Triangle* scene, Material* materials, float4& throughputScale, float max_t, int skip_tri)
{
    int nodeStack[128];
    int stackTop = 0;
    nodeStack[stackTop++] = 0; // Push the root node (index 0)

    throughputScale = f4(1.0f);
    while (stackTop > 0)
    {
        // Pop the next node to check
        int currentIndex = nodeStack[--stackTop];
        BVHnode& node = BVH[currentIndex];

        // 2. If it's a leaf node, check its triangles
        if (node.primCount > 0)
        {
            for (int i = node.first; i < node.primCount + node.first; i++)
            {
                int idx = BVHindices[i];
                Triangle* tri = &scene[idx];
                float4 barycentric;
                float t;
                bool hitTri = triangleIntersect(verts, tri, r, barycentric, t);

                if (idx == skip_tri)
                    continue;

                if (hitTri && (t < max_t))
                {
                    int matID = tri->materialID;
                    if (materials[matID].type == MAT_LEAF)
                    {
                        // 1. We hit a leaf. Don't stop. Just darken the ray.
                        float4 transColor = materials[matID].albedo;
                        float transmission = materials[matID].transmission;
                        
                        float4 n = verts->normals[tri->naInd] * barycentric.z + 
                        verts->normals[tri->nbInd] * barycentric.x + 
                        verts->normals[tri->ncInd] * barycentric.y;
                        
                        float cosTheta = fabsf(dot(r.direction, normalize(n)));
                        
                        float F = schlick_fresnel(cosTheta, 1.0f, materials[matID].ior);
                        
                        throughputScale *= transColor * transmission * (1.0f - F);

                        if (fmaxf(throughputScale.x, fmaxf(throughputScale.y, throughputScale.z)) < 0.01f) 
                        {
                            throughputScale = f4(0.0f);
                            return;
                        }
                    }
                    else 
                    {
                        throughputScale = f4(0.0f);
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
                    hitLeft = aabbIntersect(r, BVH[node.left].aabbMIN, BVH[node.left].aabbMAX, tminL, tmaxL);

                // Test right child if it exists
                if (node.right >= 0)
                    hitRight = aabbIntersect(r, BVH[node.right].aabbMIN, BVH[node.right].aabbMAX, tminR, tmaxR);

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

__device__ inline void sceneIntersection(const Ray& r, Vertices* verts, Triangle* scene, int triNum, 
    Intersection& intersect)
{
    intersect.valid = false;
    float min_t = 3.402823466e+38f;
    
    for (int i = 0; i < triNum; i++)
    {
        Triangle* tri = &scene[i];
        float4 barycentric;
        float t;
        bool hitTri = triangleIntersect(verts, tri, r, barycentric, t); // returns true if it hits the tri
        if (hitTri && (t < min_t))
        {
            min_t = t; // Update the closest-hit distance
            intersect.point = r.at(t);
            intersect.color = verts->colors[tri->aInd] * barycentric.z + 
                                verts->colors[tri->bInd] * barycentric.x + 
                                verts->colors[tri->cInd] * barycentric.y;
            intersect.normal = normalize(verts->normals[tri->naInd] * barycentric.z + 
                                verts->normals[tri->nbInd] * barycentric.x + 
                                verts->normals[tri->ncInd] * barycentric.y);

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
            intersect.emission = tri->emission;
            intersect.valid = true;
            //intersect.tri = *tri; // could be bad for performance
            intersect.triIDX = i;

            intersect.dist = t;
        }
    }
}

__global__ void cleanAndFormatImage(
    float4* accumulationBuffer, // Your raw 'colors' buffer (Sum of samples)
    float4* overlayBuffer,      // Your 'overlay' buffer
    float4* outputBuffer,       // A temporary buffer to store the result for saving
    int w, int h, 
    int currentSampleCount) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= w || idy >= h) return;

    int pixelIndex = idy * w + idx;

    // 1. Read the raw accumulated color
    float4 acc = accumulationBuffer[pixelIndex];
    float4 ov = overlayBuffer[pixelIndex];
    float4 finalColor;

    // 2. Check for NaNs/Infs BEFORE normalization
    if (isnan(acc.x) || isnan(acc.y) || isnan(acc.z)) {
        finalColor = f4(1.0f, 0.0f, 1.0f);
    } 
    else if (isinf(acc.x) || isinf(acc.y) || isinf(acc.z)) {
        finalColor = f4(0.0f, 1.0f, 0.0f);
    } 
    else if (acc.x < 0 || acc.y < 0 || acc.z < 0) {
        finalColor = f4(0.0f, 0.0f, 1.0f);
    } 
    else {
        // 3. Normalize (Average the samples)
        float scale = 1.0f / (float)(currentSampleCount + 1);
        finalColor = make_float4(acc.x * scale, acc.y * scale, acc.z * scale, 1.0f);
    }

    // 4. Apply Overlay (if present)
    // Assuming overlay logic: if overlay is NOT black, it overrides the render
    if (ov.x != 0.0f || ov.y != 0.0f || ov.z != 0.0f) {
        finalColor = ov;
    }

    // 5. Write to the output buffer
    outputBuffer[pixelIndex] = finalColor;
}

__device__ int3 GetGridIndex(float4 p, float4 sceneMin, float cellSize) {
    return make_int3(
        floorf((p.x - sceneMin.x) / cellSize),
        floorf((p.y - sceneMin.y) / cellSize),
        floorf((p.z - sceneMin.z) / cellSize)
    );
}

__device__ inline unsigned int ComputeGridHash(float4 pos, float4 sceneMin, float mergeRadius, int hashTableSize) {
    int3 gridPos;
    gridPos.x = floorf((pos.x - sceneMin.x) / mergeRadius);
    gridPos.y = floorf((pos.y - sceneMin.y) / mergeRadius);
    gridPos.z = floorf((pos.z - sceneMin.z) / mergeRadius);

    gridPos.x = gridPos.x * 73856093;
    gridPos.y = gridPos.y * 19349663;
    gridPos.z = gridPos.z * 83492791;
    
    unsigned int combined = (unsigned int)(gridPos.x ^ gridPos.y ^ gridPos.z);
    unsigned int hash = combined % hashTableSize;
    return hash;
}

__device__ inline unsigned int HashGridIndex(int3 gridPos, int hashTableSize) {
    const unsigned int p1 = 73856093;
    const unsigned int p2 = 19349663;
    const unsigned int p3 = 83492791;

    unsigned int n = (p1 * gridPos.x) ^ (p2 * gridPos.y) ^ (p3 * gridPos.z);
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

__device__ inline float4 sampleSky(float4 direction)
{
    return f4();
    float4 unit_dir = normalize(direction); 

    float t = 0.5f * (unit_dir.y + 1.0f);

    //float4 c_horizon = 2.2f* f4(1.0f, 0.8f, 0.2f);
    //float4 c_zenith  = f4(0.4f, 0.4f, 0.8f);
    float4 c_horizon = 3.0f * f4(0.8f, 0.8f, 0.7f);
    //float4 c_horizon = 1.0f * f4(1.0f, 1.0f, 0.2f);
    float4 c_zenith  = f4(0.3f, 0.4f, 0.8f);
    //float4 c_zenith  = f4(0.9f, 0.9f, 0.2f);

    float4 sky_color = (1.0f - t) * c_horizon + t * c_zenith;

    float4 sun_dir = normalize(f4(-0.45f, 0.05f, 0.866f)); 
    float sun_focus = 800.0f;
    float sun_intensity = 15.0f;
    float4 sun_base = f4(1.0f, 0.8f, 0.2f);

    float sun_factor = pow(max(0.0f, dot(unit_dir, sun_dir)), sun_focus);
    float4 sun_final = sun_base * sun_intensity * sun_factor;

    return sky_color;
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

std::vector<float4> generateRandomProbes(int count, float4 sceneCenter, float sceneRadius) 
{
    std::vector<float4> probes;
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
            float4 p;
            p.x = sceneCenter.x + (u * sceneRadius);
            p.y = sceneCenter.y + (v * sceneRadius);
            p.z = sceneCenter.z + (w * sceneRadius);
            p.w = 0.0f; // Padding/Type matching

            probes.push_back(p);
        }
    }

    return probes;
}