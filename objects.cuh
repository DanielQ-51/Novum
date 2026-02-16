#pragma once

#include "util.cuh"
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cuda_fp16.h>

struct BVHnode
{
    float4 aabbMIN;
    float4 aabbMAX;
    int left;
    int right;
    int first;
    int primCount;

    __host__ BVHnode() {}

    __host__ BVHnode(float4 min, float4 max, int l, int r, int f, int ct)
        : aabbMIN(min), aabbMAX(max), left(l), right(r), first(f), primCount(ct) {} 
};

inline void printBVH(const std::vector<BVHnode>& bvh, const std::vector<int>& indices, int level = 0, int nodeIdx = 0)
{
    if (nodeIdx >= bvh.size()) return;

    const BVHnode& node = bvh[nodeIdx];

    // Indent based on tree level
    for (int i = 0; i < level; i++) std::cout << "  ";

    std::cout << "Node " << nodeIdx 
              << " | first=" << node.first 
              << " primCount=" << node.primCount 
              << " left=" << node.left 
              << " right=" << node.right 
              << "\n";

    for (int i = 0; i < level; i++) std::cout << "  ";
    std::cout << "  AABB min=(" << node.aabbMIN.x << "," << node.aabbMIN.y << "," << node.aabbMIN.z << ")"
              << " max=(" << node.aabbMAX.x << "," << node.aabbMAX.y << "," << node.aabbMAX.z << ")\n";

    // If leaf node, print its primitives
    if (node.primCount > 0)
    {
        for (int i = 0; i < level; i++) std::cout << "  ";
        std::cout << "  Primitives: ";
        for (int i = node.first; i < node.first + node.primCount; i++)
            std::cout << indices[i] << " ";
        std::cout << "\n";
    }

    // Recurse into children
    if (node.left != -1) printBVH(bvh, indices, level + 1, node.left);
    if (node.right != -1) printBVH(bvh, indices, level + 1, node.right);
}

inline void traverseBVH(const std::vector<BVHnode>& bvh, int nodeIdx, int level,
                       int& totalNodes, 
                       std::vector<int>& leafDepths, 
                       std::vector<int>& leafSizes) 
{
    if (nodeIdx < 0 || nodeIdx >= (int)bvh.size()) return;

    const BVHnode& node = bvh[nodeIdx];
    totalNodes++;

    // Check if it's a leaf node
    if (node.left == -1 && node.right == -1) {
        leafDepths.push_back(level);
        leafSizes.push_back(node.primCount);
        return; 
    }

    // Recurse into children
    if (node.left != -1)  traverseBVH(bvh, node.left,  level + 1, totalNodes, leafDepths, leafSizes);
    if (node.right != -1) traverseBVH(bvh, node.right, level + 1, totalNodes, leafDepths, leafSizes);
}

inline void printBVHSummary(const std::vector<BVHnode>& bvh) 
{
    if (bvh.empty()) {
        std::cout << "BVH is empty.\n";
        return;
    }

    // 1. Setup local tracking variables (Allocated on stack, no leaks!)
    int totalNodes = 0;
    std::vector<int> leafDepths;
    std::vector<int> leafSizes;

    // 2. Perform the traversal
    traverseBVH(bvh, 0, 0, totalNodes, leafDepths, leafSizes);

    // 3. Perform Calculations
    int leafCount = (int)leafDepths.size();
    if (leafCount == 0) return;

    int maxDepth = *std::max_element(leafDepths.begin(), leafDepths.end());
    int largestLeaf = *std::max_element(leafSizes.begin(), leafSizes.end());

    auto mean = [](const std::vector<int>& v) {
        return v.empty() ? 0.0 : static_cast<double>(std::accumulate(v.begin(), v.end(), 0LL)) / v.size();
    };

    auto stddev = [mean](const std::vector<int>& v) {
        if (v.empty()) return 0.0;
        double m = mean(v);
        double sumSq = 0.0;
        for (int x : v) sumSq += (x - m) * (x - m);
        return std::sqrt(sumSq / v.size());
    };

    auto median = [](std::vector<int> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return n % 2 == 0 ? 0.5 * (v[n/2 - 1] + v[n/2]) : (double)v[n/2];
    };

    // 4. Print Summary
    std::cout << "\n================= BVH Summary =================\n";
    std::cout << "Total nodes:       " << totalNodes << "\n";
    std::cout << "Leaf nodes:        " << leafCount << "\n";
    std::cout << "Internal nodes:    " << (totalNodes - leafCount) << "\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << "Max leaf depth:    " << maxDepth << "\n";
    std::cout << "Average depth:     " << mean(leafDepths) << "\n";
    std::cout << "Median depth:      " << median(leafDepths) << "\n";
    std::cout << "Depth stddev:      " << stddev(leafDepths) << "\n";
    std::cout << "-----------------------------------------------\n";
    std::cout << "Largest leaf:      " << largestLeaf << " primitives\n";
    std::cout << "Average leaf size: " << mean(leafSizes) << " primitives\n";
    std::cout << "Median leaf size:  " << median(leafSizes) << " primitives\n";
    std::cout << "Leaf size stddev:  " << stddev(leafSizes) << "\n";

    // Print top 10 largest leaf sizes
    std::vector<int> sortedLeafSizes = leafSizes;
    std::sort(sortedLeafSizes.begin(), sortedLeafSizes.end(), std::greater<int>());
    std::cout << "Top 10 largest:    ";
    for (size_t i = 0; i < std::min<size_t>(10, sortedLeafSizes.size()); ++i)
        std::cout << sortedLeafSizes[i] << " ";
    
    std::cout << "\n===============================================\n\n";
}

struct Vertices
{
    float4* positions;
    float4* normals;
    float4* colors;
    float2* uvs;
};

struct Triangle
{
    int aInd;
    int bInd;
    int cInd;
    int naInd, nbInd, ncInd; // Normal indices
    //int normInd; // THIS ISNT BEING USED

    int uvaInd, uvbInd, uvcInd;
    int materialID;
    float4 emission;

    int lightInd;
    int triInd;

    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e) {}

    __host__ __device__ Triangle(int a, int b, int c, int na, int nb, int nc, int mat, int uva, int uvb, int uvc, float4 e, int lind, int tind)
        : aInd(a), bInd(b), cInd(c), naInd(na), nbInd(nb), ncInd(nc), materialID(mat), uvaInd(uva), uvbInd(uvb), uvcInd(uvc), emission(e), lightInd(lind), triInd(tind){}
};

struct Ray
{
    float4 origin;
    float4 direction;

    __host__ __device__ Ray() {}

    __host__ __device__ Ray(const float4& o, const float4& d) : origin(o), direction(d) {}

    __host__ __device__ float4 at(float t) const {return origin + t*direction;}

};

struct Camera
{
    float4 cameraOrigin;
    //float4 direction = f4(0.0f, 0.0f, -1.0f);
    int w;
    int h;

    float xRot; // in radians
    float yRot; // in radians
    float zRot; // in radians

    float aperture;
    float focalDist;
    //float imagePlaneDist; // for FOV
    float fovScale;

    float antiAliasJitterDist;

    float4 forward;
    float4 right;
    float4 up;

    __host__ static Camera Pinhole(const float4& cameraOrigin, int w, int h, float xR, float yR, float zR, float FOV, float aajitter = 2.0f)
    {
        Camera c;

        c.w = w;
        c.h = h;

        c.cameraOrigin = cameraOrigin;
        c.fovScale = tanf((FOV * 0.5f) * (3.141592f / 180.0f));
        c.xRot = xR * (3.14159265f / 180.0f);
        c.yRot = yR * (3.14159265f / 180.0f);
        c.zRot = zR * (3.14159265f / 180.0f);

        c.aperture = 0.000001f;
        c.focalDist = 1.0f/FOV;

        c.antiAliasJitterDist = aajitter;

        c.preCompute();

        return c;
    }

    __host__ static Camera NotPinhole(const float4& cameraOrigin, int w, int h, float xR, float yR, float zR, float FOV, float aperture, float focalDist, float aajitter = 2.0f)
    {
        Camera c;

        c.w = w;
        c.h = h;

        c.cameraOrigin = cameraOrigin;
        c.fovScale = tanf((FOV * 0.5f) * (3.141592f / 180.0f));
        c.xRot = xR * (3.14159265f / 180.0f);
        c.yRot = yR * (3.14159265f / 180.0f);
        c.zRot = zR * (3.14159265f / 180.0f);

        c.aperture = aperture;
        c.focalDist = focalDist;

        c.antiAliasJitterDist = aajitter;

        c.preCompute();
        return c;
    }


    // returns a camera ray with normalized direction
    __device__ Ray generateCameraRay(cudaRNGState& localState, int x, int y)
    {
        Ray r;
        float aspect = (float)w / (float)h;

        // 1. Anti-Aliasing & Screen Coords
        float jitterX = (curand_uniform(&localState) - 0.5f) * antiAliasJitterDist;
        float jitterY = (curand_uniform(&localState) - 0.5f) * antiAliasJitterDist;

        float u = (2.0f * ((x + jitterX) / (float)w) - 1.0f) * aspect * fovScale;
        float v = (2.0f * ((y + jitterY) / (float)h) - 1.0f) * fovScale;

        // 2. Calculate Focal Point using Precomputed Basis Vectors
        // This automatically handles X, Y, and Z rotation correctly.
        // Note: 'forward' corresponds to local (0,0,-1), so we move positive along 'forward'
        // to go deeper into the scene.
        float4 focalPoint = cameraOrigin + (right * (u * focalDist)) + (up * (v * focalDist)) + (forward * focalDist);

        // 3. Sample the Lens (Aperture)
        float4 lensOffset = f4(0.0f, 0.0f, 0.0f, 0.0f);
        
        if (aperture > 0.0f)
        {
            // --- CONFIGURATION ---
            int blades = 6;             // 6 = Hexagon, 5 = Pentagon, etc.
            float rotation = 0.0f;      // Rotate the bokeh shape (in radians)
            // ---------------------

            float r1 = curand_uniform(&localState); 
            float r2 = curand_uniform(&localState);

            // 1. Pick which "slice" (blade) of the aperture we are in
            float bladeIdx = floorf(r1 * (float)blades);
            
            // 2. Map r1 to the range [0, 1] within that specific slice
            // This effectively re-uses the random number to save a generator call
            float r1_local = r1 * (float)blades - bladeIdx;

            // 3. Calculate the angles for the two vertices of this triangle slice
            float twoPi = 6.283185f;
            float theta1 = ((bladeIdx) / (float)blades) * twoPi + rotation;
            float theta2 = ((bladeIdx + 1.0f) / (float)blades) * twoPi + rotation;

            // 4. Calculate the vertex positions on the aperture rim
            // We use aperture radius as the scale
            float2 v1 = make_float2(cosf(theta1), sinf(theta1));
            float2 v2 = make_float2(cosf(theta2), sinf(theta2));

            // 5. Uniformly sample the triangle defined by Center(0,0), V1, V2
            // sqrtf(r2) ensures we sample area uniformly (less density near center)
            float radiusScale = aperture * sqrtf(r2);
            
            // Linear interpolation between the two rim vertices
            float2 p = (v1 * (1.0f - r1_local) + v2 * r1_local) * radiusScale;

            float lensU = p.x;
            float lensV = p.y;

            // Apply to camera basis vectors
            lensOffset = (right * lensU) + (up * lensV);
        }

        // 4. Final Ray Construction
        r.origin = cameraOrigin + lensOffset;
        r.direction = normalize(focalPoint - r.origin);

        return r;
    }

    __host__ void preCompute()
    {
        float4 localForward = f4(0.0f, 0.0f, -1.0f, 0.0f);

        float4 worldForward = rotateX(localForward, xRot);
        worldForward = rotateY(worldForward, yRot);
        worldForward = rotateZ(worldForward, zRot);

        forward = normalize(worldForward);

        float4 localRight = f4(1.0f, 0.0f, 0.0f, 0.0f);
        right = normalize(rotateZ(rotateY(rotateX(localRight, xRot), yRot), zRot));

        float4 localUp = f4(0.0f, 1.0f, 0.0f, 0.0f);
        up = normalize(rotateZ(rotateY(rotateX(localUp, xRot), yRot), zRot));
        
    }

    __host__ __device__ float4 getForwardVector() const
    {
        return forward;
    }

    __host__ __device__ float4 getRightVector() const
    {
        return right;
    }

    __host__ __device__ float4 getUpVector() const
    {
        return up;
    }

    // dark magic
    __device__ bool worldToRaster(const float4& pointWorld, float2& pixelPos)
    {
        float4 dir = pointWorld - cameraOrigin;

        float4 fwd = getForwardVector();
        float4 right = getRightVector();
        float4 up = getUpVector();

        float distZ = dot(dir, fwd);

        if (distZ <= 0.001f) return false; 

        float distX = dot(dir, right);
        float distY = dot(dir, up);

        float slopeX = distX / distZ;
        float slopeY = distY / distZ;
        
        float aspect = (float)w / (float)h;

        float ndcX = slopeX / (aspect * fovScale);
        float ndcY = slopeY / fovScale;

        if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f) {
            return false;
        }
        
        pixelPos.x = (ndcX + 1.0f) * 0.5f * (float)w;
        pixelPos.y = (ndcY + 1.0f) * 0.5f * (float)h;

        return true;
    }
};

__device__ __forceinline__ void drawLine(float4* overlay, Camera camera, float4 p1, float4 p2, float4 color)
{
    float nearClip = 0.002f; 
    float4 camPos = camera.cameraOrigin;
    float4 camFwd = camera.forward;

    float d1 = dot(p1 - camPos, camFwd) - nearClip;
    float d2 = dot(p2 - camPos, camFwd) - nearClip;

    if (d1 < 0.0f && d2 < 0.0f) return;

    if (d1 < 0.0f) {
        float t = d1 / (d1 - d2);
        p1 = p1 + (p2 - p1) * t;
    } 
    else if (d2 < 0.0f) {
        float t = d2 / (d2 - d1);
        p2 = p2 + (p1 - p2) * t;
    }

    float2 pxf1, pxf2;

    if (!camera.worldToRaster(p1, pxf1) || !camera.worldToRaster(p2, pxf2))
        return;
    
    int x0 = (int)pxf1.x;
    int y0 = (int)pxf1.y;

    int x1 = (int)pxf2.x;
    int y1 = (int)pxf2.y;

    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy; 

    while (true) {
        if (x0 >= 0 && x0 < camera.w && y0 >= 0 && y0 < camera.h) {
            int pixelIndex = y0 * camera.w + x0;
            
            atomicExch(&overlay[pixelIndex].x, color.x);
            atomicExch(&overlay[pixelIndex].y, color.y);
            atomicExch(&overlay[pixelIndex].z, color.z);
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy) { 
            err += dy; 
            x0 += sx; 
        }
        if (e2 <= dx) { 
            err += dx; 
            y0 += sy; 
        }
    }
}

struct PathVertices {
    // material ID at this vertex
    int* materialID;
    
    // location of this vertex
    float4* pt;

    // normal at this vertex
    float4* n;

    // direction from this vertex to the previous
    float4* wo;

    // uv coordinates of this vertex
    float2* uv;

    //float4* wi; storing is not neccesary

    // throughput
    float4* beta;

    // pdf of going from previous vertex to this in area measure
    float* pdfFwd;
    
    // pdf of going from current vertex to previous vertex in area measure
    //float* pdfRev;
    
    // accumulated weight up this this vertex
    float* misWeight; 
    
    /*
    Equal to (for intermediate vertices):

    G - Geometry factor to convert a reverse pdf into area measure around the previous vertex (cos at prev, distanceSQR from previous)
    pdfFwd - Forward pdf of generating the current vertex
    PREVpdfRev - solid angle pdf of generating the vertex before the previous vertex
    d_vcm for the previous vertex
    d_vc for the previous vertex
    */
    float* d_vc;

    // stores the forward pdf for the previous vertex (area measure)
    float* d_vcm;
    bool* isDelta;
    
    int* lightInd;
    bool* backface;
};

inline __device__ __forceinline__ int pathBufferIdx(int w, int h, int x, int y, int depth)
{
    return (depth * w * h) + y * w + x;
}

// rasterizes a path defined by the pathvertices. Used for debugging and visualization
__device__ __forceinline__ void drawPath(float4* overlay, PathVertices* path, Camera camera, int x, int y, int w, 
    int depth, int maxDepth, float4 color)
{
    for (int i = 0; i < depth - 1; i++)
    {
        //float ratio = (float)(i+1) / float(depth); 
        int pathIDX1 = pathBufferIdx(w, x, y, i, maxDepth);
        int pathIDX2 = pathBufferIdx(w, x, y, i+1, maxDepth);
        drawLine(overlay, camera, path->pt[pathIDX1], path->pt[pathIDX2], color);
    }
}

static __device__ __forceinline__ void debugPrintPath(
    int w, int h, int x, int y, int maxDepth,
    const PathVertices& PV)
{
    //const int pixelIndex = y * w + x;

    // Print header first
    printf("=== PATH DUMP pixel (%d, %d) Length: %d ===\n", x, y, maxDepth);

    // Print each depth entry with its own single printf
    // (but each line is atomic so it will not mix)
    for (int depth = 0; depth < maxDepth; depth++)
    {
        int idx = pathBufferIdx(w, h, x, y, depth);

        printf(
            "Depth %d\n"
            "  materialID: %d\n"
            "  pt:   (%.3f, %.3f, %.3f)\n"
            "  n:    (%.3f, %.3f, %.3f)\n"
            "  wo:   (%.3f, %.3f, %.3f)\n"
            "  uv:   (%.3f, %.3f)\n"
            "  beta: (%.3f, %.3f, %.3f)\n"
            "  d_vc: %.6f\n"
            "  d_vcm: %.6f\n"
            "  isDelta: %d\n"
            "  lightInd: %d\n"
            "  backface: %d\n\n",

            depth,
            PV.materialID[idx],
            PV.pt[idx].x, PV.pt[idx].y, PV.pt[idx].z,
            PV.n[idx].x, PV.n[idx].y, PV.n[idx].z,
            PV.wo[idx].x, PV.wo[idx].y, PV.wo[idx].z,
            PV.uv[idx].x, PV.uv[idx].y,
            PV.beta[idx].x, PV.beta[idx].y, PV.beta[idx].z,
            PV.d_vc[idx],
            PV.d_vcm[idx],
            PV.isDelta[idx],
            PV.lightInd[idx],
            PV.backface[idx]
        );
    }
}


struct Intersection
{
    float4 point;
    float4 normal;
    float4 color;
    float4 emission;

    float2 uv;
    //Ray ray;
    //Triangle tri;
    int triIDX;
    int materialID;
    bool valid;
    bool backface;

    float dist;

    __device__ Intersection() {valid = false; uv = f2(-1.0f);};
};

enum IntegratorChoice {
    UNIDIRECTIONAL = 0,
    BIDIRECTIONAL = 1,
    NAIVE_UNIDIRECTIONAL = 2,
    VCM = 3,
    SPPM = 4
};

enum TransportMode {
    TRANSPORTMODE_IMPORTANCE = 0,
    TRANSPORTMODE_RADIANCE = 1
};

__host__ inline int matchIntegrator(std::string name)
{
    if (name == "UNIDIRECTIONAL") return 0;
    else if (name == "BIDIRECTIONAL" || name == "BDPT") return 1;
    else if (name == "NAIVE_UNIDIRECTIONAL") return 2;
    else if (name == "VCM") return 3;
    else if (name == "SPPM") return 4;

    std::cerr << "Invalid Integrator Choice!\n";
    return -1;
}

enum MaterialType {
    MAT_DIFFUSE = 0,
    MAT_METAL = 1,
    MAT_SMOOTHDIELECTRIC = 2,
    MAT_MICROFACETDIELECTRIC = 3,
    MAT_LEAF = 4,
    MAT_FLOWER = 5,
    MAT_DELTAMIRROR = 6,
    MAT_THINDIELECTRIC = 7
};

struct Material
{
    bool hasTexture;
    int startInd;
    int width;
    int height;

    bool hasTransMap;
    int tstartInd;
    int twidth;
    int theight;


    int type;
    
    float4 albedo;
    float roughness;

    float4 eta;
    float4 k;
    float ior;

    float metallic;
    float specular;
    float transmission; 

    bool isSpecular;
    bool boundary; // for mediums tack calculations

    bool thinWalled;

    float4 absorption;

    int priority; // dielectric priority, for nested dielectrics/medium stack

    __host__ Material()
        : type(MAT_DIFFUSE), albedo(f4(0.8f)),
          roughness(0.5f), eta(f4(0)), k(f4(0)),
          ior(1.5f), metallic(0.0f), specular(1.0f), transmission(0.0f) {}

    __host__ static Material Diffuse(const float4& color) {
        Material m;
        m.type = MAT_DIFFUSE;
        m.hasTexture = false;

        m.albedo = color;
        m.roughness = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;
        m.isSpecular = false;
        return m;
    }

    __host__ static Material DiffuseTextured(int sInd, int w, int h) {
        Material m;
        m.type = MAT_DIFFUSE;
        m.hasTexture = true;

        m.startInd = sInd;
        m.width = w;
        m.height = h;

        m.roughness = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;

        m.isSpecular = false;
        return m;
    }

    __host__ static Material Metal(const float4& n, const float4& k, float roughness = 0.1f) {
        Material m;
        m.type = MAT_METAL;
        m.hasTexture = false;

        m.eta = n;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);  // metals usually reflect via Fresnel, not albedo tint
        m.metallic = 1.0f;
        m.boundary = false;
        m.absorption = f4();
        m.thinWalled = false;

        m.isSpecular = false;
        return m;
    }

    __host__ static Material SmoothDielectric(float ior = 1.5f, const float4& k = f4(), int pri = 0) {
        Material m;
        m.type = MAT_SMOOTHDIELECTRIC;
        m.hasTexture = false;

        m.ior = ior;
        m.albedo = f4(1.0f);

        m.priority = pri;
        m.isSpecular = true;
        m.boundary = true;

        m.absorption = k;
        m.thinWalled = false;
        return m;
    }

    __host__ static Material ThinDielectric(float ior = 1.5f, const float4& k = f4(), int pri = 0) {
        Material m;
        m.type = MAT_THINDIELECTRIC;
        m.hasTexture = false;

        m.ior = ior;
        m.albedo = f4(1.0f);

        m.priority = pri;
        m.isSpecular = true;
        m.boundary = true;

        m.absorption = k;
        m.thinWalled = false;
        return m;
    }

    __host__  static Material MicrofacetDielectric(float ior = 1.5f, float roughness = 0.0f, const float4& k = f4()) {
        Material m;
        m.type = MAT_MICROFACETDIELECTRIC;
        m.hasTexture = false;

        m.ior = ior;
        m.k = k;
        m.roughness = roughness;
        m.albedo = f4(1.0f);

        m.thinWalled = false;
        return m;
    }

    __host__ static Material Leaf(int sInd, int w, int h,float ior = 1.5f, float roughness = 0.7, float4 albedo = f4(), float transmission = 0.05f)
    {
        Material m;
        m.type = MAT_LEAF;

        m.hasTexture = true;
        m.hasTransMap = false;

        m.ior = ior;
        m.roughness = roughness;
        m.albedo = albedo;
        m.transmission = transmission;
        m.boundary = false;

        m.startInd = sInd;
        m.width = w;
        m.height = h;

        m.thinWalled = true;

        m.isSpecular = false;

        return m;
    }

    __host__ static Material Leaf(int sInd, int w, int h, int tsInd, int tw, int th, float ior = 1.5f, float roughness = 0.7, float4 albedo = f4(), float transmission = 0.05f)
    {
        Material m;
        m.type = MAT_LEAF;

        m.hasTexture = true;
        m.hasTransMap = true;
        
        m.ior = ior;
        m.roughness = roughness;
        m.albedo = albedo;
        m.transmission = transmission;
        m.boundary = false;

        m.startInd = sInd;
        m.width = w;
        m.height = h;

        m.tstartInd = tsInd;
        m.twidth = tw;
        m.theight = th;

        m.thinWalled = true;

        m.isSpecular = false;

        return m;
    }
    
    __host__ static Material Mirror()
    {
        Material m;
        m.type = MAT_DELTAMIRROR;

        m.hasTexture = false;
        m.hasTransMap = false;

        m.isSpecular = true;

        return m;
    }
};

struct MeshConfig {
    std::string path;
    float emissionMultiplier;
    float4 emissionColor;
    int materialID;
};

struct RenderConfig {
    // Window / System
    int width = 0;
    int height = 0;

    std::string name;
    
    // Integrator Settings
    std::string integratorType;
    int sampleCount = 0;
    int maxDepth = 0;
    int bvhLeafSize = 0;
    bool sampleEnvironment = false;
    bool postProcess = false;

    // BDPT Settings
    int bdptEyeDepth = 0;
    int bdptLightDepth = 0;
    bool bdptLightTrace = false;
    bool bdptNee = false;
    bool bdptNaive = false;
    bool bdptConnection = false;
    bool bdptDrawPath = false;
    bool bdptDoMis = false;
    bool bdptPaintWeight = false;
    bool vcmDoMerge = false;
    bool doSPPM = false;

    float vcmMergeConst = 0.0f;
    float vcmInitialMergeRadiusMultiplier = 0.0f;

    // Camera
    bool pinholeCamera = false;
    float4 camPos;
    float4 camRot;
    float camFov = 0.0f;
    float camApeture = 0.0f;
    float camFocalDist = 0.0f;

    // Assets
    std::vector<MeshConfig> meshes;
};

__host__ inline bool loadConfig(const std::string& filepath, RenderConfig& config) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    bool parsingMeshes = false;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // Detect Mesh Section
        if (line.rfind("Meshes", 0) == 0) {
            parsingMeshes = true;
            continue;
        }

        if (parsingMeshes) {
            // Mesh Line Format: path; multiplier * emission; materialID
            // Example: scenedata/smallbox.obj; 1.0 * (0.0, 0.0, 0.0); 2
            
            MeshConfig mesh;
            std::stringstream ss(line);
            std::string segment;

            // 1. Path
            if(std::getline(ss, segment, ';')) mesh.path = trim(segment);

            // 2. Emission Complex Logic
            if(std::getline(ss, segment, ';')) {
                std::string complexEm = trim(segment);
                size_t starPos = complexEm.find('*');
                size_t openParen = complexEm.find('(');
                size_t closeParen = complexEm.find(')');

                if (starPos != std::string::npos && openParen != std::string::npos) {
                    // Parse Multiplier
                    mesh.emissionMultiplier = std::stof(complexEm.substr(0, starPos));
                    
                    // Parse Vector (0.0, 0.0, 0.0) -> replace commas with space for easier parsing
                    std::string vecStr = complexEm.substr(openParen + 1, closeParen - openParen - 1);
                    std::replace(vecStr.begin(), vecStr.end(), ',', ' ');
                    mesh.emissionColor = parseVec3(vecStr);
                }
            }

            // 3. Material ID
            if(std::getline(ss, segment, ';')) mesh.materialID = std::stoi(trim(segment));

            config.meshes.push_back(mesh);
        } 
        else {
            // Standard Key-Value Parsing
            size_t delimiterPos = line.find(':');
            if (delimiterPos == std::string::npos) continue; // Headers without values

            std::string key = trim(line.substr(0, delimiterPos));
            std::string value = trim(line.substr(delimiterPos + 1));

            if (value.empty()) continue; // Skip headers like "BDPT Specific Settings:"

            // Mapping
            if (key == "width") config.width = std::stoi(value);
            else if (key == "height") config.height = std::stoi(value);
            else if (key == "Integrator") config.integratorType = value;
            else if (key == "Name") config.name = value;
            else if (key == "Sample Count") config.sampleCount = std::stoi(value);
            else if (key == "Unidirectional Max Depth") config.maxDepth = std::stoi(value);
            else if (key == "BVH recommended leaf size") config.bvhLeafSize = std::stoi(value);
            else if (key == "Bidirectional Eye Depth") config.bdptEyeDepth = std::stoi(value);
            else if (key == "Bidirectional Light Depth") config.bdptLightDepth = std::stoi(value);
            
            // Booleans
            else if (key == "BDPT_LIGHTTRACE") config.bdptLightTrace = parseBool(value);
            else if (key == "BDPT_NEE") config.bdptNee = parseBool(value);
            else if (key == "BDPT_NAIVE") config.bdptNaive = parseBool(value);
            else if (key == "BDPT_CONNECTION") config.bdptConnection = parseBool(value);
            else if (key == "BDPT_DRAWPATH") config.bdptDrawPath = parseBool(value);
            else if (key == "BDPT_DOMIS") config.bdptDoMis = parseBool(value);
            else if (key == "BDPT_PAINTWEIGHT") config.bdptPaintWeight = parseBool(value);
            else if (key == "Pinhole Camera") config.pinholeCamera = parseBool(value);
            else if (key == "SAMPLE_ENVIRONMENT") config.sampleEnvironment = parseBool(value);
            else if (key == "Post Process") config.postProcess = parseBool(value);
            else if (key == "VCM_DOMERGE") config.vcmDoMerge = parseBool(value);

            // Vectors & Floats
            else if (key == "Camera Position") config.camPos = parseVec3(value);
            else if (key == "Camera Rotation") config.camRot = parseVec3(value);
            else if (key == "Camera FOV") config.camFov = std::stof(value);
            else if (key == "Camera Apeture") config.camApeture = std::stof(value);
            else if (key == "Camera FocalDist") config.camFocalDist = std::stof(value);
            else if (key == "VCM Merge Radius Power Factor") config.vcmMergeConst = std::stof(value);
            else if (key == "VCM Initial Merge Radius Multiplier") config.vcmInitialMergeRadiusMultiplier = std::stof(value);
        }
    }
    return true;
}

#define MASK_DELTA      (0x1)
#define MASK_BACKFACE   (0x2)

#define MASK_LIGHT      0xFFFFFu // 20 bits
#define MASK_MAT        0x3FFu   // 10 bits

// The raw unsigned values stored in the bitfield
#define RAW_LIGHT_ENV  0xFFFFFu // Maps to -1
#define RAW_LIGHT_NONE   0xFFFFEu // Maps to -2

/*
A highly optimized data structure to acommodate for the large spatial complexity of VCM. Please hire me
*/
struct VCMPathVertices
{
    /* To avoid the extra space of the 4th part of a float4, but also to avoid the packing issues of a float3
    separate floats are used instead of packing inside a uint because precision is important for positions.
    */
    float* pos_x; 
    float* pos_y; 
    float* pos_z;

    /*
    These are decoded into float3's essentially, but because of the covnention used in this renderer, we will turn them
    to float4's in the kernels.
    */
    unsigned int* packedNormal;
    unsigned int* packedWo;

    half* beta_x;
    half* beta_y;
    half* beta_z;

    half2* packedUV;

    /*
    bit 0: isDelta
    bit 1: backface
    bit 2-21: light index
    bit 22-31: material ID
    */
    unsigned int* packedInfo;
    
    float* d_vc;
    //float* d_vm;
    float* d_vcm;
};

__device__ __forceinline__ void setAllInfo(VCMPathVertices& x, int idx, bool isDelta, bool isBackface, int lightID, int matID) 
{
    unsigned int info = 0;
    info |= (isDelta ? 1u : 0u);
    info |= (isBackface ? 1u : 0u) << 1;
    
    // (-1 & MASK) becomes 0xFFFFF automatically
    // (-2 & MASK) becomes 0xFFFFE automatically
    info |= (unsigned int)(lightID & MASK_LIGHT) << 2; 
    
    info |= (unsigned int)(matID & MASK_MAT) << 22;
    x.packedInfo[idx] = info;
}

__device__ __forceinline__ void getAllInfo(
    const VCMPathVertices& x, 
    int idx, 
    bool& isDelta, 
    bool& isBackface, 
    int& lightID, 
    int& matID
) 
{
    unsigned int info = x.packedInfo[idx];

    // Bit 0: isDelta
    isDelta = (info & 1u);

    // Bit 1: isBackface
    isBackface = (info >> 1) & 1u;

    // Bits 2-21: lightID (20 bits)
    unsigned int rawLight = (info >> 2) & MASK_LIGHT;
    
    // Decode special flags
    if (rawLight == RAW_LIGHT_NONE) {
        lightID = -1;
    } else if (rawLight == RAW_LIGHT_ENV) {
        lightID = -2;
    } else {
        lightID = (int)rawLight;
    }

    // Bits 22-31: matID (10 bits)
    matID = (info >> 22) & MASK_MAT;
}

__device__ __forceinline__ int getLightIndex(const VCMPathVertices& x, int idx) {
    unsigned int rawLight = (x.packedInfo[idx] >> 2) & MASK_LIGHT;

    if (rawLight == RAW_LIGHT_NONE) return -1;
    if (rawLight == RAW_LIGHT_ENV)  return -2;
    return (int)rawLight;
}

__device__ __forceinline__ void setLightIndex(VCMPathVertices& x, int idx, int val) {
    unsigned int current = x.packedInfo[idx];
    
    // Clear the current light bits (Bits 2-21)
    // ~(0xFFFFF << 2) 
    current &= ~(MASK_LIGHT << 2); 
    
    // Apply new value (implicitly handles -1/-2 via masking)
    current |= (unsigned int)(val & MASK_LIGHT) << 2;
    
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ int getMaterialID(const VCMPathVertices& x, int idx) {
    return (x.packedInfo[idx] >> 22) & MASK_MAT;
}

__device__ __forceinline__ void setMaterialID(VCMPathVertices& x, int idx, int val) {
    unsigned int current = x.packedInfo[idx];
    current &= ~(MASK_MAT << 22); 
    
    current |= (unsigned int)(val & MASK_MAT) << 22;
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ bool getIsDelta(const VCMPathVertices& x, int idx) {
    return (x.packedInfo[idx] & 1u);
}

__device__ __forceinline__ bool getIsBackface(const VCMPathVertices& x, int idx) {
    return (x.packedInfo[idx] >> 1) & 1u;
}

__device__ __forceinline__ void setIsDelta(VCMPathVertices& x, int idx, bool val) {
    unsigned int current = x.packedInfo[idx];
    current &= ~MASK_DELTA; // Clear bit
    current |= (val ? 1u : 0u);
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ void setIsBackface(VCMPathVertices& x, int idx, bool val) {
    unsigned int current = x.packedInfo[idx];
    current &= ~MASK_BACKFACE;
    current |= (val ? 1u : 0u) << 1;
    x.packedInfo[idx] = current;
}

__device__ __forceinline__ float4 getPos(const VCMPathVertices& verts, int idx) 
{
    return f4(verts.pos_x[idx], verts.pos_y[idx], verts.pos_z[idx], 0.0f);
}

__device__ __forceinline__ void setPos(VCMPathVertices& verts, int idx, float4 p) 
{
    verts.pos_x[idx] = p.x;
    verts.pos_y[idx] = p.y;
    verts.pos_z[idx] = p.z;
}

__device__ __forceinline__ float4 getNormal(const VCMPathVertices& x, int idx) {
    return unpackOct(x.packedNormal[idx]);
}

__device__ __forceinline__ void setNormal(VCMPathVertices& x, int idx, float4 n) {
    x.packedNormal[idx] = packOct(n);
}

__device__ __forceinline__ float4 getWo(const VCMPathVertices& x, int idx) {
    return unpackOct(x.packedWo[idx]);
}

__device__ __forceinline__ void setWo(VCMPathVertices& x, int idx, float4 wo) {
    x.packedWo[idx] = packOct(wo);
}

__device__ __forceinline__ float4 getBeta(const VCMPathVertices& x, int idx) {
    //return fromRGB9E5(x.packedBeta[idx]);
    return f4(__half2float(x.beta_x[idx]), __half2float(x.beta_y[idx]), __half2float(x.beta_z[idx]), 0.0f);
}

__device__ __forceinline__ void setBeta(VCMPathVertices& x, int idx, float4 b) {
    //x.packedBeta[idx] = toRGB9E5(b);
    x.beta_x[idx] = __float2half(b.x);
    x.beta_y[idx] = __float2half(b.y);
    x.beta_z[idx] = __float2half(b.z);
}

__device__ __forceinline__ float2 getUV(const VCMPathVertices& x, int idx) {
    return __half22float2(x.packedUV[idx]);
}

__device__ inline void setUV(VCMPathVertices& x, int idx, float2 uv) {
    x.packedUV[idx] = __float22half2_rn(uv);
}

__device__ __forceinline__ float getD_vc(const VCMPathVertices& x, int idx) {
    return x.d_vc[idx];
}

__device__ __forceinline__ void setD_vc(VCMPathVertices& x, int idx, float val) {
    x.d_vc[idx] = val;
}

__device__ __forceinline__ float getD_vcm(const VCMPathVertices& x, int idx) {
    return x.d_vcm[idx];
}

__device__ __forceinline__ void setD_vcm(VCMPathVertices& x, int idx, float val) {
    x.d_vcm[idx] = val;
}

/*
Struct containing photon data for vcm. TODO: IMPLEMENT BACKFACE FLAG HERE. CRUCIAL IF YOU PUT DOUBLE SIDED MATERIALS!!!!
*/
struct Photons
{
    float* pos_x;
    float* pos_y;
    float* pos_z;

    // current to previous
    unsigned int* packedWi;
    unsigned int* packedNormal;
    
    half* beta_x;
    half* beta_y;
    half* beta_z;

    //float* d_vc;
    float* d_vm; 
    float* d_vcm;
};

__device__ __forceinline__ float4 getPos(const Photons& ps, int idx) 
{
    return f4(ps.pos_x[idx], ps.pos_y[idx], ps.pos_z[idx], 0.0f);
}

__device__ __forceinline__ void setPos(Photons& ps, int idx, float4 p) 
{
    ps.pos_x[idx] = p.x;
    ps.pos_y[idx] = p.y;
    ps.pos_z[idx] = p.z;
}

__device__ __forceinline__ void getNormalInfo(const Photons& x, int idx, float4& normal, bool& backface) {
    normal = unpackOctFlags(x.packedNormal[idx], &backface, nullptr);
}

__device__ __forceinline__ void setNormalInfo(Photons& x, int idx, float4 n, bool backface) {
    x.packedNormal[idx] = packOctFlags(n, backface, false);
}

__device__ __forceinline__ float4 getWi(const Photons& x, int idx) {
    return unpackOct(x.packedWi[idx]);
}

__device__ __forceinline__ void setWi(Photons& x, int idx, float4 wi) {
    x.packedWi[idx] = packOct(wi);
}

__device__ __forceinline__ float4 getBeta(const Photons& x, int idx) {
    return f4(__half2float(x.beta_x[idx]), __half2float(x.beta_y[idx]), __half2float(x.beta_z[idx]), 0.0f);
}

__device__ __forceinline__ void setBeta(Photons& x, int idx, float4 b) {
    x.beta_x[idx] = __float2half(b.x);
    x.beta_y[idx] = __float2half(b.y);
    x.beta_z[idx] = __float2half(b.z);
}

__device__ __forceinline__ float getD_vm(const Photons& x, int idx) {
    return x.d_vm[idx];
}

__device__ __forceinline__ void setD_vm(Photons& x, int idx, float val) {
    x.d_vm[idx] = val;
}

__device__ __forceinline__ float getD_vcm(const Photons& x, int idx) {
    return x.d_vcm[idx];
}

__device__ __forceinline__ void setD_vcm(Photons& x, int idx, float val) {
    x.d_vcm[idx] = val;
}