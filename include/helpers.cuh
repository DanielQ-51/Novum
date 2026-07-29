#pragma once

#include "sceneContexts.cuh"
#include "util.cuh"
#include "objects.cuh"

__device__ __forceinline__ float3 transformPosition(
    const float4& r0, const float4& r1, const float4& r2, 
    const float3& localPos) 
{
    return make_float3(
        r0.x * localPos.x + r0.y * localPos.y + r0.z * localPos.z + r0.w,
        r1.x * localPos.x + r1.y * localPos.y + r1.z * localPos.z + r1.w,
        r2.x * localPos.x + r2.y * localPos.y + r2.z * localPos.z + r2.w
    );
}

__device__ __forceinline__ float3 transformNormalRigid(
    const float4& r0, const float4& r1, const float4& r2, 
    const float3& localNorm) 
{
    float3 worldNorm = make_float3(
        r0.x * localNorm.x + r0.y * localNorm.y + r0.z * localNorm.z,
        r1.x * localNorm.x + r1.y * localNorm.y + r1.z * localNorm.z,
        r2.x * localNorm.x + r2.y * localNorm.y + r2.z * localNorm.z
    );
    return normalize(worldNorm);
}

__device__ __forceinline__ float3 transformNormalRobust(
    const float4& r0, const float4& r1, const float4& r2, 
    const float3& localNorm) 
{
    float3 v0 = make_float3(r0.x, r0.y, r0.z);
    float3 v1 = make_float3(r1.x, r1.y, r1.z);
    float3 v2 = make_float3(r2.x, r2.y, r2.z);

    float3 c0 = cross(v1, v2);
    float3 c1 = cross(v2, v0);
    float3 c2 = cross(v0, v1);

    float3 worldNorm = localNorm.x * c0 + localNorm.y * c1 + localNorm.z * c2;

    float determinant = dot(v0, c0); 
    if (determinant < 0.0f) {
        worldNorm = make_float3(-worldNorm.x, -worldNorm.y, -worldNorm.z);
    }

    return normalize(worldNorm);
}

__device__ __forceinline__ float3 transformPosition(
    const float4* matrices, unsigned int instanceId, const float3& localPos) 
{
    int offset = instanceId * 3;
    return transformPosition(
        matrices[offset], 
        matrices[offset + 1], 
        matrices[offset + 2], 
        localPos
    );
}

__device__ __forceinline__ float3 transformNormalRigid(
    const float4* matrices, unsigned int instanceId, const float3& localNorm) 
{
    int offset = instanceId * 3;
    return transformNormalRigid(
        matrices[offset], 
        matrices[offset + 1], 
        matrices[offset + 2], 
        localNorm
    );
}

__device__ __forceinline__ float3 transformNormalRobust(
    const float4* matrices, unsigned int instanceId, const float3& localNorm) 
{
    int offset = instanceId * 3;
    return transformNormalRobust(
        matrices[offset],
        matrices[offset + 1],
        matrices[offset + 2],
        localNorm
    );
}

// Recovers the hit's barycentrics with a Moller-Trumbore solve against the triangle
// we were told we hit. `r` is a world-space ray but the vertex buffer is object space,
// so under a TLAS the triangle has to be pushed through its instance transform first --
// otherwise the two operands live in different spaces and (u, v) is garbage. Passing
// 0xFFFFFFFF (the default) skips that, for callers tracing a bare GAS.
//
// Barycentrics are affine invariant, so transforming the triangle to world space and
// transforming the ray to object space are equivalent; we do the former because it
// needs only the forward matrix, and no inverse is stored.
__device__ inline float2 getBarycentrics(
    const ShadeContext sc,
    unsigned int triIndex,
    const Ray& r,
    uint32_t instanceID = 0xFFFFFFFF
)
{
    if (triIndex >= sc.triNum) {
        return f2();
    }
    const Triangle& tri = sc.scene[triIndex];

    float3 tria = f3(__ldg(&sc.vertices->positions[tri.aInd]));
    float3 trib = f3(__ldg(&sc.vertices->positions[tri.bInd]));
    float3 tric = f3(__ldg(&sc.vertices->positions[tri.cInd]));

    if (instanceID != 0xFFFFFFFF) {
        const float4* m = sc.transformationMatrices;
        const int offset = instanceID * 3;
        tria = transformPosition(m[offset], m[offset + 1], m[offset + 2], tria);
        trib = transformPosition(m[offset], m[offset + 1], m[offset + 2], trib);
        tric = transformPosition(m[offset], m[offset + 1], m[offset + 2], tric);
    }

    float3 e1 = trib - tria;
    float3 e2 = tric - tria;

    float3 h = cross(r.direction, e2);
    float a = dot(h, e1);

    float f = 1.0f / a;

    float3 s = r.origin - tria;
    float u = f * dot(s, h);
    float3 q = cross(s, e1);
    float v = f * dot(r.direction, q);

    return f2(u, v);
}

// readObjSimple stores -1 for every index an OBJ does not supply: a mesh with no
// "vt" lines gets uv*Ind == -1, and one with no "vn" lines gets n*Ind == -1.
// Interpolating those unguarded reads one element before the buffer, which is a real
// out-of-bounds access whenever the array happens to start a cudaMalloc allocation.
__device__ __forceinline__ float2 interpolateUV(
    const Triangle& tri, const ShadeContext& sc, float u, float v)
{
    if (tri.uvaInd < 0 || tri.uvbInd < 0 || tri.uvcInd < 0) {
        return f2(0.0f);
    }

    return __ldg(&sc.vertices->uvs[tri.uvaInd]) * (1.0f - u - v) +
           __ldg(&sc.vertices->uvs[tri.uvbInd]) * u +
           __ldg(&sc.vertices->uvs[tri.uvcInd]) * v;
}

// Falls back to the geometric normal when the mesh carries no shading normals.
__device__ __forceinline__ float3 interpolateNormal(
    const Triangle& tri, const ShadeContext& sc, float u, float v,
    const float3& apos, const float3& bpos, const float3& cpos)
{
    if (tri.naInd < 0 || tri.nbInd < 0 || tri.ncInd < 0) {
        return normalize(cross(bpos - apos, cpos - apos));
    }

    float3 a_n = f3(__ldg(&sc.vertices->normals[tri.naInd]));
    float3 b_n = f3(__ldg(&sc.vertices->normals[tri.nbInd]));
    float3 c_n = f3(__ldg(&sc.vertices->normals[tri.ncInd]));

    return (1.0f - u - v) * a_n + u * b_n + v * c_n;
}

__device__ __forceinline__ void getData(
    const Triangle& tri,
    ShadeContext shadeContext,
    float2 barycentrics,
    float3 inDirection,

    int& materialID,
    float2& uv,
    float3& shadingPos,
    float3& normal,
    bool& backface,
    float3& emission,

    uint32_t instanceID = 0xFFFFFFFF
) {
    materialID = tri.materialID;
    float u = barycentrics.x;
    float v = barycentrics.y;

    uv = interpolateUV(tri, shadeContext, u, v);

    float3 apos = f3(__ldg(&shadeContext.vertices->positions[tri.aInd]));
    float3 bpos = f3(__ldg(&shadeContext.vertices->positions[tri.bInd]));
    float3 cpos = f3(__ldg(&shadeContext.vertices->positions[tri.cInd]));

    shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

    normal = interpolateNormal(tri, shadeContext, u, v, apos, bpos, cpos);

    // `inDirection` is a world-space direction, so the normal has to reach world space
    // before it is used for the backface test.
    if (instanceID != 0xFFFFFFFF) {
        shadingPos = transformPosition(shadeContext.transformationMatrices, instanceID, shadingPos);
        normal = transformNormalRigid(shadeContext.transformationMatrices, instanceID, normal);
    }

    backface = dot(normal, inDirection) > 0.0f;
    normal = backface ? -normal : normal;
    emission = f3(tri.emission);
}

__device__ __forceinline__ void getDataWithoutInDirection(
    const Triangle& tri,
    ShadeContext shadeContext,
    float2 barycentrics,
    float3 origin,

    int& materialID,
    float2& uv,
    float3& shadingPos,
    float3& normal,
    bool& backface,
    float3& emission,

    uint32_t instanceID = 0xFFFFFFFF
) {
    materialID = tri.materialID;
    float u = barycentrics.x;
    float v = barycentrics.y;

    uv = interpolateUV(tri, shadeContext, u, v);

    float3 apos = f3(__ldg(&shadeContext.vertices->positions[tri.aInd]));
    float3 bpos = f3(__ldg(&shadeContext.vertices->positions[tri.bInd]));
    float3 cpos = f3(__ldg(&shadeContext.vertices->positions[tri.cInd]));

    shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

    normal = interpolateNormal(tri, shadeContext, u, v, apos, bpos, cpos);

    // World space first, then the backface test -- see getDataWithoutInDirectionAndEmission.
    if (instanceID != 0xFFFFFFFF) {
        shadingPos = transformPosition(shadeContext.transformationMatrices, instanceID, shadingPos);
        normal = transformNormalRigid(shadeContext.transformationMatrices, instanceID, normal);
    }

    float3 inDirection = normalize(shadingPos - origin);
    backface = dot(normal, inDirection) > 0.0f;
    normal = backface ? -normal : normal;
    emission = f3(tri.emission);
}

__device__ __forceinline__ void getDataSkipEmission(
    const Triangle& tri,
    ShadeContext shadeContext,
    float2 barycentrics,
    float3 inDirection,

    int& materialID,
    float2& uv,
    float3& shadingPos,
    float3& normal,
    bool& backface,

    uint32_t instanceID = 0xFFFFFFFF
) {
    materialID = tri.materialID;
    float u = barycentrics.x;
    float v = barycentrics.y;

    uv = interpolateUV(tri, shadeContext, u, v);

    float3 apos = f3(__ldg(&shadeContext.vertices->positions[tri.aInd]));
    float3 bpos = f3(__ldg(&shadeContext.vertices->positions[tri.bInd]));
    float3 cpos = f3(__ldg(&shadeContext.vertices->positions[tri.cInd]));

    shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

    normal = interpolateNormal(tri, shadeContext, u, v, apos, bpos, cpos);

    // `inDirection` is a world-space direction, so the normal has to reach world space
    // before it is used for the backface test.
    if (instanceID != 0xFFFFFFFF) {
        shadingPos = transformPosition(shadeContext.transformationMatrices, instanceID, shadingPos);
        normal = transformNormalRigid(shadeContext.transformationMatrices, instanceID, normal);
    }

    backface = dot(normal, inDirection) > 0.0f;
    normal = backface ? -normal : normal;
}

__device__ __forceinline__ void getDataWithoutInDirectionAndEmission(
    const Triangle& tri,
    ShadeContext shadeContext,
    float2 barycentrics,
    float3 origin,

    int& materialID,
    float2& uv,
    float3& shadingPos,
    float3& normal,
    bool& backface,

    uint32_t instanceID = 0xFFFFFFFF
) {
    materialID = tri.materialID;
    float u = barycentrics.x;
    float v = barycentrics.y;

    uv = interpolateUV(tri, shadeContext, u, v);

    float3 apos = f3(__ldg(&shadeContext.vertices->positions[tri.aInd]));
    float3 bpos = f3(__ldg(&shadeContext.vertices->positions[tri.bInd]));
    float3 cpos = f3(__ldg(&shadeContext.vertices->positions[tri.cInd]));

    shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

    normal = interpolateNormal(tri, shadeContext, u, v, apos, bpos, cpos);

    // Transform to world space before the backface test: `origin` is already a world
    // position, so comparing it against an object-space normal gives the wrong sign
    // for any instance whose transform rotates.
    if (instanceID != 0xFFFFFFFF) {
        shadingPos = transformPosition(shadeContext.transformationMatrices, instanceID, shadingPos);
        normal = transformNormalRigid(shadeContext.transformationMatrices, instanceID, normal);
    }

    float3 inDirection = normalize(shadingPos - origin);
    backface = dot(normal, inDirection) > 0.0f;
    normal = backface ? -normal : normal;
}

inline void readObjSimple(
    std::string filename,
    std::vector<float4>& points,
    std::vector<float4>& normals,
    std::vector<float4>& colors,
    std::vector<float2>& uvs,
    std::vector<Triangle>& mesh,
    std::vector<Triangle>& lights,
    std::vector<LightDescriptor>& lightDescriptors,
    float3 c, float3 e,
    int materialID,
    float3 offset = f3(0.0f),
    uint32_t instanceID = 0
)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open OBJ file with path " << filename << std::endl;
        return;
    }
    int startIndex = points.size();
    int normalStartIndex = normals.size();
    int uvStartIndex = uvs.size();

    int nextLightIndex = lights.size();

    LightDescriptor ld;
    if (lengthSquared(e) > 0.0f) {
        ld.startInd = nextLightIndex;
        ld.totalPower = 0.0f;
        ld.instanceID = instanceID;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 's') continue; // skip comments

        std::istringstream iss(line);
        std::string prefix;

        iss >> prefix;


        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            float4 p = make_float4(x + offset.x, y + offset.y, z + offset.z, 0.0f);
            points.push_back(p);
        }
        else if (prefix == "vt")
        {
            double u, v;
            iss >> u >> v;

            float2 uv = f2(u,1.0f-v);
            uvs.push_back(uv);
        }
        else if (prefix == "vn") {
            double x, y, z;
            iss >> x >> y >> z;

            if (iss.fail() || std::isnan(x) || std::isnan(y) || std::isnan(z)) {
                normals.push_back(make_float4(0.0f, 1.0f, 0.0f, 0.0f)); // Safe dummy default
                continue;
            }
            float4 n = make_float4((float)x, (float)y, (float)z, 0.0f);

            float lenSq = lengthSquared(n);
            if (lenSq < 1e-12f) {
                n = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
            }
            normals.push_back(n);
        }
        else if (prefix == "f") {
            std::vector<std::string> items;

            std::string vertinfo;
            std::vector<int> vertexIndices;
            std::vector<int> normalIndices;
            std::vector<int> uvIndices;
            while (iss >> vertinfo)
            {
                std::istringstream vss(vertinfo);
                std::string idx;

                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        vertexIndices.push_back(stoi(idx) - 1);
                }
                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        uvIndices.push_back(stoi(idx) - 1);
                }
                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        normalIndices.push_back(stoi(idx) - 1);
                }
            }
            bool hasUV = uvIndices.size() == vertexIndices.size();
            bool hasN  = normalIndices.size() == vertexIndices.size();
            int n = vertexIndices.size();
            // Triangulate the polygon as a fan from the first vertex
            for (int i = 1; i < n - 1; ++i) {
                bool isLight = lengthSquared(e) > 0;

                int idx0 = vertexIndices[0] + startIndex;
                int idx1 = vertexIndices[i] + startIndex;
                int idx2 = vertexIndices[i + 1] + startIndex;

                float3 p0 = f3(points[idx0]);
                float3 p1 = f3(points[idx1]);
                float3 p2 = f3(points[idx2]);

                float3 e1 = p1 - p0;
                float3 e2 = p2 - p0;

                float3 cp = cross(e1, e2);
                float area = 0.5f * length(cp);

                if (area < 1e-18f) {
                    continue;
                }

                int uv_idx0 = hasUV ? uvIndices[0] + uvStartIndex : -1;
                int uv_idx1 = hasUV ? uvIndices[i] + uvStartIndex : -1;
                int uv_idx2 = hasUV ? uvIndices[i + 1] + uvStartIndex : -1;

                int n_idx0  = hasN ? normalIndices[0] + normalStartIndex : -1;
                int n_idx1  = hasN ? normalIndices[i] + normalStartIndex : -1;
                int n_idx2  = hasN ? normalIndices[i + 1] + normalStartIndex : -1;

                Triangle tri;
                if (isLight)
                    tri = Triangle(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, uv_idx0, uv_idx1, uv_idx2, f4(e), nextLightIndex, mesh.size());
                else
                    tri = Triangle(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, uv_idx0, uv_idx1, uv_idx2, f4(e), -51, mesh.size());
                mesh.push_back(tri);

                if (isLight) {
                    lights.push_back(tri);
                    ld.totalPower += luminance(e) * h_PI * area;
                    nextLightIndex++;
                }
            }
        }
    }

    if (lengthSquared(e) > 0.0f) {
        ld.numPrim = lights.size() - ld.startInd;
        lightDescriptors.push_back(ld);
    }

    file.close();
}

__device__ inline bool sample(
    const LightSampler& sampler,
    float rand_macro,
    float4 rand_micro,
    float3 probePos,
    const Vertices* verts,
    float3& output,
    float3& outDir,
    float3& lightNorm,
    float& t_max,
    float& pdf,

    const float4* matrices = nullptr
) {

    // 1. Categorical Selection
    if (rand_macro < sampler.envWeight) {
        // --- Sample Environment Map ---
        float microPDF;
        t_max = 1E30;

        sampler.envMap.sample(rand_micro, outDir, output, microPDF);
        pdf = microPDF * sampler.envWeight;
        return 1;

    } else {
        // --- Sample Mesh Light ---
        if (sampler.numLights == 0) {
            pdf = 0.0f;
            return 0; // Edge case: branched to mesh but none exist
        }

        // Remap rand_macro to [0, 1) to search the mesh-only CDF
        float mapped_rand = (rand_macro - sampler.envWeight) / (1.0f - sampler.envWeight);

        int index = sampler.binarySearchCDF(sampler.topLevelCDF, sampler.numLights, mapped_rand);
        LightDescriptor light = sampler.lights[index];

        // PDF of choosing this specific mesh light given we chose the mesh category

        int lightTriInd = light.startInd +
            sampler.binarySearchCDF(sampler.bottomLevelCDF + light.startInd, light.numPrim, rand_micro.x);

        float3 pos;
        float area;
        {
            Triangle l = sampler.triLights[lightTriInd];

            output = f3(l.emission);

            float3 apos = f3(__ldg(&verts->positions[l.aInd]));
            float3 bpos = f3(__ldg(&verts->positions[l.bInd]));
            float3 cpos = f3(__ldg(&verts->positions[l.cInd]));

            float u = sqrtf(rand_micro.y);
            float v = rand_micro.z;

            float3 localPos = (1.0f - u) * apos + u * (1.0f - v) * bpos + u * v * cpos;
            if (matrices)
                pos = transformPosition(matrices, light.instanceID, localPos);
            else
                pos = localPos;
            area = 0.5f * length(cross(bpos-apos, cpos-apos));

            // -1 sentinels for an emitter with no "vn" lines; fall back to the face normal.
            float3 localNorm;
            if (l.naInd < 0 || l.nbInd < 0 || l.ncInd < 0) {
                localNorm = cross(bpos - apos, cpos - apos);
            } else {
                float3 anorm = f3(__ldg(&verts->normals[l.naInd]));
                float3 bnorm = f3(__ldg(&verts->normals[l.nbInd]));
                float3 cnorm = f3(__ldg(&verts->normals[l.ncInd]));

                localNorm = (1.0f - u) * anorm + u * (1.0f - v) * bnorm + u * v * cnorm;
            }
            if (matrices)
                lightNorm = transformNormalRigid(matrices, light.instanceID, localNorm);
            else
                lightNorm = normalize(localNorm);
        }

        float pdf_chooseLight = (1.0f - sampler.envWeight) * (light.totalPower / sampler.totalMeshPower);

        float triPdf = (area * luminance(output) * PI) / light.totalPower;
        pdf = pdf_chooseLight * triPdf * (1.0f / area);

        outDir = normalize(pos - probePos);

        t_max = length(pos-probePos);
        return 0;
    }
}

/**
 * Specialized helper for when the ReSTIR algorithm evalutes DI contributions, returns the reconnection data
 */
__device__ inline bool sample_ReSTIR_rc_data(
    const LightSampler& sampler,
    float rand_macro,
    float4 rand_micro,
    float3 probePos,
    const Vertices* verts,
    float3& output,
    float3& outDir,
    float3& lightNorm,
    float& t_max,
    float& pdf,
    uint32_t& primID,
    float2& barycentrics,
    uint32_t& instanceID,

    const float4* matrices = nullptr
) {
    // 1. Categorical Selection
    if (rand_macro < sampler.envWeight) {
        // --- Sample Environment Map ---
        float microPDF;
        t_max = 1E30;

        sampler.envMap.sample(rand_micro, outDir, output, microPDF);
        pdf = microPDF * sampler.envWeight;
        primID = 0xFFFFFFFF;
        instanceID = 0xFFFFFFFF; // env has no instance; sentinel gates the transform
        return 1;

    } else {
        // --- Sample Mesh Light ---
        if (sampler.numLights == 0) {
            pdf = 0.0f;
            primID = 0xFFFFFFFF;
            instanceID = 0xFFFFFFFF;
            return 0; // Edge case: branched to mesh but none exist
        }

        // Remap rand_macro to [0, 1) to search the mesh-only CDF
        float mapped_rand = (rand_macro - sampler.envWeight) / (1.0f - sampler.envWeight);

        int index = sampler.binarySearchCDF(sampler.topLevelCDF, sampler.numLights, mapped_rand);
        LightDescriptor light = sampler.lights[index];
        instanceID = light.instanceID;

        // PDF of choosing this specific mesh light given we chose the mesh category

        int lightTriInd = light.startInd +
            sampler.binarySearchCDF(sampler.bottomLevelCDF + light.startInd, light.numPrim, rand_micro.x);

        float3 pos;
        float area;
        {
            const Triangle& l = sampler.triLights[lightTriInd];
            primID = l.triInd;
            output = f3(l.emission);

            float3 apos = f3(__ldg(&verts->positions[l.aInd]));
            float3 bpos = f3(__ldg(&verts->positions[l.bInd]));
            float3 cpos = f3(__ldg(&verts->positions[l.cInd]));

            float u = sqrtf(rand_micro.y);
            float v = rand_micro.z;

            float3 localPos = (1.0f - u) * apos + u * (1.0f - v) * bpos + u * v * cpos;
            if (matrices)
                pos = transformPosition(matrices, light.instanceID, localPos);
            else
                pos = localPos;
            area = 0.5f * length(cross(bpos-apos, cpos-apos));

            // -1 sentinels for an emitter with no "vn" lines; fall back to the face normal.
            float3 localNorm;
            if (l.naInd < 0 || l.nbInd < 0 || l.ncInd < 0) {
                localNorm = cross(bpos - apos, cpos - apos);
            } else {
                float3 anorm = f3(__ldg(&verts->normals[l.naInd]));
                float3 bnorm = f3(__ldg(&verts->normals[l.nbInd]));
                float3 cnorm = f3(__ldg(&verts->normals[l.ncInd]));

                localNorm = (1.0f - u) * anorm + u * (1.0f - v) * bnorm + u * v * cnorm;
            }
            if (matrices)
                lightNorm = transformNormalRigid(matrices, light.instanceID, localNorm);
            else
                lightNorm = normalize(localNorm);
            barycentrics = f2(u * (1.0f - v), u * v);
        }

        float pdf_chooseLight = (1.0f - sampler.envWeight) * (light.totalPower / sampler.totalMeshPower);

        float triPdf = (area * luminance(output) * PI) / light.totalPower;
        pdf = pdf_chooseLight * triPdf * (1.0f / area);

        outDir = normalize(pos - probePos);

        t_max = length(pos-probePos);

        if (pdf <= 0.0f) {
            printf("DEBUG: Light sampler returned zero PDF! PrimID: %u\n", primID);
        }
        return 0;
    }
}