#pragma once

#include "util.cuh"
#include "objects.cuh"
#include "sceneContexts.cuh"



__device__ inline float2 getBarycentrics(
    const ShadeContext sc,
    unsigned int triIndex, 
    const Ray& r
)
{
    if (triIndex >= sc.triNum) {
        return f2();
    }
    const Triangle& tri = sc.scene[triIndex];

    float4 tria = __ldg(&sc.vertices->positions[tri.aInd]);
    float4 trib = __ldg(&sc.vertices->positions[tri.bInd]);
    float4 tric = __ldg(&sc.vertices->positions[tri.cInd]);
    float4 e1 = trib - tria;
    float4 e2 = tric - tria;

    float4 h = cross3(r.direction, e2);
    float a = dot(h, e1);
    
    float f = 1.0/a;

    float4 s = r.origin-tria;
    float u = f * dot(s, h);
    float4 q = cross3(s, e1);
    float v = f * dot(r.direction, q);

    return f2(u, v);
}

__device__ __forceinline__ void getData(
    const Triangle& tri,
    ShadeContext shadeContext,
    float2 barycentrics,
    float4 inDirection,

    int& materialID,
    float2& uv,
    float4& shadingPos,
    float4& normal,
    bool& backface,
    float4& emission
) {
    materialID = tri.materialID;
    float u = barycentrics.x;
    float v = barycentrics.y;

    uv = __ldg(&shadeContext.vertices->uvs[tri.uvaInd]) * (1.0f - u - v) + 
        __ldg(&shadeContext.vertices->uvs[tri.uvbInd]) * u + 
        __ldg(&shadeContext.vertices->uvs[tri.uvcInd]) * v;

    float4 apos = __ldg(&shadeContext.vertices->positions[tri.aInd]);
    float4 bpos = __ldg(&shadeContext.vertices->positions[tri.bInd]);
    float4 cpos = __ldg(&shadeContext.vertices->positions[tri.cInd]);

    shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

    float4 a_n = __ldg(&shadeContext.vertices->normals[tri.naInd]);
    float4 b_n = __ldg(&shadeContext.vertices->normals[tri.nbInd]);
    float4 c_n = __ldg(&shadeContext.vertices->normals[tri.ncInd]);
    
    normal = (1.0f - u - v) * a_n + u * b_n + v * c_n;
    backface = dot(normal, inDirection) > 0.0f;
    normal = backface ? -normal : normal;
    emission = tri.emission;
}

__device__ __forceinline__ void getDataWithoutInDirection(
    const Triangle& tri,
    ShadeContext shadeContext,
    float2 barycentrics,
    float3 origin,

    int& materialID,
    float2& uv,
    float4& shadingPos,
    float4& normal,
    bool& backface,
    float4& emission
) {
    materialID = tri.materialID;
    float u = barycentrics.x;
    float v = barycentrics.y;

    uv = __ldg(&shadeContext.vertices->uvs[tri.uvaInd]) * (1.0f - u - v) + 
        __ldg(&shadeContext.vertices->uvs[tri.uvbInd]) * u + 
        __ldg(&shadeContext.vertices->uvs[tri.uvcInd]) * v;

    float3 apos = f3(__ldg(&shadeContext.vertices->positions[tri.aInd]));
    float3 bpos = f3(__ldg(&shadeContext.vertices->positions[tri.bInd]));
    float3 cpos = f3(__ldg(&shadeContext.vertices->positions[tri.cInd]));

    shadingPos = f4((1.0f - u - v) * apos + u * bpos + v * cpos);

    float3 a_n = f3(__ldg(&shadeContext.vertices->normals[tri.naInd]));
    float3 b_n = f3(__ldg(&shadeContext.vertices->normals[tri.nbInd]));
    float3 c_n = f3(__ldg(&shadeContext.vertices->normals[tri.ncInd]));

    float3 inDirection = normalize(f3(shadingPos) - origin);
    
    normal = f4((1.0f - u - v) * a_n + u * b_n + v * c_n);
    backface = dot(f3(normal), inDirection) > 0.0f;
    normal = backface ? -normal : normal;
    emission = tri.emission;
}

__device__ __forceinline__ void getDataSkipEmission(
    const Triangle& tri,
    ShadeContext shadeContext,
    float2 barycentrics,
    float4 inDirection,

    int& materialID,
    float2& uv,
    float4& shadingPos,
    float4& normal,
    bool& backface
) {
    materialID = tri.materialID;
    float u = barycentrics.x;
    float v = barycentrics.y;

    uv = __ldg(&shadeContext.vertices->uvs[tri.uvaInd]) * (1.0f - u - v) + 
        __ldg(&shadeContext.vertices->uvs[tri.uvbInd]) * u + 
        __ldg(&shadeContext.vertices->uvs[tri.uvcInd]) * v;

    float4 apos = __ldg(&shadeContext.vertices->positions[tri.aInd]);
    float4 bpos = __ldg(&shadeContext.vertices->positions[tri.bInd]);
    float4 cpos = __ldg(&shadeContext.vertices->positions[tri.cInd]);

    shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

    float4 a_n = __ldg(&shadeContext.vertices->normals[tri.naInd]);
    float4 b_n = __ldg(&shadeContext.vertices->normals[tri.nbInd]);
    float4 c_n = __ldg(&shadeContext.vertices->normals[tri.ncInd]);
    
    normal = (1.0f - u - v) * a_n + u * b_n + v * c_n;
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
    float4 c, float4 e, 
    int materialID, 
    float4 offset = f4(0.0f)
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
            float4 p = make_float4(x, y, z, 0.0f) + offset;
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

                float4 p0 = points[idx0];
                float4 p1 = points[idx1];
                float4 p2 = points[idx2];

                float4 e1 = f4(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
                float4 e2 = f4(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
                
                float4 cp = cross3(e1, e2);
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
                    tri = Triangle(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, uv_idx0, uv_idx1, uv_idx2, e, nextLightIndex, mesh.size());
                else
                    tri = Triangle(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, uv_idx0, uv_idx1, uv_idx2, e, -51, mesh.size());
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