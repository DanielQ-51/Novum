#pragma once

#include "util.cuh"
#include "objects.cuh"
#include <iostream>
#include <vector>
#include <stack>
#include <math.h> // For fmodf

// vose style aliasing for importance sampling the environment map
struct __align__(8) AliasEntry {
    float prob;
    int alias;
};

struct EnvMapData {
    float* pixels = nullptr;
    int width = 0;
    int height = 0;

    ~EnvMapData() {
        if (pixels) free(pixels);
    }
};

__host__ inline bool loadEnvMapToCPU(const char* filepath, EnvMapData& outData) {
    const char* err = nullptr;
    
    int ret = LoadEXR(&(outData.pixels), &(outData.width), &(outData.height), filepath, &err);

    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            std::cerr << "Failed to load EXR: " << err << std::endl;
            FreeEXRErrorMessage(err);
        }
        return false;
    }
    return true;
}

struct EnvMapView {
    cudaTextureObject_t texObj;
    AliasEntry* aliasTable;
    int width;
    int height;
    float totalPower; 
    float rotationOffset; // [0, 1] mapped to 0-360 degrees

    __device__ inline float4 sampleUV(float u, float v) const {
        // Inverse rotation for direct UV lookup
        u = fmodf(u - rotationOffset + 1.0f, 1.0f);
        return tex2D<float4>(texObj, u, v);
    }

    __device__ inline float4 sampleDir(float4 dir) const {
        float2 uv = dirToUV(dir);
        // Inverse rotation when looking up a scene direction (like a background hit)
        uv.x = fmodf(uv.x - rotationOffset + 1.0f, 1.0f);
        return tex2D<float4>(texObj, uv.x, uv.y);
    }

    __device__ inline bool sample(float4 rands, float4& outDir, float4& outEmission, float& outPdf) const {
        if (totalPower <= 0.0f || width == 0 || height == 0) return false;

        int numPixels = width * height;

        int bucketIdx = min((int)(rands.x * numPixels), numPixels - 1);
        AliasEntry bucket = aliasTable[bucketIdx];
        int pixelIdx = (rands.y < bucket.prob) ? bucketIdx : bucket.alias;

        int px = pixelIdx % width;
        int py = pixelIdx / width;

        float u = (px + rands.z) / (float)width;
        float v = (py + rands.w) / (float)height;

        float theta = v * PI;
        
        // Add the rotation offset to the generated physical direction
        float phi = (u + rotationOffset) * 2.0f * PI;

        float sinTheta = sinf(theta);
        outDir = f4(
            sinTheta * cosf(phi),
            cosf(theta),
            sinTheta * sinf(phi)
        );

        // Fetch emission from the UNROTATED coordinate so it correctly matches the alias table PDF
        outEmission = tex2D<float4>(texObj, u, v);

        float lum = luminance(outEmission);

        const float two_pi_sq = 2.0f * PI * PI;
        outPdf = (lum * numPixels) / (two_pi_sq * totalPower);

        return true;
    }
};

class EnvironmentMapManager {
private:
    cudaTextureObject_t texObj = 0;
    cudaArray_t cuArray = nullptr;
    AliasEntry* d_aliasTable = nullptr; 
    
    int width = 0;
    int height = 0;
    float totalPower = 0.0f;
    float rotationOffset = 0.0f; // Stores rotation in [0, 1] range

    std::vector<AliasEntry> buildAliasTable(const std::vector<float>& weights) {
        int N = weights.size();
        std::vector<AliasEntry> table(N);
        
        std::vector<float> scaled_probs(N);
        std::stack<int> rich_stack;
        std::stack<int> poor_stack;

        for (int i = 0; i < N; ++i) {
            scaled_probs[i] = weights[i] * N;

            if (scaled_probs[i] < 1.0f) {
                poor_stack.push(i);
            } else {
                rich_stack.push(i);
            }
        }

        while (!poor_stack.empty() && !rich_stack.empty()) {
            int poor_idx = poor_stack.top(); poor_stack.pop();
            int rich_idx = rich_stack.top(); rich_stack.pop();

            table[poor_idx].prob = scaled_probs[poor_idx];
            table[poor_idx].alias = rich_idx;

            scaled_probs[rich_idx] -= (1.0f - scaled_probs[poor_idx]);

            if (scaled_probs[rich_idx] < 1.0f) {
                poor_stack.push(rich_idx);
            } else {
                rich_stack.push(rich_idx);
            }
        }

        while (!rich_stack.empty()) {
            int idx = rich_stack.top(); rich_stack.pop();
            table[idx].prob = 1.0f;
            table[idx].alias = idx;
        }
        while (!poor_stack.empty()) {
            int idx = poor_stack.top(); poor_stack.pop();
            table[idx].prob = 1.0f;
            table[idx].alias = idx;
        }

        return table;
    }

public:
    EnvironmentMapManager(const std::string& filepath) {
        EnvMapData envMap;
        if (!loadEnvMapToCPU(filepath.c_str(), envMap)) {
            std::cerr << "Failed to load EXR: " << filepath << std::endl;
            return;
        }

        width = envMap.width;
        height = envMap.height;
        int numPixels = width * height;

        std::vector<float> pixelEnergies(numPixels, 0.0f);
        totalPower = 0.0f;

        for (int y = 0; y < height; ++y) {
            float v = (y + 0.5f) / static_cast<float>(height);
            float theta = v * h_PI;
            float sinTheta = std::sin(theta);

            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                
                float r = envMap.pixels[idx * 4 + 0];
                float g = envMap.pixels[idx * 4 + 1];
                float b = envMap.pixels[idx * 4 + 2];

                float lum = (r * 0.2126f) + (g * 0.7152f) + (b * 0.0722f);
                
                float energy = std::max(0.0f, lum * sinTheta);
                
                pixelEnergies[idx] = energy;
                totalPower += energy;
            }
        }

        if (totalPower == 0.0f) {
            for (int i = 0; i < numPixels; ++i) {
                pixelEnergies[i] = 1.0f / numPixels; 
            }
            totalPower = 1.0f;
        } else {
            for (int i = 0; i < numPixels; ++i) {
                pixelEnergies[i] /= totalPower; 
            }
        }

        std::vector<AliasEntry> hostAliasTable = buildAliasTable(pixelEnergies);
        
        size_t tableSizeBytes = numPixels * sizeof(AliasEntry);
        cudaMalloc(&d_aliasTable, tableSizeBytes);
        cudaMemcpy(d_aliasTable, hostAliasTable.data(), tableSizeBytes, cudaMemcpyHostToDevice);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&cuArray, &channelDesc, width, height);

        size_t pitch = width * 4 * sizeof(float);
        cudaMemcpy2DToArray(cuArray, 0, 0, envMap.pixels, pitch, pitch, height, cudaMemcpyHostToDevice);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc = {};
        texDesc.normalizedCoords = 1;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.addressMode[0] = cudaAddressModeWrap;  // Wrap Longitude
        texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp Latitude

        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

        std::cout << "Environment map <" << filepath << "> successfully read and saved.\n\n";
    }

    ~EnvironmentMapManager() {
        if (texObj != 0) cudaDestroyTextureObject(texObj);
        if (cuArray != nullptr) cudaFreeArray(cuArray);
        if (d_aliasTable != nullptr) cudaFree(d_aliasTable);
    }

    EnvironmentMapManager(const EnvironmentMapManager&) = delete;
    EnvironmentMapManager& operator=(const EnvironmentMapManager&) = delete;

    // Call this at runtime to rotate the environment map around the Y-axis
    void setRotation(float degrees) {
        rotationOffset = fmodf(degrees / 360.0f, 1.0f);
        if (rotationOffset < 0.0f) {
            rotationOffset += 1.0f;
        }
    }

    EnvMapView getView() const {
        return EnvMapView{ texObj, d_aliasTable, width, height, totalPower, rotationOffset };
    }
};