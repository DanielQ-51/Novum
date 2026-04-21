/**
 * A header containing the logic for performing importance sampling for light source sampling.
 * * Updated to support a dynamic categorical split between Environment and Mesh lights.
 */

#pragma once

#include "util.cuh"
#include "environment.cuh"

enum LightType {
    LIGHT_MESH = 0,
    LIGHT_ENV = 1
};

struct __align__(16) LightDescriptor {
    int type; // 0 for mesh
    int startInd; // start index in the respective array
    int numPrim; // number of primitives in this light
    float totalPower; // total power of this light
};

struct LightSampler {
    LightDescriptor* lights;
    float* topLevelCDF;
    int numLights; // Note: This now represents the number of MESH lights only.

    Triangle* triLights;
    float* bottomLevelCDF;

    EnvMapView envMap;

    float totalMeshPower;
    float envWeight; // The field variable dictating the split (e.g., 0.5f for 50/50)

    __device__ inline int binarySearchCDF(const float* cdf, int count, float randNum) const {
        int left = 0;
        int right = count - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (cdf[mid] <= randNum) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    // returns area pdf
    __device__ inline float evaluateMeshPdf(const Triangle& tri) const {
        if (envWeight == 1.0f || totalMeshPower == 0.0f) return 0.0f;
        // Adjusted to account for the categorical split
        return (1.0f - envWeight) * (luminance(tri.emission) * PI) / totalMeshPower;
    }

    // returns solid angle pdf
    __device__ inline float evaluateEnvPdf(float4 dir) const {
        if (envWeight == 0.0f || envMap.totalPower == 0.0f) return 0.0f;
        float4 emission = envMap.sampleDir(dir);
        float lum = luminance(emission);

        int numPixels = envMap.width * envMap.height;
        const float two_pi_sq = 2.0f * PI * PI;

        // Adjusted to use envMap.totalPower and scale by the envWeight
        return envWeight * (lum * numPixels) / (two_pi_sq * envMap.totalPower);
    }

    __device__ inline bool sample(
        float rand_macro,
        float4 rand_micro,
        float4 probePos,
        const Vertices* verts,
        float4& output,
        float4& outDir,
        float4& lightNorm,
        float& t_max,
        float& pdf
    ) const {
        
        // 1. Categorical Selection
        if (rand_macro < envWeight) {
            // --- Sample Environment Map ---
            float microPDF;
            t_max = 1E30;

            envMap.sample(rand_micro, outDir, output, microPDF);
            pdf = microPDF * envWeight;
            return 1;

        } else {
            // --- Sample Mesh Light ---
            if (numLights == 0) {
                pdf = 0.0f;
                return 0; // Edge case: branched to mesh but none exist
            }

            // Remap rand_macro to [0, 1) to search the mesh-only CDF
            float mapped_rand = (rand_macro - envWeight) / (1.0f - envWeight);

            int index = binarySearchCDF(topLevelCDF, numLights, mapped_rand);
            LightDescriptor light = lights[index];

            // PDF of choosing this specific mesh light given we chose the mesh category

            int lightTriInd = light.startInd + 
                binarySearchCDF(bottomLevelCDF + light.startInd, light.numPrim, rand_micro.x);

            float4 pos;
            float area;
            {
                Triangle l = triLights[lightTriInd]; 

                output = l.emission;
                        
                float4 apos = __ldg(&verts->positions[l.aInd]);
                float4 bpos = __ldg(&verts->positions[l.bInd]);
                float4 cpos = __ldg(&verts->positions[l.cInd]);

                float u = sqrtf(rand_micro.y);
                float v = rand_micro.z;

                pos = (1.0f - u) * apos + u * (1.0f - v) * bpos + u * v * cpos;
                area = 0.5f * length(cross3(bpos-apos, cpos-apos));

                float4 anorm = __ldg(&verts->normals[l.naInd]);
                float4 bnorm = __ldg(&verts->normals[l.nbInd]);
                float4 cnorm = __ldg(&verts->normals[l.ncInd]);

                lightNorm = (1.0f - u) * anorm + u * (1.0f - v) * bnorm + u * v * cnorm;
            }

            float pdf_chooseLight = (1.0f - envWeight) * (light.totalPower / totalMeshPower);
            
            float triPdf = (area * luminance(output) * PI) / light.totalPower;
            pdf = pdf_chooseLight * triPdf * (1.0f / area);

            outDir = normalize(pos - probePos);
            
            t_max = length(pos-probePos);
            return 0;
        }
    }

    __host__ void printDebugState(int maxPrimsToPrint = 16) const {
        std::cout << "\n========== LIGHT SAMPLER STATE ==========\n";
        std::cout << "Environment Weight:    " << envWeight << " (" << (envWeight * 100.0f) << "% rays to Env)\n";
        std::cout << "Total Mesh Power:      " << totalMeshPower << "\n";
        std::cout << "Environment Map Power: " << envMap.totalPower << "\n";
        std::cout << "Number of Mesh Lights: " << numLights << "\n";
        std::cout << "-----------------------------------------\n";

        if (numLights == 0) {
            std::cout << "No mesh lights present in scene.\n";
            std::cout << "=========================================\n\n";
            return;
        }

        // 1. Allocate temporary host memory for Top Level
        std::vector<LightDescriptor> h_lights(numLights);
        std::vector<float> h_TLCDF(numLights);

        // Copy Top Level data from Device to Host
        cudaError_t err1 = cudaMemcpy(h_lights.data(), lights, numLights * sizeof(LightDescriptor), cudaMemcpyDeviceToHost);
        cudaError_t err2 = cudaMemcpy(h_TLCDF.data(), topLevelCDF, numLights * sizeof(float), cudaMemcpyDeviceToHost);

        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            std::cerr << "[Error] Failed to copy Top Level data from device.\n";
            return;
        }

        // 2. Determine total size of Bottom Level CDF
        int totalPrims = h_lights[numLights - 1].startInd + h_lights[numLights - 1].numPrim;

        // Allocate and copy Bottom Level CDF
        std::vector<float> h_BLCDF(totalPrims);
        if (totalPrims > 0) {
            cudaError_t err3 = cudaMemcpy(h_BLCDF.data(), bottomLevelCDF, totalPrims * sizeof(float), cudaMemcpyDeviceToHost);
            if (err3 != cudaSuccess) {
                std::cerr << "[Error] Failed to copy Bottom Level data from device.\n";
                return;
            }
        }

        // 3. Print the nested State
        std::cout << "Mesh Lights Distribution:\n";
        std::cout << "=========================================\n";
        
        for (int i = 0; i < numLights; i++) {
            std::cout << "Light [" << i << "] \n";
            std::cout << "  Start Ind : " << h_lights[i].startInd << "\n";
            std::cout << "  Num Prims : " << h_lights[i].numPrim << "\n";
            std::cout << "  Power     : " << h_lights[i].totalPower << "\n";
            std::cout << "  TLCDF Val : " << h_TLCDF[i] << "\n";
            std::cout << "  --- Bottom Level CDF (Primitives) ---\n";
            
            int start = h_lights[i].startInd;
            int count = h_lights[i].numPrim;
            
            for (int p = 0; p < count; p++) {
                // Formatting safeguard to prevent terminal flooding
                if (maxPrimsToPrint >= 0 && p >= maxPrimsToPrint && p != count - 1) {
                    if (p == maxPrimsToPrint) {
                        std::cout << "    ... (" << count - maxPrimsToPrint - 1 << " primitives omitted) ...\n";
                    }
                    continue; // Skip the print, but still ensure the last index prints
                }
                
                std::cout << "    Prim [" << start + p << "]: " << h_BLCDF[start + p] << "\n";
            }
            std::cout << "-----------------------------------------\n";
        }
        std::cout << "=========================================\n\n";
    }
};

struct LightSamplerManager {
    LightDescriptor* lights = nullptr;
    float* topLevelCDF = nullptr;
    int numLights;

    Triangle* triLights = nullptr;
    float* bottomLevelCDF = nullptr;

    float totalMeshPower;
    float envWeight;

    // envMapView's memory is managed externally. No need to manage here
    EnvMapView envMap;

    __host__ LightSamplerManager(
        std::vector<LightDescriptor> ld, // Passed by value, represents only mesh lights now
        const std::vector<Triangle>& host_lights,
        const std::vector<float4>& points,
        Triangle*& d_triLights,
        EnvMapView env,
        float desiredEnvWeight = 0.5f // Exposed variable for the split
    ) {
        envMap = env;
        numLights = ld.size();

        // 1. Calculate power strictly for mesh lights
        float meshPower = 0.0f;
        std::vector<float> TLCDF;
        TLCDF.reserve(ld.size());

        for (const LightDescriptor& curr : ld) {
            meshPower += curr.totalPower;
            TLCDF.push_back(meshPower);
        }

        totalMeshPower = meshPower;

        // 2. Normalize the top-level CDF using mesh power, not scene power
        if (meshPower > 0.0f) {
            for (float& curr : TLCDF) {
                curr /= meshPower;
            }
        }

        // 3. Robustly determine the envWeight based on scene contents
        if (env.totalPower <= 0.0f) {
            envWeight = 0.0f; // No env map, dedicate 100% rays to mesh lights
        } else if (meshPower <= 0.0f) {
            envWeight = 1.0f; // No mesh lights, dedicate 100% rays to env map
        } else {
            envWeight = desiredEnvWeight; // Use the requested split
        }

        // 4. Build bottom level CDF (Unchanged)
        std::vector<float> BLCDF(host_lights.size());

        for (const LightDescriptor& curr : ld) {
            float currPower = 0.0f;

            for (int i = curr.startInd; i < curr.startInd + curr.numPrim; i++) {
                float4 apos = points[host_lights[i].aInd];
                float4 bpos = points[host_lights[i].bInd];
                float4 cpos = points[host_lights[i].cInd];

                float area = 0.5f * length(cross3(bpos - apos, cpos - apos));
                currPower += area * luminance(host_lights[i].emission) * h_PI;
                BLCDF[i] = currPower;
            }

            if (currPower > 0.0f) {
                for (int i = curr.startInd; i < curr.startInd + curr.numPrim; i++) {
                    BLCDF[i] /= currPower;
                }
            }
        }

        // 5. Allocate and Copy to Device
        if (ld.size() > 0) {
            size_t ldSize = ld.size() * sizeof(LightDescriptor);
            cudaMalloc(&lights, ldSize);
            cudaMemcpy(lights, ld.data(), ldSize, cudaMemcpyHostToDevice);

            size_t tlSize = TLCDF.size() * sizeof(float);
            cudaMalloc(&topLevelCDF, tlSize);
            cudaMemcpy(topLevelCDF, TLCDF.data(), tlSize, cudaMemcpyHostToDevice);
        }

        if (host_lights.size() > 0) {
            size_t triSize = host_lights.size() * sizeof(Triangle);
            cudaMalloc(&triLights, triSize);
            cudaMemcpy(triLights, host_lights.data(), triSize, cudaMemcpyHostToDevice);
            
            d_triLights = triLights; 
        }

        if (BLCDF.size() > 0) {
            size_t blSize = BLCDF.size() * sizeof(float);
            cudaMalloc(&bottomLevelCDF, blSize);
            cudaMemcpy(bottomLevelCDF, BLCDF.data(), blSize, cudaMemcpyHostToDevice);
        }
    }

    __host__ ~LightSamplerManager() {
        if (lights != nullptr) cudaFree(lights);
        if (topLevelCDF != nullptr) cudaFree(topLevelCDF);
        if (triLights != nullptr) cudaFree(triLights);
        if (bottomLevelCDF != nullptr) cudaFree(bottomLevelCDF);
    }
    
    LightSamplerManager(const LightSamplerManager&) = delete;
    LightSamplerManager& operator=(const LightSamplerManager&) = delete;

    LightSampler getSampler() const {
        return LightSampler{ lights, topLevelCDF, numLights, triLights, bottomLevelCDF, envMap, totalMeshPower, envWeight };
    }
};