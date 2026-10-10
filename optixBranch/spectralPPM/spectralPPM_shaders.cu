#define RNG_STATE_64 1
#include <optix.h>
#include <optix_device.h>
#include "settings.cuh"
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"
#include "helpers.cuh"

#include "spectralPPM/spectralPPM_Utils.cuh"
#include "spectralReflectors.cuh"

extern "C" {
    __constant__ PipelineParams allParams;
}

#ifndef SPECTRAL_PPM_MERGE_ROUGHNESS_BOUND
#define SPECTRAL_PPM_MERGE_ROUGHNESS_BOUND 0.15f
#endif

extern "C" __global__ void __raygen__spectralPPM_traceLight() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const SpectralParams& spectral = allParams.spectralPPM; // gets compiled out, so not taking up registers

    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;

    RNGState localState = load_rng(pixelIdx, params.frame_index, 1, nullptr);

    Ray r;
    float3 power;
    float emit_pdf;
    float emit_cos;
    bool sampledEnvironment = sampleEmission(
        params.shadeContext.lightSampler,
        spectral.sceneCenter,
        spectral.sceneRadius,
        rand(&localState),
        rand4(&localState),
        rand2(&localState),
        params.shadeContext.vertices,
        r.origin, r.direction, power, emit_pdf, emit_cos,

        params.shadeContext.transformationMatrices
    );

    float m = fmaxf(power.x, fmaxf(power.y, power.z));
    if (m <= 0.0f || emit_pdf <= 0.0f) return; 

    float scale = 2.0f * m;
    float3 c = rgbToSigmoidCoeffs(power * (1.0f / scale),
                              spectral.sRGBToSpectrumTable_Scale, spectral.sRGBToSpectrumTable_Data);

    float fluxScale = emit_cos / emit_pdf;
    SampledSpectrum throughput;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        throughput.v[i] = fluxScale * // the cos over pdf like normal
            scale * sigmoidPolynomial(c, spectral.wl.lambda[i]) * spectral.d65[i]; // this is the radiance value
    }
    
    // whether its current on a hero only path
    bool heroOnly = false;

    for (int depth = 0; depth < params.max_depth; depth++) {
        if (depth) {
            // this is profiled extensively and guaranteed to work better 
            // for high-live-state things like restir pt, but not tested 
            // for non-screenspace coherent stuff like this. likely wont be as good
            optixMakeNopHitObject();
            optixReorder(); 
        }
        
        SurfaceHit hitData = traceClosestNoSER(params, r);

        if (!hitData.isHit) {
            return;
        }

        int materialID;
        float2 uv;
        float3 shadingPos;
        bool backface;
        float3 geoNormal;
        float3 normal;
        float3 ImplicitEmission;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];

        // no lods for photon map generation
        getDataGeo(
            tri,
            params.shadeContext,
            hitData.barycentrics,
            r.direction,

            materialID,
            uv,
            shadingPos,
            geoNormal,
            normal,
            backface,
            ImplicitEmission,

            hitData.instanceId
        );

        bool curr_delta = params.shadeContext.materials[materialID].isSpecular;

        if (!curr_delta && depth > 0) { // depth 0 (direct light) is handled by NEE in the eye pass
            // inside the branch where this thread stores a photon (only storing lanes participate)
            uint32_t mask = __activemask();
            uint32_t lane;
            #if defined(__CUDA_ARCH__)
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
            #endif
            uint32_t leader = __ffs(mask) - 1;

            uint32_t base;
            if (lane == leader)
                base = atomicAdd(spectral.photonCounter, __popc(mask));
            base = __shfl_sync(mask, base, leader);
            uint32_t slot = base + __popc(mask & ((1u << lane) - 1));

            if (slot < spectral.maxPhotons) {
                setPhoton(
                    spectral.photons, 
                    slot, 
                    shadingPos, 
                    geoNormal, // used for rejection; note that the shading normal at evaluation is already known (from the eye path)
                    backface, 
                    heroOnly, 
                    throughput, 
                    -r.direction
                );
            } else {
                return;
            }
        }

        

        SpectralShading::MaterialInfo matInfo = SpectralShading::buildMaterialInfo(
            params.shadeContext.materials,
            materialID,
            params.shadeContext.textures,
            uv,
            backface,
            spectral.wl,
            spectral.sRGBToSpectrumTable_Scale,
            spectral.sRGBToSpectrumTable_Data
        );
        float3 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        float3 outDirLocal;
        SampledSpectrum f_val;
        float bsdf_pdf;
        
        bool sampledHeroOnly;
        SpectralShading::sample_f_eval(
            localState,
            matInfo,
            incomingDirLocal,
            TRANSPORTMODE_IMPORTANCE,
            outDirLocal,
            f_val,
            bsdf_pdf,
            sampledHeroOnly
        );

        heroOnly = sampledHeroOnly || heroOnly;

        if (bsdf_pdf < EPSILON) return; 

        float rr_before = maxComponent(throughput);
        if (rr_before <= 0.0f) return;

        throughput *= f_val * fabsf(outDirLocal.z) / bsdf_pdf;

        float rr_after = maxComponent(throughput);

        float q = fminf(rr_after / rr_before, 1.0f);
        if (q <= 0.0f || rand(&localState) >= q) return;
        throughput = throughput * (1.0f / q);

        float3 outDirWorld = toWorld(outDirLocal, normal);

        bool refracted = dot(outDirWorld, r.direction) > 0.f;
        
        r.direction = outDirWorld;
        r.origin = shadingPos + (dot(outDirWorld, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
    }
}

extern "C" __global__ void __raygen__spectralPPM_traceEye() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const SpectralParams& spectral = allParams.spectralPPM; // gets compiled out, so not taking up registers

    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;

    RNGState localState = load_rng(pixelIdx, params.frame_index, 0, nullptr);

    Ray r = params.camera.generateCameraRay(localState, x, y);

    SampledSpectrum throughput;
    SampledSpectrum accum;
    for (int i = 0; i < N; ++i) {
        throughput.v[i] = 1.0f;
        accum.v[i] = 0.0f;
    }

    bool heroOnly = false;
    for (int depth = 0; depth < params.max_depth; depth++) {
        if (depth) {
            optixMakeNopHitObject();
            optixReorder(); 
        }
        
        SurfaceHit hitData = traceClosestNoSER(params, r);

        // Handle env. These cant be reproduced by merging so no mis needed
        if (!hitData.isHit) {
            float3 contribution = params.shadeContext.lightSampler.envMap.sampleDir(r.direction);
            SampledSpectrum emissive;

            float m = fmaxf(contribution.x, fmaxf(contribution.y, contribution.z));

            if (m > 0.0f) {
                float scale = 2.0f * m;
                float3 c = rgbToSigmoidCoeffs(contribution * (1.0f / scale),
                                        spectral.sRGBToSpectrumTable_Scale, spectral.sRGBToSpectrumTable_Data);

                #pragma unroll
                for (int i = 0; i < N; ++i) {
                    emissive.v[i] = scale * sigmoidPolynomial(c, spectral.wl.lambda[i]) * spectral.d65[i]; // this is the radiance value
                }

                accum += heroOnly ? (throughput * emissive) * (float)N : throughput * emissive;
            }

            
            break;
        }

        int materialID;
        float2 uv;
        float3 shadingPos;
        bool backface;
        float3 geoNormal;
        float3 normal;
        float3 ImplicitEmission;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];

        // no lods for photon map generation
        getDataGeo(
            tri,
            params.shadeContext,
            hitData.barycentrics,
            r.direction,

            materialID,
            uv,
            shadingPos,
            geoNormal,
            normal,
            backface,
            ImplicitEmission,

            hitData.instanceId
        );

        // HANDLE DIRECTLY SEEING LIGHT (partition domain st no photon on light itself)
        if (!backface && lengthSquared(ImplicitEmission) > 0.0f) {
            SampledSpectrum emissive;

            float m = fmaxf(ImplicitEmission.x, fmaxf(ImplicitEmission.y, ImplicitEmission.z));

            float scale = 2.0f * m;
            float3 c = rgbToSigmoidCoeffs(ImplicitEmission * (1.0f / scale),
                                    spectral.sRGBToSpectrumTable_Scale, spectral.sRGBToSpectrumTable_Data);

            #pragma unroll
            for (int i = 0; i < N; ++i) {
                emissive.v[i] = scale * sigmoidPolynomial(c, spectral.wl.lambda[i]) * spectral.d65[i]; // this is the radiance value
            }

            accum += heroOnly ? (throughput * emissive) * (float)N : throughput * emissive;
        }

        SpectralShading::MaterialInfo matInfo = SpectralShading::buildMaterialInfo(
            params.shadeContext.materials,
            materialID,
            params.shadeContext.textures,
            uv,
            backface,
            spectral.wl,
            spectral.sRGBToSpectrumTable_Scale,
            spectral.sRGBToSpectrumTable_Data
        );

        float3 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        if (matInfo.type == MAT_DIFFUSE || sqrtf(matInfo.alpha) > SPECTRAL_PPM_MERGE_ROUGHNESS_BOUND) {
            // direct light via NEE
            {
                float3 lightNormal, lightEmission, toLight;
                float t_max, lightPdf;

                bool sampledEnv = sample(
                    params.shadeContext.lightSampler,
                    rand(&localState), rand4(&localState),
                    shadingPos,
                    params.shadeContext.vertices,
                    lightEmission, toLight, lightNormal, t_max, lightPdf,
                    params.shadeContext.transformationMatrices
                );

                float m = fmaxf(lightEmission.x, fmaxf(lightEmission.y, lightEmission.z));
                float cosSurf = dot(normal, toLight);
                bool lightBackface = !sampledEnv && dot(lightNormal, -toLight) < 0.0f;

                if (m > 0.0f && lightPdf > 0.0f && cosSurf > 0.0f && !lightBackface &&
                    !traceVisibility(params, Ray(shadingPos + (dot(toLight, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON, toLight),
                                     t_max * (1.0f - EPSILON3)))
                {
                    float geom = sampledEnv ? cosSurf / lightPdf
                                            : cosSurf * dot(lightNormal, -toLight) / (lightPdf * t_max * t_max);

                    float scale = 2.0f * m;
                    float3 c = rgbToSigmoidCoeffs(lightEmission * (1.0f / scale),
                                                  spectral.sRGBToSpectrumTable_Scale, spectral.sRGBToSpectrumTable_Data);

                    SampledSpectrum f_val;
                    SpectralShading::f_eval(matInfo, incomingDirLocal, toLocal(toLight, normal), f_val);

                    SampledSpectrum contribution;
                    for (int i = 0; i < N; ++i)
                        contribution.v[i] = throughput.v[i] * f_val.v[i] * geom *
                            scale * sigmoidPolynomial(c, spectral.wl.lambda[i]) * spectral.d65[i];

                    accum += heroOnly ? contribution * (float)N : contribution;
                }
            }

            const float& mergeRadius = spectral.mergeRadius;
            int3 centerIndex = photonGridCell(shadingPos, mergeRadius);
            float radiusSq = mergeRadius * mergeRadius;

            for (int z1 = -1; z1 <= 1; ++z1)
            {
                for (int y1 = -1; y1 <= 1; ++y1)
                {
                    for (int x1 = -1; x1 <= 1; ++x1)
                    {
                        int3 neighborIndex = make_int3(
                            centerIndex.x + x1,
                            centerIndex.y + y1,
                            centerIndex.z + z1
                        );

                        uint32_t hash = photonGridHash(neighborIndex, spectral.hashTableSize);
                        uint32_t start, end;
                        getCellRange_ldg(spectral.cellStart, spectral.cellEnd, hash, start, end);

                        if (start == 0xFFFFFFFF) continue;

                        for (int i = start; i < end; ++i) {
                            float3 photonPos = getPos_ldg(spectral.photonsSorted, i);

                            float3 photonNorm; // geometric normal
                            bool photonBackface, photonHeroOnly;
                            getNormalInfo_ldg(spectral.photonsSorted, i, photonNorm, photonBackface, photonHeroOnly);

                            float distSq = lengthSquared(shadingPos - photonPos);
                            if (backface != photonBackface) continue; // cannot merge

                            if (distSq <= radiusSq && dot(photonNorm, geoNormal) > 0.9f) {
                                float3 toPhotonPrev = getWi_ldg(spectral.photonsSorted, i);
                                //float3 toCurrFromEyePrev = r.direction;

                                //float3 inLocal = toLocal(toCurrFromEyePrev, normal);
                                float3 outLocal = toLocal(toPhotonPrev, normal);

                                SampledSpectrum f_val;
                                SpectralShading::f_eval(
                                    matInfo,
                                    incomingDirLocal,
                                    outLocal,
                                    f_val
                                );

                                SampledSpectrum contribution = 
                                    getRadiance_ldg(spectral.photonsSorted, i) * f_val * throughput
                                    * spectral.invMergeNorm;

                                accum += (heroOnly || photonHeroOnly) ? (contribution * (float)N) : (contribution);
                            }
                        }
                    }
                }
            }

            break;
        }

        float3 outDirLocal;
        SampledSpectrum f_val;
        float bsdf_pdf;
        
        bool sampledHeroOnly;
        SpectralShading::sample_f_eval(
            localState,
            matInfo,
            incomingDirLocal,
            TRANSPORTMODE_RADIANCE,
            outDirLocal,
            f_val,
            bsdf_pdf,
            sampledHeroOnly
        );

        heroOnly = sampledHeroOnly || heroOnly;

        if (bsdf_pdf < EPSILON) break; 

        throughput *= f_val * fabsf(outDirLocal.z) / bsdf_pdf;

        float3 outDirWorld = toWorld(outDirLocal, normal);

        bool refracted = dot(outDirWorld, r.direction) > 0.f;
        
        r.direction = outDirWorld;
        r.origin = shadingPos + (dot(outDirWorld, geoNormal) > 0.0f ? geoNormal : -geoNormal) * RAY_EPSILON;
    }

    // add to accum buffer
    float R = 0.0f, G = 0.0f, B = 0.0f;
    for (int i = 0; i < N; ++i) {
        R += accum.v[i] * spectral.wR[i];
        G += accum.v[i] * spectral.wG[i];
        B += accum.v[i] * spectral.wB[i];
    }
    params.accum_buffer[pixelIdx] += f4(fireflyClamp(make_float3(R, G, B)));
}
