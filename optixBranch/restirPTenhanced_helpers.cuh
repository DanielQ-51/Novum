#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "sceneContexts.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "optixStructs.cuh"

#ifndef RECON_FOOTPRINT_C_CONSTANT
#define RECON_FOOTPRINT_C_CONSTANT 0.02f
#endif

#ifndef DEBUG_TEST_PIXEL_X
#define DEBUG_TEST_PIXEL_X 326
#endif

#ifndef DEBUG_TEST_PIXEL_Y
#define DEBUG_TEST_PIXEL_Y (800 - 247)
#endif

template<bool isReverseShift>
__device__ __forceinline__ ShiftResult evaluateHybridShift(
    const PipelineParams& allParams,
    uint32_t x, uint32_t y,           // The pixel generating the offset ray
    uint32_t seed,                        // The RNG seed to replay
    uint32_t pathLength,                  // Base path data
    uint32_t rcVertexIndex,               // Base path data
    TechniqueType type,                       // Base path data
    uint32_t rcPrimID,                    // Base path data
    float2 rcBarycentrics,                    // Base path data
    float3 rcWi,                              // Base path data
    float3 rcRadiance,                        // Base path data
    float cached_nee,                         // Base path data
    float jacobianDenom                       // Base path data
) {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers

    uint32_t reorderHint = (rcVertexIndex == FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY) ? 0u : 0xFFFFFFFF;
    optixReorder(reorderHint, 1);

    RNGState localState = load_rng(seed); // seed path using other pixel's start seed
    Ray r = params.camera.generateCameraRay(localState, x, y);
    if constexpr (!isReverseShift) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("frame %u using rng with state: %u for temporal reuse, replaying from %u, %u, with initial camera ray o(%f, %f, %f), d(%f, %f, %f)\n", params.frame_index, seed, x, y, r.origin.x, r.origin.y, r.origin.z, r.direction.x, r.direction.y, r.direction.z);
        }
    }
    
    // handles all k=d bsdf cases except for environment hit with a non specular previous vertex
    // Note: a "full replay" is also done in NEE k=d paths, but its still handled in the other block since it shares the rc shadow ray logic
    if (rcVertexIndex == FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY) {
        float3 throughput = f3(1.0f);

        // these three may be unnecesary
        float lastPDF;
        bool prevDelta;
        float lastCosine;

        float3 lastPos;
        float3 lastNormal;
        int lastMaterialID;
        float2 lastUV;
        bool lastBackface;
        float3 lastInDirLocal;

        float primaryFootprint;
    {
        SurfaceHit hitData = traceClosest(params, r);

        if (!hitData.isHit) {
            return {false, f3(0), 0.0f, 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
        }

        int materialID;
        float2 uv;
        float4 shadingPos;
        bool backface;
        float4 normal;
        float4 ImplicitEmission;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];

        getData(
            tri,
            params.shadeContext,
            hitData.barycentrics, 
            r.direction,

            materialID,
            uv,
            shadingPos,
            normal,
            backface,
            ImplicitEmission
        );
        if constexpr (!isReverseShift) {
            if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
                printf("replaying shading pos depth 0: %f, %f, %f\n", shadingPos.x, shadingPos.y, shadingPos.z);
            }
        }

        primaryFootprint = 
            (RECON_FOOTPRINT_C_CONSTANT / 100.0f) * 
            (hitData.t * hitData.t * 4.0f * PI) / (fabsf(dot(r.direction, normal)));
        
        float4 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        lastPos = f3(shadingPos);
        lastMaterialID = materialID;
        lastUV = uv;
        lastBackface = backface;
        lastInDirLocal = f3(incomingDirLocal);
        lastNormal = f3(normal);

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;
        if (!currDelta) {
            // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
            rand(&localState);
            rand4(&localState);

            rand(&localState); // to do the reservoir roll
        }

        float4 outgoing;
        float4 f_val_bsdf;
        float pdf_bsdf;

        sample_f_eval(
            localState, 
            params.shadeContext.materials, 
            materialID, 
            params.shadeContext.textures, 
            incomingDirLocal,
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            backface,
            outgoing,
            f_val_bsdf,
            pdf_bsdf,
            uv,
            TRANSPORTMODE_RADIANCE
        );

        if (pdf_bsdf < EPSILON && !K_is_D(type) && (pathLength != 2))
        {
            return {false, f3(0), 0.0f, 0.0f, 0.0f}; // something went wrong, cant finish temporal shift
        }

        
        throughput *= f3(f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf);
        lastCosine = fabsf(outgoing.z); 
        toWorld(outgoing, normal, outgoing);

        r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;
        r.direction = outgoing;

        prevDelta = currDelta;
        lastPDF = pdf_bsdf;
    }
        float4 lastPOS_GETRIDOFME = r.origin;

        // depth + 1 is the "index" of the curr vertex, so this stops at y_k-1
        for (int depth = 1; depth + 1 < pathLength; depth++) {
            SurfaceHit hitData = traceClosest(params, r);

            if (!hitData.isHit) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
            }

            int materialID;
            float2 uv;
            float4 shadingPos;
            bool backface;
            float4 normal;
            float4 lightEmission;
            const Triangle& tri = params.shadeContext.scene[hitData.primId];

            getData(
                tri,
                params.shadeContext,
                hitData.barycentrics, 
                r.direction,

                materialID,
                uv,
                shadingPos,
                normal,
                backface, 
                lightEmission
            );
            
            if constexpr (!isReverseShift) {
                if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
                    drawLine(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos, 
                        f4(0.0f, 1.0f, 1.0f), 3
                    );
                    printf("replaying shading pos depth %u: %f, %f, %f\n", depth, shadingPos.x, shadingPos.y, shadingPos.z);
                }
            }

            float4 incomingDirLocal;
            toLocal(r.direction, normal, incomingDirLocal);

            // needed for recon
            if (depth + 1 == rcVertexIndex - 1) {
                lastPos = f3(shadingPos);
                lastMaterialID = materialID;
                lastUV = uv;
                lastBackface = backface;
                lastInDirLocal = f3(incomingDirLocal);
                lastNormal = f3(normal);
            }

            float4 outgoing;
            float4 f_val_bsdf;
            float pdf_bsdf;

            sample_f_eval(
                localState, 
                params.shadeContext.materials, 
                materialID, 
                params.shadeContext.textures, 
                incomingDirLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                backface,
                outgoing,
                f_val_bsdf,
                pdf_bsdf,
                uv,
                TRANSPORTMODE_RADIANCE
            );

            lightEmission = backface ? f4(0.0f) : lightEmission;
            if (luminance(lightEmission) > 0.0f) {
                rand(&localState); // for the reservoir roll
            }

            bool currDelta = params.shadeContext.materials[materialID].isSpecular;

            float forwardFootprint = currDelta ? 0.0f : ((hitData.t * hitData.t) / (lastPDF * fabsf(incomingDirLocal.z))); // last pdf times geometry term arriving to curr
            float inverseFootprint = prevDelta ? 0.0f : ((hitData.t * hitData.t) / (pdf_bsdf * lastCosine)); // complicated stuff; see inverse footprint in paper
            
            bool isValid = true;
            if (fminf(forwardFootprint, inverseFootprint) >= primaryFootprint) {
                isValid = false;
            }

            if (!isValid) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f}; 
            }

            if (!currDelta) {
                // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
                rand(&localState);
                rand4(&localState);

                rand(&localState); // for the reservoir roll
            }

            float lum = luminance(f4(throughput));
            float p = clamp(lum, 0.05f, 1.0f);
            float rr_roll = rand(&localState);
            throughput /= p;

            if (pdf_bsdf < EPSILON && !K_is_D(type) && (pathLength != depth + 2))
            {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }
            toWorld(outgoing, normal, outgoing);
            throughput *= f3(f_val_bsdf) * fabsf(outgoing.z) / pdf_bsdf;
            r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;
            r.direction = outgoing;

            prevDelta = currDelta;
            lastPDF = pdf_bsdf;
            lastCosine = fabsf(dot(outgoing, normal));
            lastPOS_GETRIDOFME = shadingPos;
        }

        // now we are on the last bounce
        SurfaceHit hitData = traceClosest(params, r);

        if (is_env(type)) {
            if (hitData.isHit) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }
            ShiftResult result;

            float pdf_sampleLight = params.shadeContext.lightSampler.evaluateEnvPdf(r.direction);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                pdf_sampleLight // alternate strategy
            );

            result.contribution = 
                throughput * 
                f3(params.shadeContext.lightSampler.envMap.sampleDir(r.direction)) * 
                misWeight;

            result.isValid = true;
            result.p_hat = targetFunction(result.contribution);
            result.jacobian = 1.0f;
            result.new_cached_jacobian = 1.0f;
            
            return result;
        } else {
            if (!hitData.isHit) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }
        }

        if (!is_bsdf(type)) {
            printf("Error: full replay for bsdf has wrongly packed type or wrongly chosen rcvertexindex flag\n");
            return {false, f3(0), 0.0f, 0.0f, 0.0f};
        }

        int materialID;
        float2 uv;
        float4 shadingPos;
        bool backface;
        float4 normal;
        float4 lightEmission;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];

        getData(
            tri,
            params.shadeContext,
            hitData.barycentrics, 
            r.direction,

            materialID,
            uv,
            shadingPos,
            normal,
            backface, 
            lightEmission
        );

        float4 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        if constexpr (!isReverseShift) {
            if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
                drawLine(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos, 
                    f4(0.0f, 1.0f, 1.0f), 3
                );
                printf("replaying shading pos depth %u: %f, %f, %f\n", pathLength-1, shadingPos.x, shadingPos.y, shadingPos.z);
            }
        }

        lightEmission = backface ? f4(0.0f) : lightEmission;
        if (luminance(lightEmission) > 0.0f) {
            float sampleLightPDF = params.shadeContext.lightSampler.evaluateMeshPdf(tri);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                (hitData.t * hitData.t * sampleLightPDF / (fabsf(incomingDirLocal.z))) // alternate strategy
            );

            ShiftResult result;
            
            result.contribution = throughput * f3(lightEmission) * misWeight;

            result.isValid = true;
            result.p_hat = targetFunction(result.contribution);
            result.jacobian = 1.0f;
            result.new_cached_jacobian = 1.0f;

            return result;

        } else {
            return {false, f3(0), 0.0f, 0.0f, 0.0f};
        }

    } else { // handles all nee k=d cases, and the k=d environment case when the previous vertex is not specular
        float3 throughput = f3(1.0f);

        // these three may be unnecesary
        float lastPDF;
        bool prevDelta;
        float lastCosine;

        float3 lastPos;
        float3 lastNormal;
        int lastMaterialID;
        float2 lastUV;
        bool lastBackface;
        float3 lastInDirLocal;

        float primaryFootprint;
    {
        SurfaceHit hitData = traceClosest(params, r);

        if (!hitData.isHit) {
            return {false, f3(0), 0.0f, 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
        }

        int materialID;
        float2 uv;
        float4 shadingPos;
        bool backface;
        float4 normal;
        float4 ImplicitEmission;
        const Triangle& tri = params.shadeContext.scene[hitData.primId];

        getData(
            tri,
            params.shadeContext,
            hitData.barycentrics, 
            r.direction,

            materialID,
            uv,
            shadingPos,
            normal,
            backface,
            ImplicitEmission
        );
        if constexpr (!isReverseShift) {
            if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
                printf("replaying shading pos depth 0: %f, %f, %f\n", shadingPos.x, shadingPos.y, shadingPos.z);
            }
        }

        primaryFootprint = 
            (RECON_FOOTPRINT_C_CONSTANT / 100.0f) * 
            (hitData.t * hitData.t * 4.0f * PI) / (fabsf(dot(r.direction, normal)));
        
        float4 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        lastPos = f3(shadingPos);
        lastMaterialID = materialID;
        lastUV = uv;
        lastBackface = backface;
        lastInDirLocal = f3(incomingDirLocal);
        lastNormal = f3(normal);

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;
        if (!currDelta) {
            // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
            rand(&localState);
            rand4(&localState);

            rand(&localState); // to do the reservoir roll
        }

        float4 outgoing;
        float4 f_val_bsdf;
        float pdf_bsdf;

        sample_f_eval(
            localState, 
            params.shadeContext.materials, 
            materialID, 
            params.shadeContext.textures, 
            incomingDirLocal,
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            backface,
            outgoing,
            f_val_bsdf,
            pdf_bsdf,
            uv,
            TRANSPORTMODE_RADIANCE
        );

        if (pdf_bsdf < EPSILON)
        {
            return {false, f3(0), 0.0f, 0.0f, 0.0f}; // something went wrong, cant finish temporal shift
        }

        if (!K_is_D(type)) {
            throughput *= f3(f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf);
        }
        lastCosine = fabsf(outgoing.z); 
        toWorld(outgoing, normal, outgoing);

        r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;
        r.direction = outgoing;

        prevDelta = currDelta;
        lastPDF = pdf_bsdf;
    }
        float4 lastPOS_GETRIDOFME = r.origin;

        uint32_t loopBound = (K_is_D(type)) ?
            pathLength:
            rcVertexIndex;
        // depth + 1 is the "index" of the curr vertex, so this stops at y_k-1
        for (int depth = 1; depth + 1 < loopBound; depth++) {
            SurfaceHit hitData = traceClosest(params, r);

            if (!hitData.isHit) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f}; // Something went wrong, and the shift cannot be completed
            }

            int materialID;
            float2 uv;
            float4 shadingPos;
            bool backface;
            float4 normal;
            float4 lightEmission;
            const Triangle& tri = params.shadeContext.scene[hitData.primId];

            getData(
                tri,
                params.shadeContext,
                hitData.barycentrics, 
                r.direction,

                materialID,
                uv,
                shadingPos,
                normal,
                backface, 
                lightEmission
            );
            if constexpr (!isReverseShift) {
                
                if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
                    drawLine(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos, 
                        f4(0.0f, 1.0f, 1.0f), 3
                    );
                    printf("replaying shading pos depth %u: %f, %f, %f\n", depth, shadingPos.x, shadingPos.y, shadingPos.z);
                }
            }

            float4 incomingDirLocal;
            toLocal(r.direction, normal, incomingDirLocal);

            // needed for recon
            if (depth + 1 == rcVertexIndex - 1) {
                lastPos = f3(shadingPos);
                lastMaterialID = materialID;
                lastUV = uv;
                lastBackface = backface;
                lastInDirLocal = f3(incomingDirLocal);
                lastNormal = f3(normal);
            }

            float4 outgoing;
            float4 f_val_bsdf;
            float pdf_bsdf;

            sample_f_eval(
                localState, 
                params.shadeContext.materials, 
                materialID, 
                params.shadeContext.textures, 
                incomingDirLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                backface,
                outgoing,
                f_val_bsdf,
                pdf_bsdf,
                uv,
                TRANSPORTMODE_RADIANCE
            );

            lightEmission = backface ? f4(0.0f) : lightEmission;
            if (luminance(lightEmission) > 0.0f) {
                rand(&localState); // for the reservoir roll
            }

            bool currDelta = params.shadeContext.materials[materialID].isSpecular;

            float forwardFootprint = currDelta ? 0.0f : ((hitData.t * hitData.t) / (lastPDF * fabsf(incomingDirLocal.z))); // last pdf times geometry term arriving to curr
            float inverseFootprint = prevDelta ? 0.0f : ((hitData.t * hitData.t) / (pdf_bsdf * lastCosine)); // complicated stuff; see inverse footprint in paper
            
            bool isValid = true;
            if (fminf(forwardFootprint, inverseFootprint) >= primaryFootprint) {
                isValid = false;
            }
            //optixReorder(isValid ? 1 : 0, 1);
            if (!isValid) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f}; 
            }

            if (!currDelta) {
                // NEE cast takes 5 random numbers always. This wont get compiled out since it modifes the internal state
                rand(&localState);
                rand4(&localState);

                rand(&localState); // for the reservoir roll
            }

            float lum = luminance(f4(throughput));
            float p = clamp(lum, 0.05f, 1.0f);
            float rr_roll = rand(&localState);
            throughput /= p;

            if (pdf_bsdf < EPSILON)
            {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }
            toWorld(outgoing, normal, outgoing);
            if (depth + 1 != rcVertexIndex - 1) {
                throughput *= f3(f_val_bsdf) * fabsf(outgoing.z) / pdf_bsdf;
                r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;
                r.direction = outgoing;
            }

            prevDelta = currDelta;
            lastPDF = pdf_bsdf;
            lastCosine = fabsf(dot(outgoing, normal));
            lastPOS_GETRIDOFME = shadingPos;
        }

        // Now, we should be at y_k-1. At this point, rng syncing with the prefix is not important
        int materialID;
        float2 uv;
        float4 xk_pos;
        bool backface;
        float4 normal;
        float4 emission;

        if (rcPrimID != 0xFFFFFFFF) {
            if (rcPrimID >= params.shadeContext.triNum || rcPrimID < 0) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }

            const Triangle& tri = params.shadeContext.scene[rcPrimID];

            getDataWithoutInDirection(
                tri,
                params.shadeContext,
                rcBarycentrics, 
                lastPos,

                materialID,
                uv,
                xk_pos,
                normal,
                backface, 
                emission
            );
        }

        float3 new_payload_F = f3(0.0f);
        float new_cached_jacobian = 0.0f;

        float4 visibility_dir;
        float visibility_dist;

        float4 outDir;
        if (is_internal_rc_vertex(type)) {
            outDir = xk_pos - f4(lastPos);
            visibility_dist = length(outDir);
            outDir = normalize(outDir);
            visibility_dir = outDir;
        } else {
            if (is_env(type)) {
                outDir = f4(rcWi); // from the last intersect to the light pos
                visibility_dist = 1E30;
                visibility_dir = outDir;
            } else {
                visibility_dist = length(xk_pos - f4(lastPos));
                outDir = f4(rcWi);
                visibility_dir = outDir;
            }
        }

        if (dot(visibility_dir, normal) > 0.0f) {
            normal = -normal;
        }

        if constexpr (!isReverseShift) {
            if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
                drawLine(params.overlay_buffer, params.camera, f4(lastPos), f4(lastPos) + (visibility_dir * visibility_dist), 
                    f4(0.7f, 0.0f, 1.0f), 3
                );
            }
        }
        bool occluded = traceVisibility(
            params, 
            Ray(f4(lastPos) + (visibility_dir * RAY_EPSILON), visibility_dir),
            visibility_dist * (1.0f - EPSILON3)
        );

        ShiftResult result;
        result.isValid = true;
        result.contribution = f3(-1.0f);
        result.jacobian = -1.0f;
        result.new_cached_jacobian = -1.0f;
        result.p_hat = -1.0f;

        if (occluded) {
            return {false, f3(0), 0.0f, 0.0f, 0.0f};
        } else if (is_internal_rc_vertex(type)) {
            float connectionDistanceSQR = lengthSquared(xk_pos - f4(lastPos));
            
            float4 outDirLocal;
            toLocal(outDir, f4(lastNormal), outDirLocal);
            float cosine_1 = fabsf(outDirLocal.z);
            float4 f_val_1;
            float pdf_1;

            f_pdf_eval(
                params.shadeContext.materials, 
                lastMaterialID, 
                params.shadeContext.textures, 
                f4(lastInDirLocal),
                outDirLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                f_val_1,
                pdf_1,
                lastUV
            );

            float4 inDir = outDir;
            toLocal(inDir, normal, inDir);
            float cosine_2 = fabsf(inDir.z);

            outDir = f4(rcWi);
            toLocal(outDir, normal, outDir);
            float cosine_rcWi = outDir.z;
            float4 f_val_2;
            float pdf_2;

            f_pdf_eval(
                params.shadeContext.materials, 
                materialID, 
                params.shadeContext.textures, 
                inDir,
                outDir,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                f_val_2,
                pdf_2,
                uv
            );
            
            float G = cosine_2 / (connectionDistanceSQR);

            float p_new_suffix = pdf_1 * G;

            if (!(K_is_D_minus_1(type) && is_nee(type))) {
                p_new_suffix *= pdf_2;
            }
            new_cached_jacobian = p_new_suffix;

            if (p_new_suffix <= 0.0f || jacobianDenom <= 0.0f) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }

            float jacobian = p_new_suffix / jacobianDenom;

            if (pdf_1 <= 0.0f) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }
            if (pdf_2 <= 0.0f && !(K_is_D_minus_1(type) && is_nee(type))) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }
            
            if (K_is_D_minus_1(type)) {
                // Cached_nee is always in solid angle, if not k=d
                float p_sampled_light = (is_nee(type)) ? cached_nee : pdf_2;

                result.contribution = throughput 
                    * (f3(f_val_1) * cosine_1 / pdf_1) 
                    * (f3(f_val_2) * cosine_rcWi / p_sampled_light) 
                    * rcRadiance; // here rcRadiance is the raw emission
            } else { // its a deep internal vertex
                result.contribution = throughput 
                    * (f3(f_val_1) * cosine_1 / pdf_1) 
                    * (f3(f_val_2) * cosine_rcWi / pdf_2) 
                    * rcRadiance; // here rc radiance is the actual incoming radiance calculated with suffix throughput etc.
            }
            

            if (K_is_D_minus_1(type)) {
                float misWeight = powerHeuristicTwoStrategy(
                    (is_nee(type)) ? cached_nee : pdf_2,
                    (is_nee(type)) ? pdf_2 : cached_nee
                );

                result.contribution *= misWeight;
            }

            result.p_hat = targetFunction(result.contribution);
            result.jacobian = jacobian;
            result.new_cached_jacobian = new_cached_jacobian;
            if (result.p_hat <= 0.0f) {
                return {false, f3(0), 0.0f, 0.0f, 0.0f};
            }
        } else { // K=D
            if (is_env(type)) {
                float4 outDirLocal;
                toLocal(outDir, f4(lastNormal), outDirLocal);
                float cosine_surface = fabsf(outDirLocal.z);
                float4 f_val_1;
                float pdf_1;

                f_pdf_eval(
                    params.shadeContext.materials, 
                    lastMaterialID, 
                    params.shadeContext.textures, 
                    f4(lastInDirLocal),
                    outDirLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    f_val_1,
                    pdf_1,
                    lastUV
                );

                float pathMISWeight = powerHeuristicTwoStrategy( // cached_nee should be the raw solid angle pdf
                    (is_nee(type)) ? cached_nee : pdf_1,
                    (is_nee(type)) ? pdf_1 : cached_nee
                );

                float p_sampled = (is_nee(type)) ? cached_nee : pdf_1;
                if (p_sampled <= 0.0f || (!is_nee(type) && pdf_1 <= 0.0f)) {
                    return {false, f3(0), 0.0f, 0.0f, 0.0f};
                }

                result.contribution = pathMISWeight * (throughput * f3(f_val_1) * cosine_surface * rcRadiance / p_sampled);

                float jacobian;
                if (is_bsdf(type)) { // Direction copy
                    // jacobianDenom should just be a single pdf for direction copy
                    if (jacobianDenom <= 0.0f || pdf_1 <= 0.0f) {
                        return {false, f3(0), 0.0f, 0.0f, 0.0f};
                    }
                    jacobian = pdf_1 / jacobianDenom;
                    new_cached_jacobian = pdf_1;
                } else {  // random replay takes care of the entire shift
                    jacobian = 1.0f;
                    new_cached_jacobian = 1.0f;
                }
                result.jacobian = jacobian;
                result.new_cached_jacobian = new_cached_jacobian;
                
                result.p_hat = targetFunction(result.contribution);
                
                if (result.p_hat <= 0.0f) {
                    return {false, f3(0), 0.0f, 0.0f, 0.0f};
                }
            } else { // MUST be NEE case, since k=d area light bsdf is a full replay version
                float connectionDistanceSQR = lengthSquared(xk_pos - f4(lastPos));

                float4 outDirLocal;
                toLocal(outDir, f4(lastNormal), outDirLocal);
                float cosine_surface = fabsf(outDirLocal.z);
                float4 f_val_1;
                float pdf_1;

                f_pdf_eval(
                    params.shadeContext.materials, 
                    lastMaterialID, 
                    params.shadeContext.textures, 
                    f4(lastInDirLocal),
                    outDirLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    f_val_1,
                    pdf_1,
                    lastUV
                );

                float cosine_light = fabsf(dot(normal, outDir));

                // if sampled light is area (not env), then perform area to solid angle conversion
                float converted_cached_nee = cached_nee * (connectionDistanceSQR) / fmaxf(cosine_light, EPSILON);

                float p_sampled = converted_cached_nee;
                if (p_sampled <= 0.0f) {
                    return {false, f3(0), 0.0f, 0.0f, 0.0f};
                }

                float pathMISWeight = powerHeuristicTwoStrategy(
                    converted_cached_nee,
                    pdf_1
                );

                result.contribution = pathMISWeight * (throughput * f3(f_val_1) * cosine_surface * rcRadiance / p_sampled);
                
                result.p_hat = targetFunction(result.contribution);
                
                if (result.p_hat <= 0.0f) {
                    return {false, f3(0), 0.0f, 0.0f, 0.0f};
                }
                
                result.jacobian = 1.0f;
                result.new_cached_jacobian = 1.0f;
            }
        }

        return result;
    }
}

__device__ __forceinline__ bool isHistoryValid(const PipelineParams& params, int2 currentCoord, half2 motionVec, int2& out_coords) {
    out_coords = make_int2(-1, -1);
    int2 history_coord = make_int2(currentCoord.x - (int)roundf(__half2float(motionVec.x)), 
                            currentCoord.y - (int)roundf(__half2float(motionVec.y)));
    uint32_t current_idx = currentCoord.x + currentCoord.y * params.common.w;
    uint32_t history_idx = history_coord.x + history_coord.y * params.common.w;
    if (history_coord.x < 0 || history_coord.x >= params.common.w || 
        history_coord.y < 0 || history_coord.y >= params.common.h) {
        return false;
    }

    uint32_t current_id = params.restir.gbuffer.getMatID(current_idx);
    uint32_t history_id = params.restir.prevGbuffer.getMatID(history_idx);

    if (current_id != history_id) {
        return false;
    }

    float3 currNorm = params.restir.gbuffer.getNormal(current_idx);
    float3 pastNorm = params.restir.prevGbuffer.getNormal(history_idx);

    if (dot(currNorm, pastNorm) < 0.98f) {
        return false; 
    }

    // the pixel jitter is easily recreated since it is always spawned from the first two random numbers after the seed
    RNGState localState = load_rng(current_idx, params.common.frame_index, 0, nullptr);
    Ray r = params.common.camera.generateCameraRay(localState, currentCoord.x, currentCoord.y);

    float3 current_pos = f3(r.at(params.restir.gbuffer.getDepth(current_idx)));

    localState = load_rng(history_idx, params.common.frame_index - 1, 0, nullptr);
    r = params.restir.lastFrameCamera.generateCameraRay(localState, history_coord.x, history_coord.y);

    float3 history_pos = f3(r.at(params.restir.prevGbuffer.getDepth(history_idx)));

    float3 pos_diff = history_pos - current_pos;

    float plane_distance = abs(dot(pos_diff, currNorm));

    float depth_tolerance = 0.01f + (length(current_pos - f3(params.common.camera.cameraOrigin)) * 0.01f);

    if (plane_distance > depth_tolerance) {
        return false;
    }

    float true_distance = length(pos_diff);
    if (plane_distance > depth_tolerance || true_distance > depth_tolerance * 5.0f) {
        return false;
    }

    out_coords = history_coord;
    return true;
}

__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ float targetFunction(float4 contribution) {
    return luminance(fireflyClamp(contribution));
}

__device__ __forceinline__ float targetFunction(float3 contribution) {
    return luminance(fireflyClamp(f4(contribution)));
}

__device__ int2 get_frame_offset(uint32_t frame_idx, uint32_t texture_id, int2 texture_size) {
    uint32_t hx = hash_uint32(frame_idx ^ (texture_id * 1973u));
    uint32_t hy = hash_uint32(hx);
    return make_int2(hx % texture_size.x, hy % texture_size.y);
}

__device__ bool get_frame_flip_x(uint32_t frame_idx, uint32_t texture_id) {
    uint32_t h = hash_uint32(frame_idx ^ (texture_id * 31337u));
    return (h & 1) != 0;
}

__device__ bool get_frame_transpose(uint32_t frame_idx, uint32_t texture_id) {
    uint32_t h = hash_uint32(frame_idx ^ (texture_id * 8128u));
    return (h & 1) != 0;
}

__device__ short2 get_neighbor_delta(
    int2 screen_pixel, 
    uint32_t texture_id, 
    uint32_t frame_idx, 
    int2 texture_size, 
    const short2* __restrict__ pairing_buffer) 
{
    int2 random_offset = get_frame_offset(frame_idx, texture_id, texture_size);
    bool flip_x        = get_frame_flip_x(frame_idx, texture_id);
    bool transpose     = get_frame_transpose(frame_idx, texture_id);

    int2 tex_coord = screen_pixel;
    
    if (transpose) {
        tex_coord = make_int2(tex_coord.y, tex_coord.x);
    }
    
    // Apply flip across X
    if (flip_x) {
        tex_coord.x = texture_size.x - 1 - tex_coord.x;
    }
    
    // Apply offset and wrap around the boundaries
    tex_coord.x = (tex_coord.x + random_offset.x) % texture_size.x;
    tex_coord.y = (tex_coord.y + random_offset.y) % texture_size.y;
    
    // Ensure positive modulo wrapping
    if (tex_coord.x < 0) tex_coord.x += texture_size.x;
    if (tex_coord.y < 0) tex_coord.y += texture_size.y;

    // 3. Flatten 2D coord to 1D index for the raw short2* buffer
    uint32_t flat_index = tex_coord.y * texture_size.x + tex_coord.x;
    
    // Read the raw delta and cast up to int2 for math
    short2 raw_short_delta = pairing_buffer[flat_index];
    short2 final_delta = make_short2(raw_short_delta.x, raw_short_delta.y);

    if (transpose) {
        final_delta = make_short2(final_delta.y, final_delta.x);
    }
    if (flip_x) {
        final_delta.x = -final_delta.x;
    }

    return final_delta;
}

__device__ __forceinline__ float3 debugVisualizeTechnique(uint32_t type) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    // 1. Reconnection Depth -> Primary Color Axis
    if (type & SHIFT_K_IS_D) {
        color = make_float3(1.0f, 0.0f, 0.0f); // Base: Red
    } 
    else if (type & SHIFT_K_IS_D_MINUS_1) {
        color = make_float3(0.0f, 1.0f, 0.0f); // Base: Green
    } 
    else if (type & SHIFT_K_LESS_D_MINUS_1) {
        color = make_float3(0.0f, 0.0f, 1.0f); // Base: Blue
    }

    // 2. Light Type -> Shift to Secondary Color
    if (type & SHIFT_IS_ENV) {
        if (type & SHIFT_K_IS_D)                 color.y = 1.0f; // Red -> Yellow
        else if (type & SHIFT_K_IS_D_MINUS_1)    color.z = 1.0f; // Green -> Cyan
        else if (type & SHIFT_K_LESS_D_MINUS_1)  color.x = 1.0f; // Blue -> Magenta
    }

    // 3. Sampling Method -> Intensity
    // NEE = Vivid (1.0), BSDF = Dim/Muted (0.35)
    float intensity = (type & SHIFT_IS_NEE) ? 1.0f : 0.35f;

    return color * intensity;
}

__device__ __forceinline__ float3 debugVisualizeTechniqueAndLength(uint32_t type, uint32_t pathLength) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    // 1. Reconnection Depth (K) -> Base Hue
    if (type & SHIFT_K_IS_D)                 color = make_float3(1.0f, 0.0f, 0.0f); // Base: Red
    else if (type & SHIFT_K_IS_D_MINUS_1)    color = make_float3(0.0f, 1.0f, 0.0f); // Base: Green
    else if (type & SHIFT_K_LESS_D_MINUS_1)  color = make_float3(0.0f, 0.0f, 1.0f); // Base: Blue

    // 2. Light Type -> Shift to Secondary Hue
    if (type & SHIFT_IS_ENV) {
        if (type & SHIFT_K_IS_D)                 color.y = 1.0f; // Red -> Yellow
        else if (type & SHIFT_K_IS_D_MINUS_1)    color.z = 1.0f; // Green -> Cyan
        else if (type & SHIFT_K_LESS_D_MINUS_1)  color.x = 1.0f; // Blue -> Magenta
    }

    // 3. Sampling Method -> Saturation
    // NEE = Pure/Vivid. BSDF = Mix with 65% white (Pastel/Chalky)
    if (!(type & SHIFT_IS_NEE)) {
        float pastel_blend = 0.65f;
        color.x = color.x + (1.0f - color.x) * pastel_blend;
        color.y = color.y + (1.0f - color.y) * pastel_blend;
        color.z = color.z + (1.0f - color.z) * pastel_blend;
    }

    // 4. Path Length -> Brightness / Value
    // Assuming minimum path length is 2. 
    // Length 2 = 100% brightness. Each extra bounce darkens it by 15%.
    float brightness = fmaxf(1.0f - (float)(pathLength - 2) * 0.15f, 0.15f);

    return make_float3(color.x * brightness, color.y * brightness, color.z * brightness);
}