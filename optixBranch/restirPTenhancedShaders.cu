#include <optix.h>
#include <optix_device.h>
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"
#include "helpers.cuh"
#include "restirPTenhanced_helpers.cuh"
#include "settings.cuh"

#ifndef TEMPORAL_USE_DUAL_MV
#define TEMPORAL_USE_DUAL_MV 1
#endif

#ifndef TEMPORAL_SER_SORT_MORTON_CODE
#define TEMPORAL_SER_SORT_MORTON_CODE 1
#endif


extern "C" {
    __constant__ PipelineParams allParams;
}

extern "C" __global__ void __raygen__restirCandidateGeneration() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers

    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;

    RNGState localState = load_rng(pixelIdx, params.frame_index, 0, nullptr);

    uint32_t seed = localState.getSeed();
    restir.reservoir.setInitRandomSeed(pixelIdx, seed);
    half2 jitter;
    Ray r = params.camera.generateCameraRayRecordOffset(localState, x, y, jitter);

    float3 throughput = f3(1.0f);
    float3 suffixThroughput = f3(1.0f);
    float lastPDF;
    bool prevDelta;
    float lastCosine;

    float3 lastPOS_GETRIDOFME; // for displaying the debug paths

    float w_sum = 0.0f;
    uint32_t F = 0;
    uint32_t pathFlags = 0;
    uint32_t pathRcVertexIndex = FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND; // mark unchosen
    uint4 pathRcVertexGeometry = make_uint4(0, 0, 0, 0);
    uint4 actualRcVertexGeometry = make_uint4(0, 0, 0, 0);
    float pathCachedJacobian = 0.0f;
    float actualCachedJacobian = 0.0f;
    float neepdf = -1.0f;

    float primaryFootprint;
    /**
     * Trace Primary Hit, special case
     */

{
    SurfaceHit hitData = traceClosest(params, r);
    if (!hitData.isHit)
    {
        float3 contribution = params.shadeContext.lightSampler.envMap.sampleDir(r.direction);

#if ACCUMULATE_FRAMES == 1
        params.accum_buffer[pixelIdx] += f4(contribution);
#else
        params.accum_buffer[pixelIdx] = f4(contribution);
#endif
        restir.reservoir.pathFlags[pixelIdx] = 0;
        restir.gbuffer.setInvalidMotionVec(pixelIdx);
        save_rng(pixelIdx, &localState, nullptr);
        return;
    }

    int materialID;
    float2 uv;
    float3 shadingPos;
    bool backface;
    float3 normal;
    float3 ImplicitEmission;
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

    if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
        printf("candidate gen shading pos depth 0: %f, %f, %f\n", shadingPos.x, shadingPos.y, shadingPos.z);
    }

    float3 albedo;

    getAlbedo(
        params.shadeContext.materials,
        materialID,
        params.shadeContext.textures,
        uv,
        albedo
    );

    restir.gbuffer.setGeometry(pixelIdx, normal, hitData.t, materialID, albedo);

    float2 currPixelPos = make_float2((float)x + __half2float(jitter.x), (float)y + __half2float(jitter.y));
    float2 lastPixelPos;

    if (params.frame_index != 0) {
        restir.lastFrameCamera.worldToRaster(shadingPos, lastPixelPos);
        restir.gbuffer.setMotionVec(pixelIdx, currPixelPos - lastPixelPos); // validity is double checked in temporal phase
    }

    primaryFootprint =
        (RECON_FOOTPRINT_C_CONSTANT / 100.0f) *
        (hitData.t * hitData.t * 4.0f * PI) / (fabsf(dot(r.direction, normal)));

    float3 contribution = backface ? f3(0.0f) : ImplicitEmission;

    if (luminance(contribution) > EPSILON) {
#if ACCUMULATE_FRAMES == 1
        params.accum_buffer[pixelIdx] += f4(contribution);
#else
        params.accum_buffer[pixelIdx] = f4(contribution);
#endif
        restir.gbuffer.setSkipShadeFlag(pixelIdx); // still run a reservoir and reuse normally, but dont display the reservoir this frame
    }

    float3 incomingDirLocal;
    toLocal(r.direction, normal, incomingDirLocal);

    bool currDelta = params.shadeContext.materials[materialID].isSpecular;

    // Handle DI, special case
    if (!currDelta) {
        // This block ALWAYS consumes 5 + 1 = 6 rng calls
        float3 lightNormal;
        float3 emission;
        float3 shadingPosToLightNormalized;
        float t_max;
        float pdf_nee;

        uint32_t neePrimID; // 0xFFFFFFFF for env, otherwise the triangle primID
        float2 neeBarycentrics;

        bool sampledEnv = params.shadeContext.lightSampler.sample_ReSTIR_rc_data(
            rand(&localState), rand4(&localState),
            shadingPos,
            params.shadeContext.vertices,
            emission,
            shadingPosToLightNormalized,
            lightNormal,
            t_max,
            pdf_nee,
            neePrimID,
            neeBarycentrics
        );

        float3 shadingPosToLightLocal;
        toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

        bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
        bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);
        if (!surfaceBackface && !lightBackface) {
            float bsdfPDF;

            pdf_eval(
                params.shadeContext.materials,
                materialID,
                params.shadeContext.textures,
                incomingDirLocal,
                shadingPosToLightLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                bsdfPDF,
                uv
            );

            float3 f_val_nee;
            f_eval(
                params.shadeContext.materials,
                materialID,
                params.shadeContext.textures,
                incomingDirLocal,
                shadingPosToLightLocal,
                1.5f, // change later when medium stack integrated
                1.5f, // change later
                f_val_nee,
                uv
            );

            float3 contribution;
            float misWeight;

            float cosLight = dot(-shadingPosToLightNormalized, lightNormal);
            float cosSurface = dot(normal, shadingPosToLightNormalized);
            if (sampledEnv) {
                contribution = f_val_nee * emission * cosSurface / pdf_nee;
                misWeight = powerHeuristicTwoStrategy(
                    pdf_nee,
                    bsdfPDF
                );
            } else {
                contribution =
                    f_val_nee * emission * cosLight * // NEE contribution
                    cosSurface / (pdf_nee * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                misWeight = powerHeuristicTwoStrategy(
                    (t_max * t_max) * pdf_nee / cosLight, // convert area pdf to SA
                    bsdfPDF // alt strat
                );
            }

            bool occluded = traceVisibility(
                params,
                Ray((shadingPos + shadingPosToLightNormalized * RAY_EPSILON), shadingPosToLightNormalized),
                t_max * (1.0f - EPSILON2)
            );

            if (!occluded) {

                float w_i = targetFunction(contribution * misWeight);

                w_sum += w_i;

                float roll = rand(&localState);
                if (w_sum > 0.0f && roll < w_i / w_sum) {

                    F = toRGB9E5(contribution * misWeight);
                    pathFlags = packPathFlags(
                        1,          // M = 1
                        2,          // Path Length
                        2,          // Rc vertex index (forces the light to be the rc vertex, k=d)
                        sampledEnv ? PATH_TYPE_NEE_ENV_K_EQ_D : PATH_TYPE_NEE_AREA_K_EQ_D
                    );

                    actualRcVertexGeometry = packRcGeometry(
                        neePrimID,  // Also flags whether or not it is an environment or area light via sentinel value
                        neeBarycentrics,
                        shadingPosToLightNormalized,   // undefined for k=d, but we store the direction of the sampled dir
                        emission    // since the light is the rc vertex, its just the emission
                    );

                    actualCachedJacobian = 1.0f; // di case.
                    //neepdf = (t_max * t_max * pdf_nee) / cosLight;
                    neepdf = pdf_nee; // must store origin measure pdf for k=d
                }
            } else {
                rand(&localState);
            }

        } else {
            rand(&localState);
        }
    }

    float3 outgoing;
    float3 f_val_bsdf;
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
        goto finalize_pixel;
    }

    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    float rr_roll = rand(&localState);
    if (rr_roll > p) {
        goto finalize_pixel;
    }
    throughput /= p;

    throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;
    lastCosine = fabsf(outgoing.z);
    toWorld(outgoing, normal, outgoing);

    r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;
    r.direction = outgoing;

    prevDelta = currDelta;
    lastPDF = pdf_bsdf;
    lastPOS_GETRIDOFME = shadingPos;
}
    for (int depth = 1; depth < params.max_depth; depth++)
    {
        SurfaceHit hitData = traceClosest(params, r);
        if (!hitData.isHit) // ENVIRONMENT
        {
            float pdf_sampleLight = params.shadeContext.lightSampler.evaluateEnvPdf(r.direction);
            float3 envEmission = params.shadeContext.lightSampler.envMap.sampleDir(r.direction);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                pdf_sampleLight // alternate strategy
            );

            float w_i = targetFunction(throughput * envEmission * misWeight);
            w_sum += w_i;

            float roll = rand(&localState);
            if (w_sum > 0.0f && roll < w_i / w_sum) {
                F = toRGB9E5(throughput * envEmission * misWeight);

                uint32_t pathType;
                uint32_t rcInd;
                if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) {
                    // k = d
                    actualRcVertexGeometry = packRcGeometry(
                        0xFFFFFFFF, // flags a env hit
                        f2(0.0f),   // undefined for env hit
                        r.direction,   // undefined for k=d, but we store the direction of the sampled dir
                        envEmission
                    );
                    actualCachedJacobian = prevDelta ? 1.0f : lastPDF; // direction copy for that case (if not direction copy, this variable isnt neccesary)
                    neepdf = pdf_sampleLight;
                    pathType = PATH_TYPE_BSDF_ENV_K_EQ_D;
                    rcInd = prevDelta ?
                        FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY : // direction copy is impossible if the prev vertex was full specular
                        FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_DIRECTION_COPY; // direction copy for environment map lowers variance
                } else if (pathRcVertexIndex == depth) {
                    // k = d - 1
                    // This means the previous iteration, the previous vertex was marked as the rc vertex, thus, the rcvertexgeometry
                    // holds a rcWi that points towards this current vertex, which is correct.

                    actualRcVertexGeometry = updateRcVertexRadiance(pathRcVertexGeometry, envEmission); // must save raw emission
                    actualCachedJacobian = pathCachedJacobian;
                    neepdf = pdf_sampleLight;
                    pathType = PATH_TYPE_BSDF_ENV_K_EQ_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                } else {
                    // k < d - 1
                    actualRcVertexGeometry = updateRcVertexRadiance(pathRcVertexGeometry, suffixThroughput * envEmission * misWeight);
                    actualCachedJacobian = pathCachedJacobian;
                    neepdf = -1.0f;
                    pathType = PATH_TYPE_BSDF_ENV_K_LESS_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                }

                pathFlags = packPathFlags(
                    1,          // M = 1
                    depth + 1,          // Path Length
                    rcInd,
                    pathType
                );
            }

            goto finalize_pixel;
        }

        int materialID;
        float2 uv;
        float3 shadingPos;
        bool backface;
        float3 normal;
        float3 emission;
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
            emission
        );


        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("candidate gen shading pos depth %u: %f, %f, %f\n", depth, shadingPos.x, shadingPos.y, shadingPos.z);
        }



        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            drawLine(params.overlay_buffer, params.camera, lastPOS_GETRIDOFME, shadingPos,
                (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) ?
                f3(1.0f, 0.0f, 0.0f) :
                f3(0.0f, 1.0f, 0.0f), 3
            );
        }


        float3 incomingDirLocal;
        toLocal(r.direction, normal, incomingDirLocal);

        float3 outgoing;
        float3 f_val_bsdf;
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

        //---------------------------------------------------------------------------------------------------------------------------------------------------
        // Check dual footprint for rc connectability
        //---------------------------------------------------------------------------------------------------------------------------------------------------

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;

        if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) { // to catch undefined pdfs
            float forwardFootprint = currDelta ? 0.0f : ((hitData.t * hitData.t) / (lastPDF * fabsf(incomingDirLocal.z))); // last pdf times geometry term arriving to curr
            float inverseFootprint = prevDelta ? 0.0f : ((hitData.t * hitData.t) / (pdf_bsdf * lastCosine)); // complicated stuff; see inverse footprint in paper

            if (fminf(forwardFootprint, inverseFootprint) >= primaryFootprint) {
                pathRcVertexIndex = depth + 1;

                float3 out_world;
                toWorld(outgoing, normal, out_world);

                pathRcVertexGeometry = packRcGeometry(
                    hitData.primId,
                    hitData.barycentrics,
                    out_world,
                    f3() // cannot yet be determined, this will be fillbut ed in when a candidate is streamed in
                );

                // p_(x_k-1 -> x_k) * G(x_k-1 -> x_k) * P(x_k -> x_k+1)
                pathCachedJacobian = lastPDF * pdf_bsdf * fabsf(incomingDirLocal.z) / (hitData.t * hitData.t);

                restir.reservoir.rcVertexRandomSeed[pixelIdx] = localState.getSeed();
            }
        }

        float3 lightEmission = backface ? f3(0.0f) : emission;

        if (luminance(lightEmission) > EPSILON) {
            float sampleLightPDF = params.shadeContext.lightSampler.evaluateMeshPdf(tri);
            float misWeight = (prevDelta) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                (hitData.t * hitData.t * sampleLightPDF / (fabsf(incomingDirLocal.z))) // alternate strategy
            );

            float w_i = targetFunction(throughput * lightEmission * misWeight);
            w_sum += w_i;

            float roll = rand(&localState);
            if (w_sum > 0.0f && roll < w_i / w_sum) {
                F = toRGB9E5(throughput * lightEmission * misWeight);

                uint32_t pathType;
                uint32_t rcInd;
                if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) {
                    // k = d
                    actualRcVertexGeometry = packRcGeometry(
                        hitData.primId,
                        hitData.barycentrics,
                        f3(0.0f),
                        suffixThroughput * lightEmission
                    );
                    //actualCachedJacobian = lastPDF * (fabsf(incomingDirLocal.z) / (hitData.t * hitData.t));
                    actualCachedJacobian = 1.0f; // we rely on a full replay for this
                    neepdf = sampleLightPDF; // must store area pdf for k=d
                    pathType = PATH_TYPE_BSDF_AREA_K_EQ_D;
                    rcInd = FLAG_HYBRID_SHIFT_RC_INDEX_K_IS_D_FULL_REPLAY;
                } else if (pathRcVertexIndex == depth) {
                    // k = d - 1
                    actualRcVertexGeometry = updateRcVertexRadiance(pathRcVertexGeometry, lightEmission);
                    actualCachedJacobian = pathCachedJacobian;
                    neepdf = (hitData.t * hitData.t * sampleLightPDF) / fabsf(incomingDirLocal.z);
                    pathType = PATH_TYPE_BSDF_AREA_K_EQ_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                } else {
                    // k < d - 1
                    actualRcVertexGeometry = updateRcVertexRadiance(pathRcVertexGeometry, suffixThroughput * lightEmission * misWeight);
                    actualCachedJacobian = pathCachedJacobian;
                    pathType = PATH_TYPE_BSDF_AREA_K_LESS_D_MINUS_1;
                    rcInd = pathRcVertexIndex;
                    neepdf = -1.0f;
                }

                pathFlags = packPathFlags(
                    1,          // M = 1
                    depth + 1,          // Path Length
                    rcInd,
                    pathType    // This was chosen via NEE
                );
            }
        }


        if (!currDelta) {
            float3 lightNormal;
            float3 emission;
            float3 shadingPosToLightNormalized;
            float t_max;
            float pdf_nee;

            uint32_t neePrimID; // 0xFFFFFFFF for env, otherwise the triangle primID
            float2 neeBarycentrics;

            bool sampledEnv = params.shadeContext.lightSampler.sample_ReSTIR_rc_data(
                rand(&localState), rand4(&localState),
                shadingPos,
                params.shadeContext.vertices,
                emission,
                shadingPosToLightNormalized,
                lightNormal,
                t_max,
                pdf_nee,
                neePrimID,
                neeBarycentrics
            );

            float3 shadingPosToLightLocal;
            toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

            bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
            bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);
            if (!surfaceBackface && !lightBackface) {
                float bsdfPDF;

                pdf_eval(
                    params.shadeContext.materials,
                    materialID,
                    params.shadeContext.textures,
                    incomingDirLocal,
                    shadingPosToLightLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    bsdfPDF,
                    uv
                );

                float3 f_val_nee;
                f_eval(
                    params.shadeContext.materials,
                    materialID,
                    params.shadeContext.textures,
                    incomingDirLocal,
                    shadingPosToLightLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    f_val_nee,
                    uv
                );

                float3 contributionSansThroughput;
                float misWeight;

                float cosLight;
                float cosSurface;
                if (sampledEnv) {
                    cosSurface = dot(normal, shadingPosToLightNormalized);
                    contributionSansThroughput = f_val_nee * emission * cosSurface / pdf_nee;
                    misWeight = powerHeuristicTwoStrategy(
                        pdf_nee,
                        bsdfPDF
                    );
                } else {
                    cosLight = dot(-shadingPosToLightNormalized, lightNormal);
                    cosSurface = dot(normal, shadingPosToLightNormalized);

                    contributionSansThroughput =
                        f_val_nee * emission * cosLight * // NEE contribution
                        cosSurface / (pdf_nee * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                    misWeight = powerHeuristicTwoStrategy(
                        (t_max * t_max) * pdf_nee / cosLight, // convert area pdf to SA
                        bsdfPDF // alt strat
                    );
                }

                bool occluded = traceVisibility(
                    params,
                    Ray((shadingPos + shadingPosToLightNormalized * RAY_EPSILON), shadingPosToLightNormalized),
                    t_max * (1.0f - EPSILON2)
                );

                if (!occluded) {
                    float w_i = targetFunction(throughput * contributionSansThroughput * misWeight);
                    w_sum += w_i;

                    float roll = rand(&localState);
                    if (w_sum > 0.0f && roll < w_i / w_sum) {
                        F = toRGB9E5(throughput * contributionSansThroughput * misWeight);

                        uint32_t pathType;
                        if (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) {
                            // k = d
                            actualRcVertexGeometry = packRcGeometry(
                                neePrimID,
                                neeBarycentrics,
                                shadingPosToLightNormalized,
                                emission // emission ONLY
                            );
                            actualCachedJacobian = 1.0f;
                            neepdf = pdf_nee; // must store original measure for k=d
                            pathType = sampledEnv ? PATH_TYPE_NEE_ENV_K_EQ_D : PATH_TYPE_NEE_AREA_K_EQ_D;
                        } else if (pathRcVertexIndex == depth + 1) {
                            // k = d - 1
                            //float3 rcRadiance = sampledEnv ? (emission / pdf_nee) : (emission * cosLight / (pdf_nee * t_max * t_max));
                            float3 rcRadiance = emission;
                            actualRcVertexGeometry = updateRcVertexRadiance(pathRcVertexGeometry, suffixThroughput * rcRadiance);
                            actualRcVertexGeometry = updateRcVertexWi(actualRcVertexGeometry, shadingPosToLightNormalized);
                            actualCachedJacobian = lastPDF * (fabsf(incomingDirLocal.z) / (hitData.t * hitData.t));
                            neepdf = sampledEnv ? (pdf_nee) : ((t_max * t_max * pdf_nee) / cosLight);
                            pathType = sampledEnv ? PATH_TYPE_NEE_ENV_K_EQ_D_MINUS_1 : PATH_TYPE_NEE_AREA_K_EQ_D_MINUS_1;
                        } else {
                            // k < d - 1
                            actualRcVertexGeometry = updateRcVertexRadiance(pathRcVertexGeometry, suffixThroughput * contributionSansThroughput * misWeight);
                            actualCachedJacobian = pathCachedJacobian;
                            neepdf = -1.0f;
                            pathType = sampledEnv ? PATH_TYPE_NEE_ENV_K_LESS_D_MINUS_1 : PATH_TYPE_NEE_AREA_K_LESS_D_MINUS_1;
                        }

                        pathFlags = packPathFlags(
                            1,          // M = 1
                            depth + 2,          // Path Length
                            (pathRcVertexIndex == FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND) ? // Rc vertex index (sentinel means it hasnt been found yet)
                                (depth + 2) : // not yet found, so set this sampled one as rc vertex; k=d special case
                                pathRcVertexIndex, // rc vertex found; save it
                            pathType    // This was chosen via NEE
                        );
                    }
                } else {
                    rand(&localState);
                }
            } else {
                rand(&localState);
            }
        }

        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);

        if (rand(&localState) > p)   // survive with probability p
        {
            goto finalize_pixel;
        }
        throughput /= p;
        if (pathRcVertexIndex != FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND && depth + 1> pathRcVertexIndex)
            suffixThroughput /= p;
        if (pdf_bsdf < EPSILON)
        {
            goto finalize_pixel;
        }

        throughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;
        if (pathRcVertexIndex != FLAG_CANDIDATE_GEN_RC_INDEX_UNFOUND && depth + 1 > pathRcVertexIndex)
            suffixThroughput *= f_val_bsdf * fabsf(outgoing.z) / pdf_bsdf;

        toWorld(outgoing, normal, outgoing);

        r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;
        r.direction = outgoing;

        prevDelta = currDelta;
        lastPDF = pdf_bsdf;
        lastCosine = fabsf(dot(outgoing, normal));
        lastPOS_GETRIDOFME = shadingPos;
    }

finalize_pixel:
    if (w_sum <= 0.0f) {
        restir.reservoir.setW(pixelIdx, 1.0f);
        restir.reservoir.setCachedJacobian(pixelIdx, -1.0f);
        restir.reservoir.pathFlags[pixelIdx] = packPathFlags(1, 0, 0, 0);
        return;
    }

    float p_hat = targetFunction(fromRGB9E5(F));
    float W = (p_hat > EPSILON) ? (w_sum / p_hat) : 0.0f;

    restir.reservoir.saveReservoirFinal(
        pixelIdx,
        W,
        F,
        pathFlags,
        actualRcVertexGeometry,
        actualCachedJacobian,
        neepdf
    );
}

extern "C" __global__ void __raygen__restirTemporalReuse() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers



    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;

    if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
        printf("Fresh candidate gen reservoir: \n");
        printPixelData(restir.reservoir, restir.gbuffer, pixelIdx, params.frame_index);
        printf("History reservoir: \n");
        printPixelData(restir.lastFrameReservoir, restir.gbuffer, pixelIdx, params.frame_index);
    }
    half2 mv = restir.gbuffer.getMV(pixelIdx);
    int2 historyCoord = make_int2(-1, -1);
    uint32_t reorderHint = 0u;

    uint32_t mvBits = reinterpret_cast<const uint32_t&>(mv);
    if (mvBits != 0xFFFFFFFF) { // 0xFFFFFFFF = no reprojectable surface (env miss). Skip-shade pixels keep a real MV and reuse normally.
        if (isHistoryValid(allParams, make_int2(x, y), mv, historyCoord)) { // check primary movtion vec
            reorderHint = 0xFFFFFFFF;
        } else {
            mv = restir.gbuffer.getDualMV(pixelIdx);
            if (isHistoryValid(allParams, make_int2(x, y), mv, historyCoord)) { // check dual motion vec
                reorderHint = 0xFFFFFFFF;
            }
        }
    }

    // Optionally go one step beyond stream compaction, and sort by morton code
#if TEMPORAL_SER_SORT_MORTON_CODE == 1
    uint32_t cx = x >> 5;
    uint32_t cy = y >> 5;

    uint32_t spatial_hint = (expandBits(cx) | (expandBits(cy) << 1)) & 0x7Fu;
    if (reorderHint != 0u) {
        reorderHint = spatial_hint | 0x80u;
    }

    optixReorder(reorderHint, 8);
#else
    optixReorder(reorderHint, 1);
#endif

    if (reorderHint == 0u)
        return;

    uint32_t historyIdx = historyCoord.x + historyCoord.y * params.w;

    uint8_t dupe_val = __ldg(&restir.duplication_map[historyIdx]);
    float D = (float)dupe_val / 255.0f;

    float cCap = lerp(LERP_MCAP, 1.0f, powf(D, 0.1f));

    uint32_t hist_M_int;
    uint32_t hist_pathLength;
    uint32_t hist_rcVertexIndex;
    TechniqueType hist_type;
    restir.lastFrameReservoir.getPathFlags(historyIdx, hist_M_int, hist_pathLength, hist_rcVertexIndex, hist_type);

    float hist_M = fminf(cCap, hist_M_int);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Proceed to perform shift
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    uint32_t hist_rcPrimID;
    float2 hist_rcBarycentrics;
    float3 hist_rcWi;
    float3 hist_rcRadiance;

    restir.lastFrameReservoir.getRcVertexGeometry_globalLoad(historyIdx, hist_rcPrimID, hist_rcBarycentrics, hist_rcWi, hist_rcRadiance);

    float hist_cachedJacobianDenom = restir.lastFrameReservoir.getCachedJacobian_globalLoad(historyIdx);
    float hist_cachedNeePdf = -1.0f;

    if (needNeePDF(hist_type)) {
        hist_cachedNeePdf = restir.lastFrameReservoir.getCachedNEE_globalLoad(historyIdx);
    }

    uint32_t hist_seed = restir.lastFrameReservoir.getSeed_notstreaming(historyIdx);

    if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
        printf("frame %u at the start of temporal seed: %u\n", params.frame_index, hist_seed);
    }

    // ==============================================================================
    // 1. UNPACK CURRENT PATH DATA (Needed for the Backward Shift)
    // ==============================================================================
    uint32_t curr_pathFlags = restir.reservoir.pathFlags[pixelIdx];
    uint32_t curr_M = extractM(curr_pathFlags);
    uint32_t curr_pathLength = (curr_pathFlags >> 8) & 0xFF;
    uint32_t curr_rcVertexIndex = (curr_pathFlags >> 16) & 0xFF;
    TechniqueType curr_type = static_cast<TechniqueType>((curr_pathFlags >> 24) & 0xFF);

    float3 curr_F = restir.reservoir.getF_globalLoad(pixelIdx);
    float curr_W = restir.reservoir.getW_globalLoad(pixelIdx);
    float curr_p_hat = targetFunction(curr_F);

    uint32_t curr_seed = restir.reservoir.getSeed_notstreaming(pixelIdx);

    uint32_t curr_rcPrimID;
    float2 curr_rcBarycentrics;
    float3 curr_rcWi;
    float3 curr_rcRadiance;
    restir.reservoir.getRcVertexGeometry_globalLoad(pixelIdx, curr_rcPrimID, curr_rcBarycentrics, curr_rcWi, curr_rcRadiance);

    float curr_cachedJacobianDenom = restir.reservoir.getCachedJacobian_globalLoad(pixelIdx);
    float curr_cachedNeePdf = -1.0f;
    if (needNeePDF(curr_type)) {
        curr_cachedNeePdf = restir.reservoir.getCachedNEE_globalLoad(pixelIdx);
    }

    uint32_t new_M = curr_M + (uint32_t)hist_M;
    float hist_W = restir.lastFrameReservoir.getW_globalLoad(historyIdx);
    float3 hist_F = restir.lastFrameReservoir.getF_globalLoad(historyIdx);
    float hist_p_hat = targetFunction(hist_F);

    // ==============================================================================
    // 2. THE FORWARD SHIFT (History Path -> Current Pixel)
    // ==============================================================================
    ShiftResult fwdResult;
    if (hist_M_int > 0 && hist_cachedJacobianDenom != -1.0f) {
        fwdResult = evaluateHybridShift<false>(
            allParams,
            x, y,
            hist_seed, hist_pathLength, hist_rcVertexIndex, hist_type,
            hist_rcPrimID, hist_rcBarycentrics, hist_rcWi, hist_rcRadiance,
            hist_cachedNeePdf, hist_cachedJacobianDenom
        );
    } else {
        fwdResult = {false, f3(0), 0.0f, 0.0f};
    }


    // ==============================================================================
    // 3. THE BACKWARD SHIFT (Current Path -> History Pixel)
    // ==============================================================================
    ShiftResult bwdResult; // TODO, replace this (and evaluate hybrid shift) with a reverseresult and evaluatereverseshift, to save registers
    bool needs_bwd_shift = (curr_M > 0) &&
                           (curr_cachedJacobianDenom != -1.0f);

    if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y && !needs_bwd_shift) {
        printf("Backwards shift judged to not be needed.\n");
    }
    //optixReorder(needs_bwd_shift, 1);

    if (needs_bwd_shift) {
        // just for now, we want to print out everything.
        bwdResult = evaluateHybridShift<false>(
            allParams,
            historyCoord.x, historyCoord.y, // Backward shift originates from the history pixel
            curr_seed, curr_pathLength, curr_rcVertexIndex, curr_type,
            curr_rcPrimID, curr_rcBarycentrics, curr_rcWi, curr_rcRadiance,
            curr_cachedNeePdf, curr_cachedJacobianDenom
        );
    } else {
        bwdResult = {false, f3(0), 0.0f, 0.0f};
    }


    // ==============================================================================
    // 4. UNBIASED MIS WEIGHT CALCULATIONS (Factored to avoid NaN)
    // ==============================================================================
    float w_tentative = 0.0f;
    float mis_weight_curr = 1.0f; // Safely defaults to 1.0 if backward shift fails

    float fwd_phat = targetFunction(fwdResult.contribution);
    // A. Evaluate History Path MIS (Evaluated at Y_h = fwdResult)
    if (fwdResult.isValid) {
        // denom = M_c * p_c(Y_h) * J_{h->c} + M_h * p_h(X_h)
        float denom_hist = (curr_M * fwd_phat * fwdResult.jacobian) + (hist_M * hist_p_hat);
        if (denom_hist > 0.0f) {
            float mis_weight_hist = (hist_M * hist_p_hat) / denom_hist;
            w_tentative = mis_weight_hist * fwd_phat * hist_W * fwdResult.jacobian;
        }
    }
    if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y)
        printf("forward shift resulted in a jacobian of %f\n forward shfit produced an F of: <%f, %f, %f>, new Jacobian Denom: %f\n", fwdResult.jacobian, fwdResult.contribution.x, fwdResult.contribution.y, fwdResult.contribution.z, fwdResult.new_cached_jacobian);

    float bwd_phat = targetFunction(bwdResult.contribution);
    // B. Evaluate Current Path MIS (Evaluated at X_c)
    if (bwdResult.isValid) {
        // denom = M_c * p_c(X_c) + M_h * p_h(X_{c->h}) * J_{c->h}
        float denom_curr = (curr_M * curr_p_hat) + (hist_M * bwd_phat * bwdResult.jacobian);
        if (denom_curr > 0.0f) {
            mis_weight_curr = (curr_M * curr_p_hat) / denom_curr;
        }
    }

    if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y && needs_bwd_shift) {
        printf("backwards shift resulted in a jacobian of %f\n backwards shfit produced an F of: <%f, %f, %f>, new Jacobian Denom: %f\n", bwdResult.jacobian, bwdResult.contribution.x, bwdResult.contribution.y, bwdResult.contribution.z, bwdResult.new_cached_jacobian);

    }


    float w_curr_weighted = mis_weight_curr * curr_W * curr_p_hat;


    // ==============================================================================
    // 5. UNIFIED RESERVOIR UPDATE
    // ==============================================================================
    float w_sum = w_curr_weighted + w_tentative;
    RNGState refreshedLocalState = load_rng(hash_uint32(pixelIdx), hash_uint32(params.frame_index), hash_uint32(0), nullptr);

    bool history_won = false;
    if (w_sum > 0.0f && rand(&refreshedLocalState) < (w_tentative / w_sum)) {
        history_won = true;
    }

    if (history_won) {
        float W_final = (fwd_phat > 0.0f) ? (w_sum / fwd_phat) : 0.0f;
        if (isnan(W_final) || isinf(W_final)) W_final = 0.0f;

        restir.reservoir.saveReservoirAll(
            pixelIdx,
            W_final,
            fwdResult.contribution,
            hist_seed,
            new_M,
            hist_pathLength,
            hist_rcVertexIndex,
            hist_type,
            hist_rcPrimID,
            hist_rcBarycentrics,
            hist_rcWi,
            hist_rcRadiance,
            fwdResult.new_cached_jacobian,
            hist_cachedNeePdf
        );
    } else {
        float W_final = (curr_p_hat > 0.0f) ? (w_sum / curr_p_hat) : 0.0f;
        if (isnan(W_final) || isinf(W_final)) W_final = 0.0f;

        if (curr_M == 0 || W_final == 0.0f) {
            restir.reservoir.setPathFlags(pixelIdx, packPathFlags(1, 0, 0, 0));
            restir.reservoir.setW_noCS(pixelIdx, 0.0f);
        } else {
            curr_pathFlags = updateM(curr_pathFlags, new_M);
            restir.reservoir.setPathFlags(pixelIdx, curr_pathFlags);
            restir.reservoir.setW_noCS(pixelIdx, W_final);
        }
    }
}

/**
 *
 */
extern "C" __global__ void __raygen__restirSpatialReuse() {
    const CommonParams& params = allParams.common; // gets compiled out, so not taking up registers
    const RestirCommonParams& restir = allParams.restir; // gets compiled out, so not taking up registers

    uint3 launch_index = optixGetLaunchIndex();

    uint32_t x = launch_index.x;
    uint32_t y = launch_index.y;
    int pixelIdx = y*params.w + x;

    int2 neighborCoord = get_paired_neighbor(
        make_int2(x, y),
        restir.currentSpatialReuseIndex,
        params.frame_index,
        restir.reuseTextureSizes[restir.currentSpatialReuseIndex],
        make_int2(params.w, params.h),
        restir.reuseTextures[restir.currentSpatialReuseIndex]
    );
    uint32_t reorderHint = 0u;

    half2 mv = restir.gbuffer.getMV(pixelIdx); // just because this is a flag for ignoring

    if (reinterpret_cast<const uint32_t&>(mv) != 0xFFFFFFFF) { // check whether it was marked as ignore
        if ((neighborCoord.x != -1) && isSpatialNeighborValid(allParams, make_int2(x, y), neighborCoord)) { // check primary movtion vec
            reorderHint = 0xFFFFFFFF;
        }
    }

    // Optionally go one step beyond stream compaction, and sort by morton code
#if TEMPORAL_SER_SORT_MORTON_CODE == 1
    uint32_t cx = neighborCoord.x >> 5;
    uint32_t cy = neighborCoord.y >> 5;

    uint32_t spatial_hint = (expandBits(cx) | (expandBits(cy) << 1)) & 0x7Fu;
    if (reorderHint != 0u) {
        reorderHint = spatial_hint | 0x80u;
    }

    optixReorder(reorderHint, 8);
#else
    optixReorder(reorderHint, 1);
#endif

    if (reorderHint == 0u) {
        restir.shiftResultBuffer.setResult(pixelIdx, false, f3(), 0.0f, 0.0f);
        return;
    }

    uint32_t neighborIdx = neighborCoord.x + neighborCoord.y * params.w;

    uint32_t neighbor_M;
    uint32_t neighbor_pathLength;
    uint32_t neighbor_rcVertexIndex;
    TechniqueType neighbor_type;
    restir.reservoir.getPathFlags(neighborIdx, neighbor_M, neighbor_pathLength, neighbor_rcVertexIndex, neighbor_type);

    uint32_t neighbor_rcPrimID;
    float2 neighbor_rcBarycentrics;
    float3 neighbor_rcWi;
    float3 neighbor_rcRadiance;

    restir.reservoir.getRcVertexGeometry_globalLoad(neighborIdx, neighbor_rcPrimID, neighbor_rcBarycentrics, neighbor_rcWi, neighbor_rcRadiance);

    float neighbor_cachedJacobianDenom = restir.reservoir.getCachedJacobian_globalLoad(neighborIdx);
    float neighbor_cachedNeePdf = -1.0f;

    if (needNeePDF(neighbor_type)) {
        neighbor_cachedNeePdf = restir.reservoir.getCachedNEE_globalLoad(neighborIdx);
    }

    uint32_t neighbor_seed = restir.reservoir.getSeed_notstreaming(neighborIdx);

    ShiftResult fwdResult;
    if (neighbor_M > 0 && neighbor_cachedJacobianDenom != -1.0f) {
        fwdResult = evaluateHybridShift<false>(
            allParams,
            x, y,
            neighbor_seed, neighbor_pathLength, neighbor_rcVertexIndex, neighbor_type,
            neighbor_rcPrimID, neighbor_rcBarycentrics, neighbor_rcWi, neighbor_rcRadiance,
            neighbor_cachedNeePdf, neighbor_cachedJacobianDenom
        );
    } else {
        fwdResult = {false, f3(0), 0.0f, 0.0f};
    }

    restir.shiftResultBuffer.setResult(pixelIdx, fwdResult.isValid, fwdResult.contribution, fwdResult.jacobian, fwdResult.new_cached_jacobian);
}