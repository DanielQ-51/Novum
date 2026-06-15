#include <optix.h>
#include <optix_device.h>
#include "optixSetup.cuh"
#include "optixStructs.cuh"
#include "optixUtils.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "reflectors.cuh"

extern "C" {
    __constant__ PipelineParams params;
}

extern "C" __global__ void __closesthit__gather() {
    // 1. Grab the hardware attributes
    float2 uvs = optixGetTriangleBarycentrics();
    
    // 2. Pack the data into the 5 payload buckets
    optixSetPayload_0(1);                                   // p0: Hit flag (1 = true)
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));  // p1: Hit distance (t)
    optixSetPayload_2(__float_as_uint(uvs.x));              // p2: Barycentric U
    optixSetPayload_3(__float_as_uint(uvs.y));              // p3: Barycentric V
    optixSetPayload_4(optixGetPrimitiveIndex());            // p4: Primitive ID
}

extern "C" __global__ void __miss__gather() {
    // If the ray misses everything, set the hit flag to 0
    optixSetPayload_0(0);
}

extern "C" __global__ void __raygen__unidirectional() {
    uint3 launch_index = optixGetLaunchIndex();
    
    unsigned int x = launch_index.x;
    unsigned int y = launch_index.y;
    int pixelIdx = y*params.w + x;

    RNGState localState = load_rng(pixelIdx, params.frame_index, 0, nullptr);

    Ray r = params.camera.generateCameraRay(localState, x, y);

    float4 Li = f4(0.0f);
    float4 throughput = f4(1.0f);
    bool prevDelta = false;
    float lastPDF = 0.0f;
    for (int depth = 0; depth < params.max_depth; depth++)
    {
        SurfaceHit hitData = traceClosest(params, r);
        if (!hitData.isHit)
        {
            float4 contribution = throughput * params.shadeContext.lightSampler.envMap.sampleDir(r.direction);
            float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                params.shadeContext.lightSampler.evaluateEnvPdf(r.direction) // alternate strategy
            );
            Li += contribution * misWeight;
            break;
        }

        const Triangle& tri = params.shadeContext.scene[hitData.primId];
        int materialID = tri.materialID;
        float u = hitData.barycentrics.x;
        float v = hitData.barycentrics.y;

        float2 uv = __ldcs(&params.shadeContext.vertices->uvs[tri.uvaInd]) * (1.0f - u - v) + 
            __ldg(&params.shadeContext.vertices->uvs[tri.uvbInd]) * u + 
            __ldg(&params.shadeContext.vertices->uvs[tri.uvcInd]) * v;

        float4 apos = __ldg(&params.shadeContext.vertices->positions[tri.aInd]);
        float4 bpos = __ldg(&params.shadeContext.vertices->positions[tri.bInd]);
        float4 cpos = __ldg(&params.shadeContext.vertices->positions[tri.cInd]);

        float4 shadingPos = (1.0f - u - v) * apos + u * bpos + v * cpos;

        float4 a_n = __ldg(&params.shadeContext.vertices->normals[tri.naInd]);
        float4 b_n = __ldg(&params.shadeContext.vertices->normals[tri.nbInd]);
        float4 c_n = __ldg(&params.shadeContext.vertices->normals[tri.ncInd]);
        
        float4 normal = (1.0f - u - v) * a_n + u * b_n + v * c_n;
        bool backface = dot(normal, r.direction) > 0.0f;
        normal = backface ? -normal : normal;

        float4 contribution = backface ? f4(0.0f) : tri.emission * throughput;

        float4 incomingDir;
        toLocal(r.direction, normal, incomingDir);

        if (luminance(contribution) > EPSILON) {
            float misWeight = (prevDelta || depth == 0) ? 1.0f : powerHeuristicTwoStrategy(
                lastPDF, // primary strategy
                (hitData.t * hitData.t * params.shadeContext.lightSampler.evaluateMeshPdf(tri) / (fabsf(incomingDir.z))) // alternate strategy
            );

            Li += contribution * misWeight;
        }

        bool currDelta = params.shadeContext.materials[materialID].isSpecular;

        if (!currDelta) {
            float4 lightNormal;
            float4 emission;
            float4 shadingPosToLightNormalized;
            float t_max;
            float pdf;

            bool sampledEnv = params.shadeContext.lightSampler.sample(
                rand(&localState), rand4(&localState), 
                shadingPos, 
                params.shadeContext.vertices, 
                emission,
                shadingPosToLightNormalized, 
                lightNormal, 
                t_max, 
                pdf
            );

            float4 shadingPosToLightLocal;
            toLocal(shadingPosToLightNormalized, normal, shadingPosToLightLocal);

            bool surfaceBackface = dot(normal, shadingPosToLightNormalized) < 0.0f;
            bool lightBackface = (!sampledEnv) && (dot(lightNormal, -shadingPosToLightNormalized) < 0.0f);
            if (!surfaceBackface && !lightBackface) {
                float bsdfPDF;

                pdf_eval(
                    params.shadeContext.materials, 
                    materialID, 
                    params.shadeContext.textures, 
                    incomingDir,
                    shadingPosToLightLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    bsdfPDF,
                    uv
                );

                float4 f_val;
                f_eval(
                    params.shadeContext.materials, 
                    materialID, 
                    params.shadeContext.textures, 
                    incomingDir,
                    shadingPosToLightLocal,
                    1.5f, // change later when medium stack integrated
                    1.5f, // change later
                    f_val,
                    uv
                );

                float4 contribution;
                float misWeight;

                if (sampledEnv) {
                    float cosSurface = dot(normal, shadingPosToLightNormalized);
                    contribution = throughput * f_val * emission * cosSurface / pdf; 
                    misWeight = powerHeuristicTwoStrategy(
                        pdf,
                        bsdfPDF
                    );
                } else {
                    float cosLight = dot(-shadingPosToLightNormalized, lightNormal);
                    float cosSurface = dot(normal, shadingPosToLightNormalized);

                    contribution = throughput * // throughput
                        f_val * emission * cosLight * // NEE contribution
                        cosSurface / (pdf * t_max * t_max); // "pdf" here is the raw flux over total flux area pdf

                    misWeight = powerHeuristicTwoStrategy(
                        (t_max * t_max) * pdf / cosLight, // convert area pdf to SA
                        bsdfPDF // alt strat
                    );
                }

                bool occluded = traceVisibility(
                    params, 
                    Ray((shadingPos + shadingPosToLightNormalized * RAY_EPSILON), shadingPosToLightNormalized),
                    t_max * (1.0f - EPSILON3)
                );

                if (!occluded) {
                    Li += contribution * misWeight;
                }  
            }
        }
        
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);

        if (rand(&localState) > p)   // survive with probability p
        {
            save_rng(pixelIdx, &localState, nullptr);
            break;
        }
        throughput /= p; 

        
        float4 outgoing;
        float4 f_val;
        float pdf;

        sample_f_eval(
            localState, 
            params.shadeContext.materials, 
            materialID, 
            params.shadeContext.textures, 
            incomingDir,
            1.5f, // change later when medium stack integrated
            1.5f, // change later
            backface,
            outgoing,
            f_val,
            pdf,
            uv,
            TRANSPORTMODE_RADIANCE
        );

        if (pdf < EPSILON)
        {
            save_rng(pixelIdx, &localState, nullptr);
            break;
        }
        
        throughput *= f_val * fabsf(outgoing.z) / pdf;
        toWorld(outgoing, normal, outgoing);

        // write to next rayQueue
        r.origin = shadingPos + (dot(outgoing, normal) > 0.0f ? normal : -normal) * RAY_EPSILON;
        r.direction = outgoing;

        prevDelta = currDelta;
        lastPDF = pdf;
        save_rng(pixelIdx, &localState, nullptr);
    }
    params.accum_buffer[pixelIdx] += Li;
}