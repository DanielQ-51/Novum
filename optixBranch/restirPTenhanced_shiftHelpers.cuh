#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "sceneContexts.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "optixStructs.cuh"
#include "settings.cuh"
#include "optixUtils.cuh"
#include "reflectors.cuh"


/** Overall RR and bsdf scaling logic is split between two cases:
 *
 * If NEE, then it must be that x_k-1 rolled a RR and performed throughput scaling,
 * then hit x_k, spawned the NEE sample for this reservoir without rolling RR.
 *
 * If BSDF, then it must be that x_k-1 rolled a RR and performed throughput scaling,
 * then hit x_k, which rolled a RR and performed throughput scaling, then bounced and hit the light.
 *
 * We assume that no RR has been done for x_k-1 or x_k for the throughput passed to this function
 *
 */
__device__ __forceinline__ inline ShiftResult perform_K_is_D_minus_1_reconnection(
    const CommonParams& params,
    uint32_t pathType,
    uint32_t x, uint32_t y,
    bool isReverseShift,

    uint32_t rc_xk_materialID,
    float2 rc_xk_uv,
    float3 rc_xk_pos,
    bool rc_xk_backface,
    float3 rc_xk_normal,

    uint32_t xkminus1_materialID,
    float2 xkminus1_uv,
    float3 xkminus1_pos,
    bool xkminus1_backface,
    float3 xkminus1_normal,
    float3 xkminus1_inDirLocal,

    float3 throughput, // the throughput entering x_k-1

    float3 rcWi,
    float pdf_sampledLight_nee_sa, // Candidate generation ensures that for these paths, this pdf is always in solid angle already
    float3 lightEmissionRaw,
    float jacobian_denominator
) {
    ShiftResult result;

    float3 xkminus1_to_xk_direction_normalized = normalize(rc_xk_pos - xkminus1_pos);
    float xkminus1_to_xk_distance = length(rc_xk_pos - xkminus1_pos);

    // Shading systems expect the normal to be facing opposite direction from the incoming direction
    if (dot(xkminus1_to_xk_direction_normalized, rc_xk_normal) > 0.0f) {
        rc_xk_normal = -rc_xk_normal;
    }

    bool occluded = traceVisibility(
        params,
        Ray(xkminus1_pos + (xkminus1_to_xk_direction_normalized * RAY_EPSILON), xkminus1_to_xk_direction_normalized),
        xkminus1_to_xk_distance * (1.0f - EPSILON2)
    );

    if (occluded) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: recon failed reconnection visibility test for k=d-1\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // First, scale RR for x_k-1. This can be done right away since we do RR before the throughput scaling at each vertex.
{
    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    throughput /= p; // assume the roll succeeded
}
    // Now, perform the throughput scaling from exiting x_k-1

    float3 xkminus1_outDirLocal = toLocal(xkminus1_to_xk_direction_normalized, xkminus1_normal);

    float3 xkminus1_f_val;
    float xkminus1_pdf;
    f_pdf_eval(
        params.shadeContext.materials,
        xkminus1_materialID,
        params.shadeContext.textures,
        xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
        xkminus1_outDirLocal, // direction to the rc vertex from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        xkminus1_f_val,
        xkminus1_pdf,
        xkminus1_uv
    );

    float xkminus1_outgoing_cosine = fabsf(xkminus1_outDirLocal.z);

    // Now throughput is updated to that of entering x_k/leaving x_k-1
    throughput *= xkminus1_f_val * xkminus1_outgoing_cosine / xkminus1_pdf;

    // Next, scale RR for x_k, but only if it was a bsdf sample This can be done now since we do RR before the throughput scaling at each vertex.

    if (is_bsdf(pathType)) {
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);
        throughput /= p; // assume the roll succeeded
    }

    // Now, perform throughput scaling for exiting x_k

    float3 rc_xk_f_val;
    float rc_xk_bsdf_pdf;

    // This engine has the incoming direciton pointing into the surface
    float3 rc_xk_inDirLocal = toLocal(xkminus1_to_xk_direction_normalized, rc_xk_normal);
    float3 rc_xk_outDirLocal = toLocal(rcWi, rc_xk_normal);
    float rc_xk_outgoing_cosine = fabsf(rc_xk_outDirLocal.z);

    f_pdf_eval(
        params.shadeContext.materials,
        rc_xk_materialID,
        params.shadeContext.textures,
        rc_xk_inDirLocal, // direction to the y_k-1 from y_k-2
        rc_xk_outDirLocal, // direction to the rc vertex from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        rc_xk_f_val,
        rc_xk_bsdf_pdf,
        rc_xk_uv
    );

    float rc_xk_p_sampled_light = is_nee(pathType) ? (pdf_sampledLight_nee_sa) : (rc_xk_bsdf_pdf);

    throughput *= rc_xk_f_val * rc_xk_outgoing_cosine / rc_xk_p_sampled_light;

    float path_misWeight = powerHeuristicTwoStrategy(
        (is_nee(pathType)) ? pdf_sampledLight_nee_sa : rc_xk_bsdf_pdf,
        (is_nee(pathType)) ? rc_xk_bsdf_pdf : pdf_sampledLight_nee_sa
    );

    result.contribution = throughput * lightEmissionRaw * path_misWeight;

    // Now, calculate jacobian.

    float rc_xk_incoming_cosine = fabsf(rc_xk_inDirLocal.z);
    float geometryTerm = rc_xk_incoming_cosine / (xkminus1_to_xk_distance * xkminus1_to_xk_distance);
    float jacobian_numerator = is_bsdf(pathType) ?
        (xkminus1_pdf * geometryTerm * rc_xk_bsdf_pdf) : // if it was the bsdf case, it should be the normal jacobian
        (xkminus1_pdf * geometryTerm) // if it was a NEE case, the last pdf cancels out since the light selection pdf is equal for both domains paths
    ;

    result.jacobian = jacobian_numerator / jacobian_denominator;
    result.new_cached_jacobian = jacobian_numerator;
    result.isValid = true;

    // Safety Checks
    if (jacobian_numerator <= 0.0f || jacobian_denominator <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k=d-1 zero p_new_suffix or jacobianDenom\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    if (xkminus1_pdf <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k=d-1 xkminus1_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (rc_xk_bsdf_pdf <= 0.0f && !(is_nee(pathType))) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k=d-1 rc_xk_bsdf_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (targetFunction(result.contribution) <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k=d-1 phat zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    return result;
}


/**
 * The most straightforward reconneciton, since the reconnection is agnostic to the suffix geometry, and can easily just
 * process a clean incoming radiance at the rc vertex.
 *
 * We assume that no RR has been done for x_k-1 or x_k for the throughput passed to this function
 */
__device__ __forceinline__ inline ShiftResult perform_K_less_than_D_minus_1_reconnection(
    const CommonParams& params,
    uint32_t pathType,
    uint32_t x, uint32_t y,
    bool isReverseShift,

    uint32_t rc_xk_materialID,
    float2 rc_xk_uv,
    float3 rc_xk_pos,
    bool rc_xk_backface,
    float3 rc_xk_normal,

    uint32_t xkminus1_materialID,
    float2 xkminus1_uv,
    float3 xkminus1_pos,
    bool xkminus1_backface,
    float3 xkminus1_normal,
    float3 xkminus1_inDirLocal,

    float3 throughput, // the throughput entering x_k-1

    float3 rcWi,

    // As this is meant to be stitched onto the end of the prefix, this must only include RR and bsdf throughput scaling terms starting at x_k+1, since all that
    // for x_k is handled here in the shift. Contains the suffix throughput (including RR scaling), the emission, and the path mis term
    float3 rcRadiance,

    float jacobian_denominator
) {
    ShiftResult result;

    float3 xkminus1_to_xk_direction_normalized = normalize(rc_xk_pos - xkminus1_pos);
    float xkminus1_to_xk_distance = length(rc_xk_pos - xkminus1_pos);

    // Shading systems expect the normal to be facing opposite direction from the incoming direction
    if (dot(xkminus1_to_xk_direction_normalized, rc_xk_normal) > 0.0f) {
        rc_xk_normal = -rc_xk_normal;
    }

    bool occluded = traceVisibility(
        params,
        Ray(xkminus1_pos + (xkminus1_to_xk_direction_normalized * RAY_EPSILON), xkminus1_to_xk_direction_normalized),
        xkminus1_to_xk_distance * (1.0f - EPSILON2)
    );

    if (occluded) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: recon failed reconnection visibility test for k<d-1\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // First, scale RR for x_k-1. This can be done right away since we do RR before the throughput scaling at each vertex.
{
    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    throughput /= p; // assume the roll succeeded
}
    // Now, perform the throughput scaling from exiting x_k-1

    float3 xkminus1_outDirLocal = toLocal(xkminus1_to_xk_direction_normalized, xkminus1_normal);

    float3 xkminus1_f_val;
    float xkminus1_pdf;
    f_pdf_eval(
        params.shadeContext.materials,
        xkminus1_materialID,
        params.shadeContext.textures,
        xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
        xkminus1_outDirLocal, // direction to the rc vertex from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        xkminus1_f_val,
        xkminus1_pdf,
        xkminus1_uv
    );

    float xkminus1_outgoing_cosine = fabsf(xkminus1_outDirLocal.z);

    // Now throughput is updated to that of entering x_k/leaving x_k-1
    throughput *= xkminus1_f_val * xkminus1_outgoing_cosine / xkminus1_pdf;

    // Next, scale RR for x_k
{
    float lum = luminance(throughput);
    float p = clamp(lum, 0.05f, 1.0f);
    throughput /= p; // assume the roll succeeded
}
    // Now, perform throughput scaling for exiting x_k

    float3 rc_xk_f_val;
    float rc_xk_bsdf_pdf;

    // This engine has the incoming direciton pointing into the surface
    float3 rc_xk_inDirLocal = toLocal(xkminus1_to_xk_direction_normalized, rc_xk_normal);
    float3 rc_xk_outDirLocal = toLocal(rcWi, rc_xk_normal);
    float rc_xk_outgoing_cosine = fabsf(rc_xk_outDirLocal.z);

    f_pdf_eval(
        params.shadeContext.materials,
        rc_xk_materialID,
        params.shadeContext.textures,
        rc_xk_inDirLocal, // direction to the y_k-1 from y_k-2
        rc_xk_outDirLocal, // direction to the rc vertex from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        rc_xk_f_val,
        rc_xk_bsdf_pdf,
        rc_xk_uv
    );

    throughput *= rc_xk_f_val * rc_xk_outgoing_cosine / rc_xk_bsdf_pdf;

    result.contribution = throughput * rcRadiance;

    // Now, calculate jacobian.

    float rc_xk_incoming_cosine = fabsf(rc_xk_inDirLocal.z);
    float geometryTerm = rc_xk_incoming_cosine / (xkminus1_to_xk_distance * xkminus1_to_xk_distance);
    float jacobian_numerator = (xkminus1_pdf * geometryTerm * rc_xk_bsdf_pdf); // if it was the bsdf case, it should be the normal jacobian

    result.jacobian = jacobian_numerator / jacobian_denominator;
    result.new_cached_jacobian = jacobian_numerator;

    result.isValid = true;

    // Safety Checks
    if (jacobian_numerator <= 0.0f || jacobian_denominator <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k<d-1 zero p_new_suffix or jacobianDenom\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    if (xkminus1_pdf <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k<d-1 xkminus1_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (rc_xk_bsdf_pdf <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k<d-1 rc_xk_bsdf_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }
    if (targetFunction(result.contribution) <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: internal recon k<d-1 phat zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    return result;
}

__device__ __forceinline__ inline ShiftResult perform_K_is_D_reconnection(
    const CommonParams& params,
    uint32_t pathType,
    uint32_t x, uint32_t y,
    bool isReverseShift,

    uint32_t rc_xk_materialID,
    float2 rc_xk_uv,
    float3 rc_xk_pos,
    bool rc_xk_backface,
    float3 rc_xk_normal,

    uint32_t xkminus1_materialID,
    float2 xkminus1_uv,
    float3 xkminus1_pos,
    bool xkminus1_backface,
    float3 xkminus1_normal,
    float3 xkminus1_inDirLocal,

    float3 throughput, // the throughput entering x_k-1

    float3 xkminus1_to_xk_direction_normalized, // only used for environment cases
    float pdf_sampledLight_nee, // Either in solid angle or area depending on whether its env or area light
    float3 lightEmissionRaw,

    float jacobian_denominator
) {
    ShiftResult result;

    float xkminus1_to_xk_distance;

    if (is_area(pathType)) { // if its an area light (triangle)
        xkminus1_to_xk_direction_normalized = normalize(rc_xk_pos - xkminus1_pos);
        xkminus1_to_xk_distance = length(rc_xk_pos - xkminus1_pos);
    } else {
        // xkminus1_to_xk_direction_normalized is already defined, and passed in
        xkminus1_to_xk_distance = 1e30;
    }

    if (is_area(pathType)) { // Normal isnt defined for environment ending vertex
        // Shading systems expect the normal to be facing opposite direction from the incoming direction
        if (dot(xkminus1_to_xk_direction_normalized, rc_xk_normal) > 0.0f) {
            rc_xk_normal = -rc_xk_normal;
        }
    }

    bool occluded = traceVisibility(
        params,
        Ray(xkminus1_pos + (xkminus1_to_xk_direction_normalized * RAY_EPSILON), xkminus1_to_xk_direction_normalized),
        xkminus1_to_xk_distance * (1.0f - EPSILON2)
    );

    if (occluded) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: recon failed reconnection visibility test for k=d\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // Now, this should make sure that pdf_sampled_Light_nee is in solid angle
    if (is_area(pathType)) {
        float xk_lightCosine = fabsf(dot(rc_xk_normal, xkminus1_to_xk_direction_normalized));
        pdf_sampledLight_nee *= (xkminus1_to_xk_distance * xkminus1_to_xk_distance) / xk_lightCosine;
    }

    // If the path was generated by bsdf, then it must be that RR was rolled at x_k-1, before then bouncing to x_k.
    // On the contrary, if it was nee, then nee was cast from x_k-1, before any RR was rolled for x_k-1, so it isnt applied
    if (is_bsdf(pathType)) {
        float lum = luminance(throughput);
        float p = clamp(lum, 0.05f, 1.0f);
        throughput /= p; // assume the roll succeeded
    }

    // Now, perform the throughput scaling from exiting x_k-1

    float3 xkminus1_outDirLocal = toLocal(xkminus1_to_xk_direction_normalized, xkminus1_normal);

    float3 xkminus1_f_val;
    float xkminus1_bsdf_pdf;
    f_pdf_eval(
        params.shadeContext.materials,
        xkminus1_materialID,
        params.shadeContext.textures,
        xkminus1_inDirLocal, // direction to the y_k-1 from y_k-2
        xkminus1_outDirLocal, // direction to the rc vertex aka light from y_k-1
        1.5f, // change later when medium stack integrated
        1.5f, // change later
        xkminus1_f_val,
        xkminus1_bsdf_pdf,
        xkminus1_uv
    );

    float xkminus1_outgoing_cosine = fabsf(xkminus1_outDirLocal.z);

    float p_sampled_light = is_bsdf(pathType) ? xkminus1_bsdf_pdf : pdf_sampledLight_nee;

    // Now throughput is updated to that of entering x_k/leaving x_k-1
    throughput *= xkminus1_f_val * xkminus1_outgoing_cosine / p_sampled_light;

    float path_misWeight = powerHeuristicTwoStrategy(
        (is_nee(pathType)) ? pdf_sampledLight_nee : xkminus1_bsdf_pdf,
        (is_nee(pathType)) ? xkminus1_bsdf_pdf : pdf_sampledLight_nee
    );

    result.contribution = throughput * lightEmissionRaw * path_misWeight;

    // For only this case, we execute a bsdf direction copy, which reduces variance but has a
    // non-identity jacobian
    if (is_bsdf(pathType) && is_env(pathType)) {
        float jacobian_numerator = xkminus1_bsdf_pdf;
        result.jacobian = jacobian_numerator / jacobian_denominator;
        result.new_cached_jacobian = jacobian_numerator;
    } else {
        result.jacobian = 1.0f; // a full random replay path
        result.new_cached_jacobian = 1.0f;
    }

    result.isValid = true;

    // Add safety checks at the end like the other funcions

    if (p_sampled_light <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: recon k=d p_sampled_light zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // 2. Ensure the BSDF at y_{k-1} can actually scatter towards the light.
    // Even if this is an NEE path, if the BSDF PDF is zero, the connection is physically impossible.
    if (xkminus1_bsdf_pdf <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: recon k=d xkminus1_bsdf_pdf zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // 3. Ensure we aren't dividing by zero when calculating the Jacobian.
    // This is only applicable for the BSDF Environment hit, since the other paths hardcode jacobian = 1.0f.
    if (is_bsdf(pathType) && is_env(pathType) && jacobian_denominator <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: recon k=d env bsdf zero jacobian_denominator\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    // 4. Ensure the final target function evaluates to a valid, non-zero weight.
    if (targetFunction(result.contribution) <= 0.0f) {
        if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
            printf("SHIFT ABORT [%s]: recon k=d phat zero\n", isReverseShift ? "REVERSE" : "FORWARD");
        }
        return {false, f3(0), 0.0f, 0.0f};
    }

    return result;
}