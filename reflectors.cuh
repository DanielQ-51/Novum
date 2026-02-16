#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "util.cuh"
#include "objects.cuh"
#include <curand_kernel.h>

__device__ void cosine_f(const float4& baseColor, float4& newColor)
{
    newColor = baseColor/PI;
}

__device__ void cosine_pdf(const float4& wo_local, float& pdf)
{
    pdf = fmaxf(wo_local.z, EPSILON)/PI;
}


__device__ void cosine_sample_f(curandState& localState, const float4& baseColor, float4& wo, float4& f_val, float& pdf)
{
    float u1 = curand_uniform(&localState);

    u1 = fminf(u1, 1.0f-EPSILON);
    float u2 = curand_uniform(&localState); 

    float r = sqrt(u1);
    float phi = 2.0f * PI * u2; 

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(1.0f - u1); 

    wo = f4(x,y,z);

    cosine_f(baseColor, f_val);
    cosine_pdf(wo, pdf);
}

__device__ void cosine_emit(curandState& localState, float4& wo, float& pdf)
{
    float u1 = curand_uniform(&localState);

    u1 = fminf(u1, 1.0f-EPSILON);
    float u2 = curand_uniform(&localState); 

    float r = sqrt(u1);
    float phi = 2.0f * PI * u2; 

    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(1.0f - u1); 

    wo = f4(x,y,z);
    cosine_pdf(wo, pdf);
}

__device__ void mirror_f(float4& f_val, float4 wo)
{
    float cos_theta = fmaxf(wo.z, EPSILON);
    f_val = f4(1.0f / cos_theta);
}

__device__ void mirror_pdf(float& pdf)
{
    pdf = 1.0f;
}

__device__ void mirror_sample_f(float4 wi, float4& wo, float4& f_val, float& pdf)
{
    wo = f4(-wi.x, -wi.y, wi.z);
    float cos_theta = fmaxf(wo.z, EPSILON);
    f_val = f4(1.0f / cos_theta);
    pdf = 1.0f;
}

__device__ float D_GGX(const float4& h, float alpha) 
{
    float cosThetaH = h.z;
    float alpha2 = alpha * alpha;
    float denom = cosThetaH*cosThetaH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denom * denom);
}

__device__ float G1_Smith(float nDotV, float alpha) 
{
    float a = alpha * sqrtf(1.0f - nDotV*nDotV) / nDotV;
    return 2.0f / (1.0f + sqrtf(1.0f + a*a));
}

__device__ float G1_GGX(const float4& v, const float4& h, float alpha)
{
    float cosTheta = v.z;
    float tanTheta = sqrtf(1.0f - cosTheta * cosTheta) / cosTheta;
    float a = 1.0f / (alpha * tanTheta);
    if (a < 1.6f)
        return (3.535f * a + 2.181f * a * a) / (1.0f + 2.276f * a + 2.577f * a * a);
    else
        return 1.0f;
}

__device__ float G_Smith(const float4& wi, const float4& wo, const float4& h, float alpha)
{
    return G1_GGX(wi, h, alpha) * G1_GGX(wo, h, alpha);
}

__device__ float4 Fresnel_Conductor(float cosTheta, const float4& eta, const float4& k) {
    float4 cosTheta2 = f4(cosTheta*cosTheta);
    float4 sinTheta2 = f4(1.0f) - cosTheta2;
    
    float4 eta2 = eta * eta;
    float4 k2 = k * k;
    
    float4 t0 = eta2 - k2 - sinTheta2;
    float4 a2plusb2 = sqrtf4(t0*t0 + 4.0f * eta2 * k2);
    float4 t1 = a2plusb2 + cosTheta2;
    float4 a = sqrtf4(0.5f * (a2plusb2 + t0));
    float4 t2 = 2.0f * cosTheta * a;
    
    float4 Rs = (t1 - t2) / (t1 + t2);
    float4 t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
    float4 t4 = t2 * sinTheta2;
    float4 Rp = (t3 - t4) / (t3 + t4);
    //return (Rs + Rp) * 0.5f;
    return (t1 - t2) / (t1 + t2);
}

__device__ void microfacet_metal_f(const float4& eta, const float4& k, float roughness, const float4& wi, const float4& wo, float4& f_val)
{
    if (wi.z <= 0.0f || wo.z <= 0.0f) {
        f_val = f4(0.0f);
        return;
    }
    float nDotWi = wi.z;
    float nDotWo = wo.z;

    float4 h = normalize(wi + wo);

    if (h.z <= 0.0f) h = f4(-h.x, -h.y, -h.z); // flip so h.z > 0
    
    float alpha = roughness * roughness;

    float D = D_GGX(h, alpha);
    float G = G_Smith(wi, wo, h, alpha);

    float4 f = Fresnel_Conductor(dot(wi, h), eta, k);

    f_val = (D * G * f) / fmaxf(4.0f * nDotWi * nDotWo, EPSILON);
}

__device__ void microfacet_pdf(const float& roughness, const float4& wi, const float4& wo, float& pdf)
{
    float4 h = normalize(wi + wo);
    float D = D_GGX(h, roughness *roughness);
    float denom = 4.0f * dot(wo, h);
    pdf = (D * h.z) / (denom);
}

__device__ void microfacet_metal_sample_f(curandState& localState, const float4& eta, const float4& k, float roughness, const float4& wi, 
    float4& wo, float4& f_val, float& pdf)
{
    float u1 = curand_uniform(&localState);

    float alpha = roughness * roughness;
    float phi = 2.0f * PI * curand_uniform(&localState);
    float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha*alpha - 1.0f) * u1));
    float sinTheta = sqrtf(fmaxf(1.0f - cosTheta*cosTheta, 0.0f));

    float4 h_local;
    h_local.x = sinTheta * cosf(phi);
    h_local.y = sinTheta * sinf(phi);
    h_local.z = cosTheta; // points along local normal

    wo = 2.0f * dot(wi, h_local) * h_local - wi;
    if (wo.z <= 0.0f) wo.z = -wo.z;
    
    microfacet_metal_f(eta, k, roughness, wi, wo, f_val); // f_val set
    microfacet_pdf(roughness, wi, wo, pdf); // pdf set
}

// fabs is used here for costheta
__device__ inline float schlick_fresnel(float cosTheta, float etaI, float etaT)
{
    float R0 = (etaI - etaT) / (etaI + etaT);
    R0 = R0 * R0;
    return R0 + (1.0f - R0) * powf(1.0f - fabsf(cosTheta), 5.0f);
}

// -----------------------------------------------------------------------------
// EVAL (given wi, wo, etaI, etaT, and reflect boolean)
// -----------------------------------------------------------------------------
__device__ void smooth_dielectric_f(
    const float4& wi, const float4& wo,
    float etaI, float etaT, bool reflect, bool TIR,
    float4& f_val)
{
    printf("never called");
    float cosThetaI = wi.z;
    float cosThetaO = wo.z;
    float F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (reflect) {
        // Perfect specular reflection: wo.z = wi.z
        // BSDF value for delta reflection (handled as Dirac delta)
        if (TIR)
            f_val = f4(1.0f / fabsf(cosThetaO));
        else
            f_val = f4(F / fabsf(cosThetaO));
    } else {
        // Perfect specular refraction
        float eta = etaI / etaT;
        f_val = f4((1.0f - F) * eta * eta / fabsf(cosThetaO));
    }
}

// -----------------------------------------------------------------------------
// SAMPLE_F (importance sample the reflection/refract direction)
// -----------------------------------------------------------------------------
__device__ void smooth_dielectric_sample_f(curandState& localState,
    const float4& wi, float etaI, float etaT, float4& wo, float4& f_val, float& pdf)
{
    float cosThetaI = wi.z; // always pos

    float eta = etaI / etaT;
    float cosThetaT2 = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

    float F;
    if (cosThetaT2 < 0.0f)
    {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;

        if (fabsf(cosThetaO) < EPSILON) {
            pdf = 0.0f;
            f_val = f4(0.0f);
            return;
        }

        f_val = f4(1.0f / cosThetaO);
        pdf = 1.0f;
        return;
    }
    
    F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (curand_uniform(&localState) < F) {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        if (fabsf(cosThetaO) < EPSILON) {
            pdf = 0.0f;
            f_val = f4(0.0f);
            return;
        }
        pdf = F;
        f_val = f4(F / fabsf(cosThetaO));
    } 
    else {
        float eta = etaI / etaT;

        wo = f4(
            -eta * wi.x, 
            -eta * wi.y, 
            - (sqrtf(cosThetaT2))
        );
        wo = normalize(wo);

        float cosThetaO = wo.z;

        if (fabsf(cosThetaO) < EPSILON) {
            pdf = 0.0f;
            f_val = f4(0.0f);
            return;
        }
        // BSDF term
        f_val = f4((1.0f - F) * eta * eta / fabsf(cosThetaO));
        pdf = 1.0f - F;
    }
    
}

// -----------------------------------------------------------------------------
// PDF (for MIS weighting)
// -----------------------------------------------------------------------------
__device__ void smooth_dielectric_pdf(
    const float4& wi, const float4& wo,
    float etaI, float etaT, bool reflect, bool TIR,
    float& pdf)
{
    printf("never called");
    float cosThetaI = wi.z;
    float F = schlick_fresnel(cosThetaI, etaI, etaT);

    if (reflect) {
        if (TIR)
            pdf = 1.0f;
        else
            pdf = F;
    } else {
        pdf = 1.0f - F;
    }
}
// wi is always point away from the surface on the positive z hemisphere. since we flipped the intersect normal if it was a backface before converting to local
__device__ void dumb_smooth_dielectric_sample_f(curandState& localState,
    const float4& wi, float etaSurface, bool backface, int transportMode, float4& wo, float4& f_val, float& pdf)
{
    float etaI, etaT;
    if (backface)
    {
        etaI = etaSurface;
        etaT = 1.0f;
    }
    else
    {
        etaI = 1.0f;
        etaT = etaSurface;
    }

    float cosThetaI = fminf(fmaxf(wi.z, EPSILON), 1.0f);

    float eta = etaI / etaT;
    float cosThetaT2 = 1.0f - eta * eta * (1.0f - cosThetaI * cosThetaI);

    float F;
    
    F = schlick_fresnel(cosThetaI, etaI, etaT);
    
    if (cosThetaT2 < 0.0f || F >= 0.99999f)
    {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        f_val = f4(1.0f / fmaxf(cosThetaO, EPSILON));
        pdf = 1.0f;
        return;
    }
    
    if (curand_uniform(&localState) < F) {
        wo = f4(-wi.x, -wi.y, wi.z);
        float cosThetaO = wo.z;
        pdf = F;
        f_val = f4(F / fmaxf(cosThetaO, EPSILON));
    } 
    else {
        float eta = etaI / etaT;

        wo = f4(
            -eta * wi.x, 
            -eta * wi.y, 
            - (sqrtf(cosThetaT2))
        );

        float cosThetaO = wo.z;

        // BSDF term
        float denom = fmaxf(fabsf(cosThetaO), EPSILON);
        f_val = f4((1.0f - F)/ denom);
        pdf = 1.0f - F;

        // adjoint bsdf handling
        if (transportMode == TRANSPORTMODE_IMPORTANCE)
        {
            // do nothing, throughput conserved
        }
        else if (transportMode == TRANSPORTMODE_RADIANCE)
        {
            f_val *= eta * eta;
        }
    }
}
__device__ void thin_dielectric_sample_f(curandState& localState,
    const float4& wi, float etaSurface, bool backface, int transportMode, float4& wo, float4& f_val, float& pdf)
{
    // 1. Setup IOR
    // A thin wall is effectively a sheet suspended in air (1.0).
    // The interaction "Air -> Material -> Air" is symmetric, so we don't swap 
    // IORs based on backface. We always calculate F for Air(1.0) hitting Surface(eta).
    float etaI = 1.0f;
    float etaT = etaSurface;

    float cosThetaI = fminf(fmaxf(wi.z, EPSILON), 1.0f);

    // 2. Calculate Single-Interface Fresnel
    float F_single = schlick_fresnel(cosThetaI, etaI, etaT);

    // 3. Adjust for Thin-Walled "Double Reflection"
    // F_thin accounts for the infinite series of internal bounces 
    // (front reflection + transmitted-then-back-reflected light).
    // Formula: R = 2F / (1 + F)
    float F = (2.0f * F_single) / (1.0f + F_single);

    // 4. Sample
    if (curand_uniform(&localState) < F) 
    {
        // --- REFLECTION ---
        // Standard reflection vector
        wo = f4(-wi.x, -wi.y, wi.z);
        
        float cosThetaO = wo.z; 
        
        pdf = F;
        
        // BSDF = F * (1 / |cosTheta|)
        f_val = f4(F / fmaxf(cosThetaO, EPSILON));
    } 
    else 
    {
        // --- TRANSMISSION ---
        // Pass straight through (wo = -wi).
        // No refraction bending.
        wo = f4(-wi.x, -wi.y, -wi.z);

        float cosThetaO = wo.z; 

        pdf = 1.0f - F;

        // BSDF = (1-F) * (1 / |cosTheta|)
        // Note: No 'eta * eta' scaling because the ray starts and ends 
        // in the same medium (Air), so radiance is conserved naturally.
        float denom = fmaxf(fabsf(cosThetaO), EPSILON);
        f_val = f4((1.0f - F) / denom);
    }
}

__device__ void sampleTexture(const Material& mat, float4* textures, const float2 uv, float4& albedo)
{
    int width = mat.width;
    int height = mat.height;

    // 1. Coordinate scaling
    // We do not floor u/v here yet to preserve precision for the fraction
    float fx = uv.x * width - 0.5f;
    float fy = uv.y * height - 0.5f;

    // 2. Determine grid corners (flooring)
    int x_int = static_cast<int>(floorf(fx));
    int y_int = static_cast<int>(floorf(fy));

    // 3. Calculate fractions (The weights)
    // We use the float coordinate minus the floor coordinate, BEFORE clamping/wrapping
    float sx = fx - floorf(fx);
    float sy = fy - floorf(fy);

    // 4. Handle Wrapping (Modulo Arithmetic) 
    // This ensures pixel (width) wraps to 0, and pixel (-1) wraps to (width-1)
    auto wrap = [](int val, int dim) {
        int r = val % dim;
        return r < 0 ? r + dim : r;
    };

    int x0 = wrap(x_int, width);
    int y0 = wrap(y_int, height);
    int x1 = wrap(x_int + 1, width);
    int y1 = wrap(y_int + 1, height);

    // 5. Fetch texels
    // Note: Depending on memory layout, these 4 reads are likely uncoalesced 
    // and will cause significant memory latency.
    float4 c00 = textures[mat.startInd + y0 * width + x0];
    float4 c10 = textures[mat.startInd + y0 * width + x1];
    float4 c01 = textures[mat.startInd + y1 * width + x0];
    float4 c11 = textures[mat.startInd + y1 * width + x1];

    // 6. Interpolate (Lerp)
    // Using mix/lerp helper functions is usually cleaner:
    // lerp(a, b, t) = a + t*(b-a)
    float4 bottom = c00 * (1.0f - sx) + c10 * sx;
    float4 top    = c01 * (1.0f - sx) + c11 * sx;
    
    albedo = bottom * (1.0f - sy) + top * sy;
}

// convention: wi always faces away from the surface (same dir as surface normal)
__device__ void leaf_f(const float4& albedo, float ior, float currIOR, float roughness, float transmission, const float4& wi, const float4& wo, float4& f_val)
{
    bool is_reflection = wo.z * wi.z > 0.0f;

    float4 h;
    float F;
    
    F = schlick_fresnel(wi.z, currIOR, ior); 

    if (is_reflection) // if it reflected with respect to the surface normal (add f_val from both events, reflection and diffuse)
    {
        h = normalize(wi + wo);
        float microfacet_F = schlick_fresnel(dot(wi, h), currIOR, ior);
        float nDotWi = wi.z;
        float nDotWo = wo.z;

        if (h.z <= 0.0f) h = -h; // flip so h.z > 0
        
        float alpha = roughness * roughness;

        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);

        float4 f_cuticle = f4(D * G * microfacet_F / fmaxf(4.0f * nDotWi * nDotWo, EPSILON));

        if (microfacet_F < 0.0f) printf("negative microfacetF");

        if (microfacet_F > 1.0f) printf("greater than 1 microfacetF %f: dot = %f, currIOR = %f, ior = %f\n", microfacet_F, dot(wi, h), currIOR, ior);

        float4 f_diffuse_val;
        cosine_f(albedo, f_diffuse_val);

        f_val = (1.0f-microfacet_F) * (1.0f - transmission) * f_diffuse_val + f_cuticle;
    }
    else
    {
        cosine_f(albedo, f_val);
        f_val *= transmission * (1.0f - F);
    }
}

__device__ void leaf_pdf(float ior, float currIOR, float roughness, float transmission, const float4& wi, const float4& wo, float& pdf)
{
    bool is_reflection = wo.z * wi.z > 0.0f;

    float4 h;
    float F;
    
    F = schlick_fresnel(abs(wi.z), currIOR, ior); 

    F = fminf(F, 1.0f - 0.1f * roughness);

    float p_specular = F;
    float p_diffuse_refl = (1.0f - F) * (1.0f - transmission);
    float p_diffuse_trans = (1.0f - F) * transmission;
    
    if (is_reflection) // if it reflected with respect to the surface normal
    {
        h = normalize(wi + wo);
        if (h.z < 0.0f) h = -h;
        
        //float nDotWi = wi.z;
        //float nDotWo = wo.z;
        float alpha = roughness * roughness;

        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);

        float denom = 4.0f * dot(wo, h);
        float pdf_cuticle_bounce = (D * h.z) / (denom);

        float pdf_diffuse;
        
        cosine_pdf(wo, pdf_diffuse);

        pdf = (p_specular * pdf_cuticle_bounce) + (p_diffuse_refl * pdf_diffuse);
    }
    else
    {
        float pdf_trans;
        cosine_pdf(-wo, pdf_trans);

        pdf = pdf_trans * p_diffuse_trans;
    }
}

__device__ void leaf_sample_f(curandState& localState, const float4& wi, float ior, float currIOR, float roughness, const float4& albedo, float transmission, float4& wo, float4& f_val, float& pdf)
{
    float F = schlick_fresnel(wi.z, currIOR, ior);

    if (curand_uniform(&localState) < F) // reflection on cuticle layer
    {
        float u1 = curand_uniform(&localState);

        float alpha = roughness * roughness;
        float phi = 2.0f * PI * curand_uniform(&localState);
        float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha*alpha - 1.0f) * u1));
        float sinTheta = sqrtf(fmaxf(1.0f - cosTheta*cosTheta, 0.0f));

        float4 h_local;
        h_local.x = sinTheta * cosf(phi);
        h_local.y = sinTheta * sinf(phi);
        h_local.z = cosTheta; // points along local normal

        wo = 2.0f * dot(wi, h_local) * h_local - wi;
    }
    else // transmit through cuticle
    {
        if (curand_uniform(&localState) < transmission) // go through leaf
        {
            cosine_sample_f(localState, albedo, wo, f_val, pdf);
            wo.z = -wo.z;
        }
        else // diffuse bounce off leaf
        {
            cosine_sample_f(localState, albedo, wo, f_val, pdf);
        }
    }

    leaf_f(albedo, ior, currIOR, roughness, transmission, wi, wo, f_val);
    leaf_pdf(ior, currIOR, roughness, transmission, wi, wo, pdf);
}

__device__ void microfacet_dielectric_f(
    float etaI, float etaT, float roughness, 
    const float4& wi, const float4& wo, float4& f_val)
{
    // Determine if this is reflection or transmission based on hemispheres
    bool is_reflection = wo.z * wi.z > 0.0f;
    
    float cosThetaWi = wi.z;
    float cosThetaWo = wo.z;
    float alpha = roughness * roughness;

    if (is_reflection) 
    {
        // Reflection: Half vector is standard half-angle
        float4 h = normalize(wi + wo);
        if (h.z <= 0.0f) h = -h; // Ensure h points into upper hemisphere

        // Calculate D, G, F
        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);
        
        // Fresnel term (using dot(wi, h))
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);

        // Microfacet Reflection Formula
        // f_r = (D * G * F) / (4 * cosThetaWi * cosThetaWo)
        float denom = 4.0f * fabsf(cosThetaWi * cosThetaWo);
        f_val = f4((D * G * F) / fmaxf(denom, EPSILON));
    }
    else 
    {
        // Transmission: Half vector depends on IOR
        // Walter et al. 2007 Eq. 16
        // h = - normalize(etaI * wi + etaT * wo)
        float4 h = normalize(etaI * wi + etaT * wo);
        if (h.z < 0.0f) h = -h; // Flip to align with normal

        float D = D_GGX(h, alpha);
        float G = G_Smith(wi, wo, h, alpha);
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);

        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);

        // Microfacet Transmission Formula
        // f_t = ( |wi.h| * |wo.h| * etaT^2 * D * G * (1-F) ) / ( |wi.n| * |wo.n| * (etaI * wi.h + etaT * wo.h)^2 )
        
        float sqrtDenom = etaI * wiDotH + etaT * woDotH;
        float denom = fmaxf(fabsf(cosThetaWi * cosThetaWo), EPSILON) * sqrtDenom * sqrtDenom;
        
        // Note: We include etaT^2 here as per standard derivation, 
        // but see sample_f for final adjoint/transport handling.
        float num = fabsf(wiDotH * woDotH) * (etaT * etaT) * D * G * (1.0f - F);
        
        f_val = f4(num / fmaxf(denom, EPSILON));
    }
}

__device__ void microfacet_dielectric_pdf(
    float etaI, float etaT, float roughness, 
    const float4& wi, const float4& wo, float& pdf)
{
    bool is_reflection = wo.z * wi.z > 0.0f;
    float alpha = roughness * roughness;

    if (is_reflection)
    {
        float4 h = normalize(wi + wo);
        if (h.z < 0.0f) h = -h;

        // Jacobian d_wh / d_wo = 1 / (4 * wo.h)
        float D = D_GGX(h, alpha);
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);
        
        // We weigh the PDF by F because we chose reflection with probability F
        float pdf_h = (D * h.z); 
        float denom = 4.0f * fabsf(dot(wo, h));
        
        pdf = F * (pdf_h / fmaxf(denom, EPSILON));
    }
    else
    {
        float4 h = normalize(etaI * wi + etaT * wo);
        if (h.z < 0.0f) h = -h;

        float D = D_GGX(h, alpha);
        float F = schlick_fresnel(dot(wi, h), etaI, etaT);
        float pdf_h = (D * h.z);

        // Jacobian for refraction
        // d_wh / d_wo = (etaT^2 * |wo.h|) / (etaI * wi.h + etaT * wo.h)^2
        float wiDotH = dot(wi, h);
        float woDotH = dot(wo, h);
        float sqrtDenom = etaI * wiDotH + etaT * woDotH;
        
        float jacobian = (etaT * etaT * fabsf(woDotH)) / fmaxf(sqrtDenom * sqrtDenom, EPSILON);

        // We weigh PDF by (1-F) because we chose transmission with probability (1-F)
        pdf = (1.0f - F) * pdf_h * jacobian;
    }
}

__device__ void microfacet_dielectric_sample_f(curandState& localState,
    const float4& wi, float etaSurface, float roughness, bool backface, int transportMode, float4& wo, float4& f_val, float& pdf)
{
    float etaI, etaT;
    if (backface)
    {
        etaI = etaSurface;
        etaT = 1.0f;
    }
    else
    {
        etaI = 1.0f;
        etaT = etaSurface;
    }

    // 1. Sample Microfacet Normal (h) using GGX
    float u1 = curand_uniform(&localState);
    float alpha = roughness * roughness;
    float phi = 2.0f * PI * curand_uniform(&localState);
    
    // Standard GGX sampling (matching leaf_sample_f implementation)
    float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1));
    float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));

    float4 h;
    h.x = sinTheta * cosf(phi);
    h.y = sinTheta * sinf(phi);
    h.z = cosTheta; 
    
    // Ensure h aligns with wi to avoid artifacts
    if (dot(wi, h) < 0.0f) h = -h;

    // 2. Fresnel and Selection
    float wiDotH = dot(wi, h);
    float F = schlick_fresnel(fabsf(wiDotH), etaI, etaT);

    if (curand_uniform(&localState) < F) 
    {
        // --- Reflection ---
        wo = 2.0f * wiDotH * h - wi;
    } 
    else 
    {
        // --- Transmission (Refraction) ---
        float eta = etaI / etaT;
        float c = dot(wi, h);
        float term = 1.0f - eta * eta * (1.0f - c * c);

        // Check for Total Internal Reflection (TIR)
        if (term < 0.0f) 
        {
            // TIR: Default to reflection (probability of TIR is handled by F going to 1.0 usually, 
            // but for safety in microfacet sampling, we reflect)
            wo = 2.0f * c * h - wi;
        }
        else
        {
            // Refract vector
            wo = (eta * c - sqrtf(term)) * h - eta * wi;
        }
    }

    // 3. Evaluate f and pdf
    microfacet_dielectric_f(etaI, etaT, roughness, wi, wo, f_val);
    microfacet_dielectric_pdf(etaI, etaT, roughness, wi, wo, pdf);

    // 4. Adjoint / Transport Mode Handling
    // Matching 'dumb_smooth_dielectric' convention:
    // If we transmitted (signs differ) and we are in Radiance mode, scale by eta^2.
    // (eta = etaI / etaT)
    if (wo.z * wi.z < 0.0f && transportMode == TRANSPORTMODE_RADIANCE)
    {
        float eta = etaI / etaT;
        f_val *= eta * eta;
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ void f_eval(const Material* materials, int materialID, float4* textures,
    const float4& wi, const float4& wo, float etaI, float etaT, float4& f_val, const float2 uv,
    int transportMode = TRANSPORTMODE_RADIANCE)
{
    const Material& mat = materials[materialID];
    float4 albedo = mat.albedo;
    if (mat.hasTexture)
        sampleTexture(mat, textures, uv, albedo);

    float trans = mat.transmission;
    float4 trans4;
    if (mat.hasTransMap)
    {
        sampleTexture(mat, textures, uv, trans4);
        trans = trans4.x;
    }

    if (mat.type == MAT_DIFFUSE)
    {
        cosine_f(mat.albedo, f_val);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_f(mat.eta, mat.k, mat.roughness, -wi, wo, f_val);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        //smooth_dielectric_f(-wi, wo, etaI, etaT, reflect_dielectric, TIR, f_val);
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_f(albedo, mat.ior, etaI, mat.roughness, trans, -wi, wo, f_val);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_f(f_val, wo);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_f(etaI, etaT, mat.roughness, -wi, wo, f_val);
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ void sample_f_eval(curandState& localState, const Material* materials, int materialID, float4* textures, 
    const float4& wi, float etaI, float etaT, bool backface, float4& wo, float4& f_val, float& pdf, const float2 uv, 
    int transportMode = TRANSPORTMODE_RADIANCE)
{
    if (isnan(wi.x) || isnan(wi.y) || isnan(wi.z)) {
        //printf("NaN DETECTED ON INPUT WI: (%f, %f, %f)\n", wi.x, wi.y, wi.z);
    }
    const Material& mat = materials[materialID];
    float4 albedo = mat.albedo;
    if (mat.hasTexture)
        sampleTexture(mat, textures, uv, albedo);
    
    float trans = mat.transmission;
    float4 trans4;
    if (mat.hasTransMap)
    {
        sampleTexture(mat, textures, uv, trans4);
        trans = trans4.x;
    }
        
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_sample_f(localState, albedo, wo, f_val, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_metal_sample_f(localState, mat.eta, mat.k, mat.roughness, -wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        dumb_smooth_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
        //smooth_dielectric_sample_f(localState, -wi, etaI, etaT, wo, f_val, pdf);
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_sample_f(localState, -wi, mat.ior, etaI, mat.roughness, albedo, trans, wo, f_val, pdf);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_sample_f(-wi, wo, f_val, pdf);
    }
    else if (mat.type == MAT_THINDIELECTRIC)
    {
        thin_dielectric_sample_f(localState, -wi, mat.ior, backface, transportMode, wo, f_val, pdf);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_sample_f(localState, -wi, mat.ior, mat.roughness, backface, transportMode, wo, f_val, pdf);
    }
}

// For dielectrics, when this function is called, we know whether or not it refracts, and that etaI and etaT are in fact correct
// wi passed in is facing the surface, so we flip it normally. The shading uses wi as pointing away
__device__ void pdf_eval(Material* materials, int materialID, float4* textures, const float4& wi, const float4& wo, 
    float etaI, float etaT, float& pdf, const float2 uv)
{
    const Material& mat = materials[materialID];

    float trans = mat.transmission;
    float4 trans4;
    if (mat.hasTransMap)
    {
        sampleTexture(mat, textures, uv, trans4);
        trans = trans4.x;
    }
    
    if (mat.type == MAT_DIFFUSE)
    {
        cosine_pdf(wo, pdf);
    }
    else if (mat.type == MAT_METAL)
    {
        microfacet_pdf(mat.roughness, -wi, wo, pdf);
    }
    else if (mat.type == MAT_SMOOTHDIELECTRIC)
    {
        pdf = 0.0f;
    }
    else if (mat.type == MAT_LEAF)
    {
        leaf_pdf(mat.ior, etaI, mat.roughness, trans, -wi, wo, pdf);
    }
    else if (mat.type == MAT_DELTAMIRROR)
    {
        mirror_pdf(pdf);
    }
    else if (mat.type == MAT_MICROFACETDIELECTRIC)
    {
        microfacet_dielectric_pdf(etaI, etaT, mat.roughness, -wi, wo, pdf);
    }
}