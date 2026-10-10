#pragma once

#include "reflectors.cuh"
#include "spectralPPM/spectralPPM_Utils.cuh"

namespace SpectralShading {
    struct MaterialInfo { // cached material data specfic to one shading position
        int  type;
        bool isSpecular;
        bool backface;              // which side was hit: orients eta for dielectrics

        SampledSpectrum albedo;     // diffuse / principled: base color at the iteration's wavelengths
        float metallic;             // principled, after mrTex
        float alpha;                // principled / metal: roughness (after mrTex, clamped to 0.025), squared
        float pSpec;                // principled lobe-selection probability, computed from RGB (wavelength-independent)

        SampledSpectrum eta;        // metal: eta(lambda). dispersive dielectric: Sellmeier n(lambda), eta.v[0] picks the refraction direction
        SampledSpectrum k;          // metal: k(lambda)
    };

    // Everything an evaluation at this hit needs, computed once: texture fetches, RGB -> spectrum at wl, derived terms.
    // Supported: MAT_DIFFUSE, MAT_GLTF_PRINCIPLED_BSDF (opaque), MAT_METAL, MAT_DISPERSIVEDIELECTRIC.
    __device__ inline MaterialInfo buildMaterialInfo(
        const Material* __restrict__ materials, int materialID, const TextureView& textures,
        float2 uv, bool backface, const Wavelengths& wl,
        const float* __restrict__ tableScale, const float* __restrict__ tableData,
        float lod = 0.0f)
    {
        const Material& mat = materials[materialID];

        MaterialInfo info = {};
        info.type = mat.type;
        info.isSpecular = mat.isSpecular;
        info.backface = backface;

        if (mat.type == MAT_DISPERSIVEDIELECTRIC) {
            // Sellmeier n^2 = 1 + sum B l^2 / (l^2 - C), l in micrometers, then the spread around nd is
            // scaled by dispersionScale. Kept above 1 so an extreme scale can't push red below vacuum.
            for (int i = 0; i < N; ++i) {
                float l2 = wl.lambda[i] * wl.lambda[i] * 1e-6f;
                float n = sqrtf(1.0f + mat.sellB[0] * l2 / (l2 - mat.sellC[0])
                                     + mat.sellB[1] * l2 / (l2 - mat.sellC[1])
                                     + mat.sellB[2] * l2 / (l2 - mat.sellC[2]));
                info.eta.v[i] = fmaxf(mat.nd + mat.dispersionScale * (n - mat.nd), 1.0001f);
            }
            return info;
        }

        if (mat.type == MAT_METAL) {
            float roughness = fmaxf(mat.roughness, 0.025f);
            info.alpha = roughness * roughness;

            // eta and k are not reflectances (eta > 1 is normal), so upsample them as unbounded spectra:
            // scale = 2 * max component, look up rgb / scale, multiply the curve back by scale.
            float3 eta = f3(mat.eta);
            float etaScale = 2.0f * fmaxf(eta.x, fmaxf(eta.y, eta.z));
            if (etaScale > 0.0f) {
                float3 c = rgbToSigmoidCoeffs(eta * (1.0f / etaScale), tableScale, tableData);
                for (int i = 0; i < N; ++i)
                    info.eta.v[i] = etaScale * sigmoidPolynomial(c, wl.lambda[i]);
            }

            float3 k = f3(mat.k);
            float kScale = 2.0f * fmaxf(k.x, fmaxf(k.y, k.z));
            if (kScale > 0.0f) {
                float3 c = rgbToSigmoidCoeffs(k * (1.0f / kScale), tableScale, tableData);
                for (int i = 0; i < N; ++i)
                    info.k.v[i] = kScale * sigmoidPolynomial(c, wl.lambda[i]);
            }
            return info;
        }

        // diffuse / principled: base color. The table expects linear RGB in [0,1].
        float3 baseColor = f3(mat.albedo);
        if (mat.baseColorTex >= 0)
            baseColor = f3(sampleTex(textures, mat.baseColorTex, uv, lod));
        baseColor = make_float3(fminf(fmaxf(baseColor.x, 0.0f), 1.0f),
                                fminf(fmaxf(baseColor.y, 0.0f), 1.0f),
                                fminf(fmaxf(baseColor.z, 0.0f), 1.0f));

        float3 c = rgbToSigmoidCoeffs(baseColor, tableScale, tableData);
        for (int i = 0; i < N; ++i)
            info.albedo.v[i] = sigmoidPolynomial(c, wl.lambda[i]);

        if (mat.type == MAT_GLTF_PRINCIPLED_BSDF) {
            float metallic = mat.metallic;
            float roughness = mat.roughness;
            if (mat.mrTex >= 0) {
                float4 mr = sampleTex(textures, mat.mrTex, uv, lod);
                roughness *= mr.y; // glTF: roughness in G
                metallic  *= mr.z; // glTF: metallic in B
            }
            roughness = fmaxf(roughness, 0.025f);
            info.metallic = metallic;
            info.alpha = roughness * roughness;

            // same selection probability as the RGB principled_sample_f / principled_pdf, from the RGB base color
            float3 F0    = f3(0.04f) * (1.0f - metallic) + baseColor * metallic;
            float3 cDiff = baseColor * (1.0f - metallic);
            info.pSpec = principled_specProb(F0, cDiff);
        }
        return info;
    }

    // -------------------------------------------------------------------------------------------------
    // Scalar Fresnel, called once per wavelength
    // -------------------------------------------------------------------------------------------------

    // Exact dielectric Fresnel (unpolarized). eta = n_transmitted / n_incident. A negative cosine means the
    // ray arrives from the other side, so the ratio flips. Returns 1 on total internal reflection.
    __forceinline__ __device__ __host__ float fresnelDielectric(float cosThetaI, float eta) {
        cosThetaI = fminf(fmaxf(cosThetaI, -1.0f), 1.0f);
        if (cosThetaI < 0.0f) { eta = 1.0f / eta; cosThetaI = -cosThetaI; }

        float sin2ThetaT = (1.0f - cosThetaI * cosThetaI) / (eta * eta);
        if (sin2ThetaT >= 1.0f) return 1.0f;
        float cosThetaT = sqrtf(1.0f - sin2ThetaT);

        float rParl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
        float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
        return 0.5f * (rParl * rParl + rPerp * rPerp);
    }

    // Conductor Fresnel for complex IOR eta + i k (unpolarized average of Rs and Rp).
    __forceinline__ __device__ __host__ float fresnelConductor(float cosTheta, float eta, float k) {
        cosTheta = fminf(fmaxf(cosTheta, 0.0f), 1.0f);
        float cos2 = cosTheta * cosTheta;
        float sin2 = 1.0f - cos2;
        float eta2 = eta * eta;
        float k2 = k * k;

        float t0 = eta2 - k2 - sin2;
        float a2plusb2 = sqrtf(t0 * t0 + 4.0f * eta2 * k2);
        float t1 = a2plusb2 + cos2;
        float a = sqrtf(fmaxf(0.5f * (a2plusb2 + t0), 0.0f));
        float t2 = 2.0f * cosTheta * a;
        float Rs = (t1 - t2) / (t1 + t2);

        float t3 = cos2 * a2plusb2 + sin2 * sin2;
        float t4 = t2 * sin2;
        float Rp = Rs * (t3 - t4) / (t3 + t4);
        return 0.5f * (Rs + Rp);
    }

    // -------------------------------------------------------------------------------------------------
    // Lobes. Local shading frame; v points away from the surface toward the previous vertex, l toward the next.
    // -------------------------------------------------------------------------------------------------

    // ----- diffuse -----

    __forceinline__ __device__ void diffuse_f(const MaterialInfo& info, const float3& v, const float3& l, SampledSpectrum& f) {
        if (v.z <= 0.0f || l.z <= 0.0f) {
            for (int i = 0; i < N; ++i) f.v[i] = 0.0f;
            return;
        }
        for (int i = 0; i < N; ++i) f.v[i] = info.albedo.v[i] * INVPI;
    }

    __forceinline__ __device__ void diffuse_sample_f(RNGState& localState, const MaterialInfo& info, const float3& v,
        float3& l, SampledSpectrum& f, float& pdf)
    {
        cosine_emit(localState, l, pdf);
        diffuse_f(info, v, l, f);
    }

    // ----- principled (opaque): same model as the RGB principled_f, evaluated per wavelength -----

    __forceinline__ __device__ void principled_f(const MaterialInfo& info, const float3& v, const float3& l, SampledSpectrum& f) {
        if (v.z <= 0.0f || l.z <= 0.0f) {
            for (int i = 0; i < N; ++i) f.v[i] = 0.0f;
            return;
        }
        float3 h = normalize(v + l);
        float D = D_GGX(h, info.alpha);
        float G = G_Smith(v, l, h, info.alpha);
        float specScale = D * G / fmaxf(4.0f * v.z * l.z, EPSILON);

        float m = fminf(fmaxf(1.0f - fmaxf(dot(v, h), 0.0f), 0.0f), 1.0f);
        float m5 = (m * m) * (m * m) * m;

        for (int i = 0; i < N; ++i) {
            float F0 = 0.04f * (1.0f - info.metallic) + info.albedo.v[i] * info.metallic;
            float F  = F0 + (1.0f - F0) * m5;
            f.v[i] = (1.0f - F) * info.albedo.v[i] * (1.0f - info.metallic) * INVPI + specScale * F;
        }
    }

    // Marginal pdf over both lobes, with the cached RGB-based selection probability (wavelength independent).
    __forceinline__ __device__ float principled_pdf(const MaterialInfo& info, const float3& v, const float3& l) {
        if (v.z <= 0.0f || l.z <= 0.0f) return 0.0f;
        float3 h = normalize(v + l);
        float specPdf = D_GGX(h, info.alpha) * h.z / fmaxf(4.0f * dot(v, h), EPSILON);
        float diffPdf = l.z * INVPI;
        return info.pSpec * specPdf + (1.0f - info.pSpec) * diffPdf;
    }

    // Same random number sequence as the RGB principled_sample_f: lobe choice, then two for the direction.
    __forceinline__ __device__ void principled_sample_f(RNGState& localState, const MaterialInfo& info, const float3& v,
        float3& l, SampledSpectrum& f, float& pdf)
    {
        if (rand(&localState) < info.pSpec) {
            float u1  = rand(&localState);
            float phi = 2.0f * PI * rand(&localState);
            float a2 = info.alpha * info.alpha;
            float cosT = sqrtf((1.0f - u1) / (1.0f + (a2 - 1.0f) * u1));
            float sinT = sqrtf(fmaxf(1.0f - cosT * cosT, 0.0f));
            float3 h = make_float3(sinT * cosf(phi), sinT * sinf(phi), cosT);
            l = 2.0f * dot(v, h) * h - v;
        } else {
            float pdfTmp;
            cosine_emit(localState, l, pdfTmp);
        }

        if (l.z <= 0.0f) {
            for (int i = 0; i < N; ++i) f.v[i] = 0.0f;
            pdf = 0.0f;
            return;
        }
        principled_f(info, v, l, f);
        pdf = principled_pdf(info, v, l);
    }

    // ----- microfacet conductor -----

    __forceinline__ __device__ void conductor_f(const MaterialInfo& info, const float3& v, const float3& l, SampledSpectrum& f) {
        if (v.z <= 0.0f || l.z <= 0.0f) {
            for (int i = 0; i < N; ++i) f.v[i] = 0.0f;
            return;
        }
        float3 h = normalize(v + l);
        float D = D_GGX(h, info.alpha);
        float G = G_Smith(v, l, h, info.alpha);
        float s = D * G / fmaxf(4.0f * v.z * l.z, EPSILON);
        float vh = dot(v, h);
        for (int i = 0; i < N; ++i)
            f.v[i] = s * fresnelConductor(vh, info.eta.v[i], info.k.v[i]);
    }

    // GGX half-vector sampling. Unlike the RGB version, a direction below the surface ends the path
    // (f = 0, pdf = 0) instead of being mirrored back up, which would not match the pdf.
    __forceinline__ __device__ void conductor_sample_f(RNGState& localState, const MaterialInfo& info, const float3& v,
        float3& l, SampledSpectrum& f, float& pdf)
    {
        float u1  = rand(&localState);
        float phi = 2.0f * PI * rand(&localState);
        float a2 = info.alpha * info.alpha;
        float cosT = sqrtf((1.0f - u1) / (1.0f + (a2 - 1.0f) * u1));
        float sinT = sqrtf(fmaxf(1.0f - cosT * cosT, 0.0f));
        float3 h = make_float3(sinT * cosf(phi), sinT * sinf(phi), cosT);
        float vh = dot(v, h);
        l = 2.0f * vh * h - v;

        if (l.z <= 0.0f || vh <= 0.0f) {
            for (int i = 0; i < N; ++i) f.v[i] = 0.0f;
            pdf = 0.0f;
            return;
        }
        conductor_f(info, v, l, f);
        pdf = D_GGX(h, info.alpha) * h.z / fmaxf(4.0f * vh, EPSILON);
    }

    // ----- dispersive smooth dielectric (delta) -----
    // Reflect vs refract is chosen with the hero's Fresnel. Reflection keeps every wavelength (same direction for all,
    // per-wavelength weight F_i / F_hero). Refraction uses the hero's index for the direction and, if the indices
    // actually differ across the sampled wavelengths, drops the secondaries (heroOnly).
    __forceinline__ __device__ void dispersive_dielectric_sample_f(RNGState& localState, const MaterialInfo& info, const float3& v,
        int transportMode, float3& l, SampledSpectrum& f, float& pdf, bool& heroOnly)
    {
        float cosI = fminf(fmaxf(v.z, EPSILON), 1.0f);
        float eta0 = info.backface ? 1.0f / info.eta.v[0] : info.eta.v[0]; // transmitted over incident
        float F0 = fresnelDielectric(cosI, eta0);

        if (rand(&localState) < F0) {
            l = make_float3(-v.x, -v.y, v.z);
            pdf = F0;
            for (int i = 0; i < N; ++i) {
                float etaI = info.backface ? 1.0f / info.eta.v[i] : info.eta.v[i];
                f.v[i] = fresnelDielectric(cosI, etaI) / cosI;
            }
            return;
        }

        float sin2T = (1.0f - cosI * cosI) / (eta0 * eta0);
        float cosT = sqrtf(fmaxf(1.0f - sin2T, 0.0f));
        l = make_float3(-v.x / eta0, -v.y / eta0, -cosT);
        pdf = 1.0f - F0;

        float val = (1.0f - F0) / fmaxf(cosT, EPSILON);
        if (transportMode == TRANSPORTMODE_RADIANCE)
            val /= eta0 * eta0; // radiance is compressed into the denser side; importance is not

        bool dispersive = false;
        for (int i = 1; i < N; ++i) dispersive = dispersive || (info.eta.v[i] != info.eta.v[0]);

        f.v[0] = val;
        for (int i = 1; i < N; ++i) f.v[i] = dispersive ? 0.0f : val;
        heroOnly = dispersive;
    }

    // -------------------------------------------------------------------------------------------------
    // Dispatchers. Same conventions as the RGB f_eval / sample_f_eval: local frame, wi passed facing the surface.
    // -------------------------------------------------------------------------------------------------

    // Delta and unsupported materials return 0, as in the RGB f_eval.
    __forceinline__ __device__ void f_eval(const MaterialInfo& info, const float3& wi, const float3& wo, SampledSpectrum& f) {
        float3 v = -wi;
        if (info.type == MAT_DIFFUSE)
            diffuse_f(info, v, wo, f);
        else if (info.type == MAT_GLTF_PRINCIPLED_BSDF && !info.isSpecular)
            principled_f(info, v, wo, f);
        else if (info.type == MAT_METAL)
            conductor_f(info, v, wo, f);
        else
            for (int i = 0; i < N; ++i) f.v[i] = 0.0f;
    }

    // heroOnly is set when this sample refracted through a dispersive surface; the caller ORs it into the path's flag
    // (the returned f already has the secondaries zeroed). Unsupported materials return pdf = 0, ending the path.
    __forceinline__ __device__ void sample_f_eval(RNGState& localState, const MaterialInfo& info, const float3& wi, int transportMode,
        float3& wo, SampledSpectrum& f, float& pdf, bool& heroOnly)
    {
        float3 v = -wi;
        heroOnly = false;
        if (info.type == MAT_DIFFUSE)
            diffuse_sample_f(localState, info, v, wo, f, pdf);
        else if (info.type == MAT_GLTF_PRINCIPLED_BSDF && !info.isSpecular)
            principled_sample_f(localState, info, v, wo, f, pdf);
        else if (info.type == MAT_METAL)
            conductor_sample_f(localState, info, v, wo, f, pdf);
        else if (info.type == MAT_DISPERSIVEDIELECTRIC)
            dispersive_dielectric_sample_f(localState, info, v, transportMode, wo, f, pdf, heroOnly);
        else {
            for (int i = 0; i < N; ++i) f.v[i] = 0.0f;
            pdf = 0.0f;
        }
    }
}

