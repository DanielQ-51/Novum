#pragma once

constexpr int N = 4;

struct SampledWavelengths {
    float lambda[N];
    float pdf[N];
};

struct Wavelengths {
    float lambda[N];
};

struct __align__(16) SampledSpectrum {
    float v[N];
};

__forceinline__ __device__ __host__ SampledSpectrum operator+(const SampledSpectrum& a, const SampledSpectrum& b) {
    SampledSpectrum r;
    for (int i = 0; i < N; ++i) r.v[i] = a.v[i] + b.v[i];
    return r;
}

__forceinline__ __device__ __host__ SampledSpectrum operator*(const SampledSpectrum& a, const SampledSpectrum& b) {
    SampledSpectrum r;
    for (int i = 0; i < N; ++i) r.v[i] = a.v[i] * b.v[i];
    return r;
}

__forceinline__ __device__ __host__ SampledSpectrum operator*(const SampledSpectrum& a, float s) {
    SampledSpectrum r;
    for (int i = 0; i < N; ++i) r.v[i] = a.v[i] * s;
    return r;
}

__forceinline__ __device__ __host__ SampledSpectrum operator/(const SampledSpectrum& a, float s) {
    SampledSpectrum r;
    for (int i = 0; i < N; ++i) r.v[i] = a.v[i] / s;
    return r;
}

__forceinline__ __device__ __host__ SampledSpectrum& operator+=(SampledSpectrum& a, const SampledSpectrum& b) {
    for (int i = 0; i < N; ++i) a.v[i] += b.v[i];
    return a;
}

__forceinline__ __device__ __host__ SampledSpectrum& operator*=(SampledSpectrum& a, const SampledSpectrum& b) {
    for (int i = 0; i < N; ++i) a.v[i] *= b.v[i];
    return a;
}

__forceinline__ __device__ __host__ float maxComponent(const SampledSpectrum& a) {
    float m = a.v[0];
    for (int i = 1; i < N; ++i) m = fmaxf(m, a.v[i]);
    return m;
}

// u in [0,1). Uniform over 360..830 nm.
__forceinline__ __device__ __host__ SampledWavelengths sampleWavelengthsUniform(float u) {
    SampledWavelengths wl;
    for (int i = 0; i < N; ++i) {
        float ui = u + float(i) / N;
        if (ui >= 1.0f) ui -= 1.0f;
        wl.lambda[i] = 360.0f + 470.0f * ui;
        wl.pdf[i]    = 1.0f / 470.0f;
    }
    return wl;
}

// u in [0,1). pdf proportional to 1 / cosh^2(0.0072 * (lambda - 538)) on 360..830 nm.
__forceinline__ __device__ __host__ SampledWavelengths sampleWavelengthsVisible(float u) {
    SampledWavelengths wl;
    for (int i = 0; i < N; ++i) {
        float ui = u + float(i) / N;
        if (ui >= 1.0f) ui -= 1.0f;
        wl.lambda[i] = 538.0f - 138.888889f * atanhf(0.85691062f - 1.82750197f * ui);
        float c = coshf(0.0072f * (wl.lambda[i] - 538.0f));
        wl.pdf[i] = 0.0039398042f / (c * c);
    }
    return wl;
}

// Keep only the hero (index 0). Dividing its pdf by N gives the xN compensation at the film.
__forceinline__ __device__ __host__ void terminateSecondary(SampledWavelengths& wl) {
    if (wl.pdf[1] == 0.0f) return;
    for (int i = 1; i < N; ++i) wl.pdf[i] = 0.0f;
    wl.pdf[0] /= N;
}

// RGB -> sigmoid polynomial coefficients (Jakob & Hanika 2019), adapted from pbrt-v4
// RGBToSpectrumTable (Apache-2.0, Copyright(c) 1998-2020 Pharr, Jakob, Humphreys).
// Table data comes from rgb2spec_opt (sRGB, res 64) as two flat arrays, host or device:
//   scale  [RGB2SPEC_RES]                                   (pbrt: sRGBToSpectrumTable_Scale, "zNodes")
//   coeffs [3][RGB2SPEC_RES][RGB2SPEC_RES][RGB2SPEC_RES][3]  (pbrt: sRGBToSpectrumTable_Data)
constexpr int RGB2SPEC_RES = 64;

// rgb in [0,1]. Returns (c0, c1, c2): s(lambda) = S(c0*lambda^2 + c1*lambda + c2), lambda in nm.
__forceinline__ __device__ __host__ float3 rgbToSigmoidCoeffs(float3 rgb, const float* __restrict__ scale, const float* __restrict__ coeffs) {
    // The table is only defined on [0,1]^3: a negative component (e.g. an HDR texel slightly below 0) would index
    // outside it. fmaxf also maps NaN to 0.
    rgb = make_float3(fminf(fmaxf(rgb.x, 0.0f), 1.0f), fminf(fmaxf(rgb.y, 0.0f), 1.0f), fminf(fmaxf(rgb.z, 0.0f), 1.0f));

    if (rgb.x == rgb.y && rgb.y == rgb.z)
        return make_float3(0.0f, 0.0f, (rgb.x - 0.5f) / sqrtf(rgb.x * (1.0f - rgb.x)));

    float c[3] = { rgb.x, rgb.y, rgb.z };
    int maxc = (c[0] > c[1]) ? ((c[0] > c[2]) ? 0 : 2) : ((c[1] > c[2]) ? 1 : 2);
    float z = c[maxc];
    float x = c[(maxc + 1) % 3] * (RGB2SPEC_RES - 1) / z;
    float y = c[(maxc + 2) % 3] * (RGB2SPEC_RES - 1) / z;

    int xi = (int)x < RGB2SPEC_RES - 2 ? (int)x : RGB2SPEC_RES - 2;
    int yi = (int)y < RGB2SPEC_RES - 2 ? (int)y : RGB2SPEC_RES - 2;

    // largest zi in [0, RES-2] with scale[zi] < z (pbrt FindInterval)
    int lo = 0, hi = RGB2SPEC_RES - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
#ifdef __CUDA_ARCH__
        float s = __ldg(&scale[mid]);
#else
        float s = scale[mid];
#endif
        if (s < z) lo = mid; else hi = mid - 1;
    }
    int zi = lo;

#ifdef __CUDA_ARCH__
    float z0 = __ldg(&scale[zi]), z1 = __ldg(&scale[zi + 1]);
#else
    float z0 = scale[zi], z1 = scale[zi + 1];
#endif
    float fx = x - xi, fy = y - yi, fz = (z - z0) / (z1 - z0);

    // corner k = dx + 2*dy + 4*dz
    float co[8][3];
    for (int k = 0; k < 8; ++k) {
        const float* p = coeffs + ((((maxc * RGB2SPEC_RES + zi + (k >> 2)) * RGB2SPEC_RES + yi + ((k >> 1) & 1)) * RGB2SPEC_RES + xi + (k & 1)) * 3);
#ifdef __CUDA_ARCH__
        co[k][0] = __ldg(p); co[k][1] = __ldg(p + 1); co[k][2] = __ldg(p + 2);
#else
        co[k][0] = p[0]; co[k][1] = p[1]; co[k][2] = p[2];
#endif
    }

    float out[3];
    for (int i = 0; i < 3; ++i) {
        float x00 = (1.0f - fx) * co[0][i] + fx * co[1][i];
        float x10 = (1.0f - fx) * co[2][i] + fx * co[3][i];
        float x01 = (1.0f - fx) * co[4][i] + fx * co[5][i];
        float x11 = (1.0f - fx) * co[6][i] + fx * co[7][i];
        float y0 = (1.0f - fy) * x00 + fy * x10;
        float y1 = (1.0f - fy) * x01 + fy * x11;
        out[i] = (1.0f - fz) * y0 + fz * y1;
    }
    return make_float3(out[0], out[1], out[2]);
}

// pbrt RGBSigmoidPolynomial::operator(). pbrt only special-cases infinite x (black/white); the
// threshold also catches finite x large enough that x * x overflows and the result collapses to 0.5.
__forceinline__ __device__ __host__ float sigmoidPolynomial(float3 c, float lambda) {
    float x = (c.x * lambda + c.y) * lambda + c.z;
    if (fabsf(x) > 1e18f) return x > 0.0f ? 1.0f : 0.0f;
    return 0.5f + x / (2.0f * sqrtf(1.0f + x * x));
}

#ifndef SPECTRAL_PHOTON_MAP_SIZE
#define SPECTRAL_PHOTON_MAP_SIZE 36u
#endif

struct SpectralPhotonMap {
    float4* __restrict__ pos_plus_normal; // hot data (normal is 2x15 oct with backface as flag)

    SampledSpectrum* __restrict__ radiance; // 16 byte
    uint32_t* __restrict__ wi; // 4 byte
};

__host__ inline void* allocateSpectralPhotonMap(SpectralPhotonMap& r, uint32_t numPhoton) {
    numPhoton = (numPhoton + 31) & ~31;

    void* raw;
    cudaMalloc(&raw, numPhoton * SPECTRAL_PHOTON_MAP_SIZE);

    char* ptr = static_cast<char*>(raw);
    r.pos_plus_normal = reinterpret_cast<float4*>(ptr); ptr += numPhoton * sizeof(float4);          // 16B
    r.radiance = reinterpret_cast<SampledSpectrum*>(ptr); ptr += numPhoton * sizeof(SampledSpectrum); // 16B
    r.wi = reinterpret_cast<uint32_t*>(ptr); ptr += numPhoton * sizeof(uint32_t);          // 4B

    return raw;
}

// ---------------------------------------------------------------------------
// Photon hash grid. Cell size = merge radius, grid origin = world origin (the grid is hashed, so negative
// cells are fine). Used by both computeHashes (binning) and the eye raygen (lookup), so they can't disagree.
// ---------------------------------------------------------------------------

__forceinline__ __device__ __host__ int3 photonGridCell(float3 p, float cellSize) {
    return make_int3((int)floorf(p.x / cellSize), (int)floorf(p.y / cellSize), (int)floorf(p.z / cellSize));
}

__forceinline__ __device__ __host__ uint32_t photonGridHash(int3 cell, uint32_t hashTableSize) {
    uint32_t n = (73856093u * (uint32_t)cell.x) ^ (19349663u * (uint32_t)cell.y) ^ (83492791u * (uint32_t)cell.z);
    return n % hashTableSize;
}


// ---------------------------------------------------------------------------
// Photon map access. pos_plus_normal.w holds packOctFlags(normal, backface, heroOnly)
// as raw bits. _cs = streaming (evict-first), _ldg = read-only cache.
// heroOnly: the photon path refracted through a dispersive surface, so only radiance.v[0] is valid.
// Unsorted map: written once by the light raygen, read once by hashing/reorder -> _cs.
// Sorted map: written by reorder, read many times by the eye raygen gather -> _ldg.
// ---------------------------------------------------------------------------

static_assert(N == 4, "photon radiance accessors move SampledSpectrum as a single float4");

// ----- write: light raygen -> unsorted map -----

__forceinline__ __device__ void setPosNormal(const SpectralPhotonMap& m, uint32_t idx, float3 pos, float3 normal, bool backface, bool heroOnly) {
    m.pos_plus_normal[idx] = make_float4(pos.x, pos.y, pos.z, __uint_as_float(packOctFlags(normal, backface, heroOnly)));
}

__forceinline__ __device__ void setPosNormal_cs(const SpectralPhotonMap& m, uint32_t idx, float3 pos, float3 normal, bool backface, bool heroOnly) {
    __stcs(&m.pos_plus_normal[idx], make_float4(pos.x, pos.y, pos.z, __uint_as_float(packOctFlags(normal, backface, heroOnly))));
}

__forceinline__ __device__ void setRadiance(const SpectralPhotonMap& m, uint32_t idx, const SampledSpectrum& s) {
    *reinterpret_cast<float4*>(&m.radiance[idx]) = make_float4(s.v[0], s.v[1], s.v[2], s.v[3]);
}

__forceinline__ __device__ void setRadiance_cs(const SpectralPhotonMap& m, uint32_t idx, const SampledSpectrum& s) {
    __stcs(reinterpret_cast<float4*>(&m.radiance[idx]), make_float4(s.v[0], s.v[1], s.v[2], s.v[3]));
}

__forceinline__ __device__ void setWi(const SpectralPhotonMap& m, uint32_t idx, float3 wi) {
    m.wi[idx] = packOct(wi);
}

__forceinline__ __device__ void setWi_cs(const SpectralPhotonMap& m, uint32_t idx, float3 wi) {
    __stcs(reinterpret_cast<unsigned int*>(&m.wi[idx]), packOct(wi));
}

__forceinline__ __device__ void setPhoton(const SpectralPhotonMap& m, uint32_t idx, float3 pos, float3 normal, bool backface, bool heroOnly,
    const SampledSpectrum& radiance, float3 wi) {
    setPosNormal(m, idx, pos, normal, backface, heroOnly);
    setRadiance(m, idx, radiance);
    setWi(m, idx, wi);
}

__forceinline__ __device__ void setPhoton_cs(const SpectralPhotonMap& m, uint32_t idx, float3 pos, float3 normal, bool backface, bool heroOnly,
    const SampledSpectrum& radiance, float3 wi) {
    setPosNormal_cs(m, idx, pos, normal, backface, heroOnly);
    setRadiance_cs(m, idx, radiance);
    setWi_cs(m, idx, wi);
}

// ----- raw (still packed) loads/stores: hashing and reorder -----

__forceinline__ __device__ float4 getPosNormalPacked_cs(const SpectralPhotonMap& m, uint32_t idx) {
    return __ldcs(&m.pos_plus_normal[idx]);
}

__forceinline__ __device__ float4 getPosNormalPacked_ldg(const SpectralPhotonMap& m, uint32_t idx) {
    return __ldg(&m.pos_plus_normal[idx]);
}

__forceinline__ __device__ float4 getRadiancePacked_cs(const SpectralPhotonMap& m, uint32_t idx) {
    return __ldcs(reinterpret_cast<const float4*>(&m.radiance[idx]));
}

__forceinline__ __device__ float4 getRadiancePacked_ldg(const SpectralPhotonMap& m, uint32_t idx) {
    return __ldg(reinterpret_cast<const float4*>(&m.radiance[idx]));
}

__forceinline__ __device__ uint32_t getWiPacked_cs(const SpectralPhotonMap& m, uint32_t idx) {
    return __ldcs(reinterpret_cast<const unsigned int*>(&m.wi[idx]));
}

__forceinline__ __device__ uint32_t getWiPacked_ldg(const SpectralPhotonMap& m, uint32_t idx) {
    return __ldg(reinterpret_cast<const unsigned int*>(&m.wi[idx]));
}

__forceinline__ __device__ void setPosNormalPacked(const SpectralPhotonMap& m, uint32_t idx, float4 packed) {
    m.pos_plus_normal[idx] = packed;
}

__forceinline__ __device__ void setPosNormalPacked_cs(const SpectralPhotonMap& m, uint32_t idx, float4 packed) {
    __stcs(&m.pos_plus_normal[idx], packed);
}

__forceinline__ __device__ void setRadiancePacked(const SpectralPhotonMap& m, uint32_t idx, float4 packed) {
    *reinterpret_cast<float4*>(&m.radiance[idx]) = packed;
}

__forceinline__ __device__ void setRadiancePacked_cs(const SpectralPhotonMap& m, uint32_t idx, float4 packed) {
    __stcs(reinterpret_cast<float4*>(&m.radiance[idx]), packed);
}

__forceinline__ __device__ void setWiPacked(const SpectralPhotonMap& m, uint32_t idx, uint32_t packed) {
    m.wi[idx] = packed;
}

__forceinline__ __device__ void setWiPacked_cs(const SpectralPhotonMap& m, uint32_t idx, uint32_t packed) {
    __stcs(reinterpret_cast<unsigned int*>(&m.wi[idx]), packed);
}

// ----- hashing: coalesced, read-once pass over the unsorted map -----

__forceinline__ __device__ float3 getPos_cs(const SpectralPhotonMap& m, uint32_t idx) {
    float4 p = __ldcs(&m.pos_plus_normal[idx]);
    return make_float3(p.x, p.y, p.z);
}

// ----- reorder: gather unsorted[src] (read once) -> sorted[dst] (kept for the gather) -----

__forceinline__ __device__ uint32_t getSortedIndex_cs(const uint32_t* __restrict__ indices, uint32_t i) {
    return __ldcs(reinterpret_cast<const unsigned int*>(&indices[i]));
}

__forceinline__ __device__ void copyPhoton(const SpectralPhotonMap& src, uint32_t srcIdx, const SpectralPhotonMap& dst, uint32_t dstIdx) {
    dst.pos_plus_normal[dstIdx] = __ldcs(&src.pos_plus_normal[srcIdx]);
    *reinterpret_cast<float4*>(&dst.radiance[dstIdx]) = __ldcs(reinterpret_cast<const float4*>(&src.radiance[srcIdx]));
    dst.wi[dstIdx] = __ldcs(reinterpret_cast<const unsigned int*>(&src.wi[srcIdx]));
}

__forceinline__ __device__ void copyPhoton_cs(const SpectralPhotonMap& src, uint32_t srcIdx, const SpectralPhotonMap& dst, uint32_t dstIdx) {
    __stcs(&dst.pos_plus_normal[dstIdx], __ldcs(&src.pos_plus_normal[srcIdx]));
    __stcs(reinterpret_cast<float4*>(&dst.radiance[dstIdx]), __ldcs(reinterpret_cast<const float4*>(&src.radiance[srcIdx])));
    __stcs(reinterpret_cast<unsigned int*>(&dst.wi[dstIdx]), __ldcs(reinterpret_cast<const unsigned int*>(&src.wi[srcIdx])));
}

__forceinline__ __device__ void copyPhotonHot(const SpectralPhotonMap& src, uint32_t srcIdx, const SpectralPhotonMap& dst, uint32_t dstIdx) {
    dst.pos_plus_normal[dstIdx] = __ldcs(&src.pos_plus_normal[srcIdx]);
}

__forceinline__ __device__ void copyPhotonCold(const SpectralPhotonMap& src, uint32_t srcIdx, const SpectralPhotonMap& dst, uint32_t dstIdx) {
    *reinterpret_cast<float4*>(&dst.radiance[dstIdx]) = __ldcs(reinterpret_cast<const float4*>(&src.radiance[srcIdx]));
    dst.wi[dstIdx] = __ldcs(reinterpret_cast<const unsigned int*>(&src.wi[srcIdx]));
}

// ----- gather: eye raygen over the sorted map -----

__forceinline__ __device__ void getCellRange_ldg(const uint32_t* __restrict__ cellStart, const uint32_t* __restrict__ cellEnd, uint32_t hash,
    uint32_t& start, uint32_t& end) {
    start = __ldg(reinterpret_cast<const unsigned int*>(&cellStart[hash]));
    end   = __ldg(reinterpret_cast<const unsigned int*>(&cellEnd[hash]));
}

__forceinline__ __device__ float3 getPos_ldg(const SpectralPhotonMap& m, uint32_t idx) {
    float4 p = __ldg(&m.pos_plus_normal[idx]);
    return make_float3(p.x, p.y, p.z);
}

// Flag reads straight off a packed hot load, so backface rejects and the hero-only check skip the oct decode.
__forceinline__ __device__ bool getBackface(float4 packed) {
    return (__float_as_uint(packed.w) >> 30) & 1u;
}

__forceinline__ __device__ bool getHeroOnly(float4 packed) {
    return (__float_as_uint(packed.w) >> 31) & 1u;
}

__forceinline__ __device__ float3 getNormal(float4 packed) {
    return unpackOctFlags(__float_as_uint(packed.w), nullptr, nullptr);
}

__forceinline__ __device__ void unpackPosNormal(float4 packed, float3& pos, float3& normal, bool& backface, bool& heroOnly) {
    pos = make_float3(packed.x, packed.y, packed.z);
    normal = unpackOctFlags(__float_as_uint(packed.w), &backface, &heroOnly);
}

__forceinline__ __device__ void getPosNormal_ldg(const SpectralPhotonMap& m, uint32_t idx, float3& pos, float3& normal, bool& backface, bool& heroOnly) {
    float4 packed = __ldg(&m.pos_plus_normal[idx]);
    pos = make_float3(packed.x, packed.y, packed.z);
    normal = unpackOctFlags(__float_as_uint(packed.w), &backface, &heroOnly);
}

__forceinline__ __device__ void getNormalInfo_ldg(const SpectralPhotonMap& m, uint32_t idx, float3& normal, bool& backface, bool& heroOnly) {
    normal = unpackOctFlags(__float_as_uint(__ldg(&m.pos_plus_normal[idx]).w), &backface, &heroOnly);
}

__forceinline__ __device__ SampledSpectrum getRadiance_ldg(const SpectralPhotonMap& m, uint32_t idx) {
    float4 r = __ldg(reinterpret_cast<const float4*>(&m.radiance[idx]));
    SampledSpectrum s;
    s.v[0] = r.x; s.v[1] = r.y; s.v[2] = r.z; s.v[3] = r.w;
    return s;
}

__forceinline__ __device__ SampledSpectrum getRadiance_cs(const SpectralPhotonMap& m, uint32_t idx) {
    float4 r = __ldcs(reinterpret_cast<const float4*>(&m.radiance[idx]));
    SampledSpectrum s;
    s.v[0] = r.x; s.v[1] = r.y; s.v[2] = r.z; s.v[3] = r.w;
    return s;
}

__forceinline__ __device__ float3 getWi_ldg(const SpectralPhotonMap& m, uint32_t idx) {
    return unpackOct(__ldg(reinterpret_cast<const unsigned int*>(&m.wi[idx])));
}

__forceinline__ __device__ float3 getWi_cs(const SpectralPhotonMap& m, uint32_t idx) {
    return unpackOct(__ldcs(reinterpret_cast<const unsigned int*>(&m.wi[idx])));
}
