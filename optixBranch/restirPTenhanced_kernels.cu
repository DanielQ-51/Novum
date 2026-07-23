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


#ifndef DUALMV_SEARCH_RADIUS
#define DUALMV_SEARCH_RADIUS 2
#endif

#ifndef DUPE_MAP_SEARCH_RADIUS
#define DUPE_MAP_SEARCH_RADIUS 8
#endif

#ifndef DUPE_MAP_SEARCH_STRIDE
#define DUPE_MAP_SEARCH_STRIDE 1
#endif

__global__ void computeDualMV(
    GBuffer gbuffer,
    uint32_t w,
    uint32_t h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int center_idx = y * w + x;

    float closest_depth = gbuffer.getDepth_notstreaming(center_idx);
    half2 best_dual_mv = gbuffer.getMV_notstreaming(center_idx);

    for (int dy = -DUALMV_SEARCH_RADIUS; dy <= DUALMV_SEARCH_RADIUS; dy++) {
        for (int dx = -DUALMV_SEARCH_RADIUS; dx <= DUALMV_SEARCH_RADIUS; dx++) {

            int nx = clamp(x + dx, 0, w - 1);
            int ny = clamp(y + dy, 0, h - 1);
            int neighbor_idx = ny * w + nx;

            float neighbor_depth = gbuffer.getDepth_notstreaming(neighbor_idx);

            // If neighbor is significantly closer to the camera, steal its MV
            if (neighbor_depth < closest_depth) {
                closest_depth = neighbor_depth;
                best_dual_mv = gbuffer.getMV_notstreaming(neighbor_idx);
            }
        }
    }

    gbuffer.setDualMotionVec(center_idx, best_dual_mv);
}

__global__ void computeDuplicationMapKernel(
    Reservoir lastFrameReservoir,
    uint8_t* __restrict__ duplication_map,
    uint32_t w,
    uint32_t h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int center_idx = y * w + x;

    uint32_t currSeed = lastFrameReservoir.getSeed_notstreaming(center_idx);

    if (currSeed == 0u) {
        duplication_map[center_idx] = 0u;
        return;
    }

    int match_count = 0;
    int valid_pixels = 0;

    for (int dy = -DUPE_MAP_SEARCH_RADIUS; dy <= DUPE_MAP_SEARCH_RADIUS; dy += DUPE_MAP_SEARCH_STRIDE) {
        for (int dx = -DUPE_MAP_SEARCH_RADIUS; dx <= DUPE_MAP_SEARCH_RADIUS; dx += DUPE_MAP_SEARCH_STRIDE) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                continue;
            }

            uint32_t neighbor_seed = lastFrameReservoir.getSeed_notstreaming(ny * w + nx);

            if (neighbor_seed == currSeed) {
                match_count++;
            }

            valid_pixels++;
        }
    }

    __stcs(&duplication_map[center_idx],(uint8_t)(((float)match_count / (float)valid_pixels) * 255.0f));
}

__global__ void displayWinningReservoirs(PipelineParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.common.w || y >= params.common.h) return;

    int pixelIdx = y * params.common.w + x;

    if (x == DEBUG_TEST_PIXEL_X && y == DEBUG_TEST_PIXEL_Y) {
        printf("frame %u at the end seed: %u\n", params.common.frame_index, params.restir.reservoir.initRandomSeed[pixelIdx]);
        printPixelData(params.restir.reservoir, params.restir.gbuffer, pixelIdx, params.common.frame_index);
    }

    half2 mv = params.restir.gbuffer.getMV(pixelIdx);
    if (reinterpret_cast<const uint32_t&>(mv) != 0xFFFFFFFF && !params.restir.gbuffer.getSkipShade(pixelIdx)) {
        float3 output = fireflyClamp(fromRGB9E5(__ldcs(&params.restir.reservoir.F[pixelIdx])) * __ldcs(&params.restir.reservoir.W[pixelIdx]));
        if ((isnan(output.x) || isnan(output.y) || isnan(output.z))) {
            printf("nan at pixel idx: %d\n", pixelIdx);
        }

#if DEBUG_VISUALIZE_TYPE == 1
        uint32_t pathFlags = params.restir.reservoir.pathFlags[pixelIdx];
        output = debugVisualizeTechnique(extractType(pathFlags), extractRcInd(pathFlags));
#endif

#if ACCUMULATE_FRAMES == 1
        params.common.accum_buffer[pixelIdx] += f4(output);
#else
        params.common.accum_buffer[pixelIdx] = f4(output);
#endif
    }
}

__global__ void initLinks(uint32_t* buffer, uint32_t dimension) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    int pixelIdx = y * dimension + x;
    buffer[pixelIdx] = pixelIdx / 2;
}

__global__ void shuffleLinks(uint32_t* bufferA, uint32_t* bufferB, uint32_t dimension, uint32_t iteration) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dimension / 2|| y >= dimension / 2) return;

    int startX = 2 * x;
    int startY = 2 * y;

    int pixelIdx = y * dimension + x;

    if (iteration % 2 == 1) {
        startX += 1;
        startY += 1;
    }

    int2 coords[4] = {
        make_int2(startX % dimension, startY % dimension),
        make_int2((startX + 1) % dimension, startY % dimension),
        make_int2(startX % dimension, (startY + 1) % dimension),
        make_int2((startX + 1) % dimension, (startY + 1) % dimension)
    };

    uint32_t ids[4];
    for (int i = 0; i < 4; i++) {
        ids[i] = bufferA[coords[i].y * dimension + coords[i].x];
    }

    RNGState localState = load_rng(pixelIdx, iteration, hash_uint32(dimension), nullptr);

    for (int i = 3; i > 0; i--) {
        uint32_t j = (uint32_t)(rand(&localState) * (i + 1));
        j = min(j, (uint32_t)i);

        uint32_t temp = ids[i];
        ids[i] = ids[j];
        ids[j] = temp;
    }

    for (int i = 0; i < 4; i++) {
        bufferB[coords[i].y * dimension + coords[i].x] = ids[i];
    }
}

__global__ void resolvePairsPassA(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t dimension) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    uint32_t link_id = finalBuf[y * dimension + x];
    uint32_t packed_xy = (y << 16) | x;

    atomicCAS(&indexTableSlot0[link_id], 0xFFFFFFFF, packed_xy);
}

__global__ void resolvePairsPassB(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, uint32_t dimension) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    uint32_t link_id = finalBuf[y * dimension + x];
    uint32_t packed_xy = (y << 16) | x;

    if (indexTableSlot0[link_id] != packed_xy) {
        indexTableSlot1[link_id] = packed_xy;
    }
}

__global__ void extractDeltasKernel(uint32_t* finalBuf, uint32_t* indexTableSlot0, uint32_t* indexTableSlot1, short2* outputTex, uint32_t dimension) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dimension || y >= dimension) return;

    uint32_t link_id = finalBuf[y * dimension + x];
    uint32_t my_packed_xy = (y << 16) | x;

    uint32_t slot0 = indexTableSlot0[link_id];
    uint32_t slot1 = indexTableSlot1[link_id];

    uint32_t partner_packed = (slot0 == my_packed_xy) ? slot1 : slot0;

    int partner_x = partner_packed & 0xFFFF;
    int partner_y = partner_packed >> 16;

    int dx = partner_x - (int)x;
    int dy = partner_y - (int)y;

    if (dx > dimension / 2) dx -= dimension;
    if (dx < -(dimension / 2)) dx += dimension;
    if (dy > (dimension / 2)) dy -= dimension;
    if (dy < -(dimension / 2)) dy += dimension;

    outputTex[y * dimension + x] = make_short2(dx, dy);
}

__global__ void resolveSpatialReuse(
    PipelineParams params
) {
    const CommonParams& common = params.common;
    const RestirCommonParams& restir = params.restir;

    const Reservoir& reservoirIn = restir.reservoir;
    const Reservoir& reservoirOut = restir.lastFrameReservoir;

    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= common.w || y >= common.h) return;

    uint32_t self_M;
    uint32_t self_pathLength;
    uint32_t self_rcVertexIndex;
    TechniqueType self_type;

    const uint32_t selfIdx = y * common.w + x;
    reservoirIn.getPathFlags(selfIdx, self_M, self_pathLength, self_rcVertexIndex, self_type);

    const float3 self_F = reservoirIn.getF_globalLoad(selfIdx);
    const float self_W = reservoirIn.getW_globalLoad(selfIdx);
    const float self_phat = targetFunction(self_F);
    const uint32_t self_seed = reservoirIn.getSeed_notstreaming(selfIdx);

    uint32_t self_rcPrimID;
    float2 self_rcBary;
    float3 self_rcWi;
    float3 self_rcRadiance;
    reservoirIn.getRcVertexGeometry_globalLoad(selfIdx, self_rcPrimID, self_rcBary, self_rcWi, self_rcRadiance);

    float self_cachedJacobian;
    float self_cachedNee;
    reservoirIn.getCachedValues_globalLoad(selfIdx, self_cachedJacobian, self_cachedNee);

    const uint32_t texIdx = restir.currentSpatialReuseIndex;
    const int2 neighborCoord = get_paired_neighbor(
        make_int2(x, y), texIdx, common.frame_index,
        restir.reuseTextureSizes[texIdx],
        make_int2(common.w, common.h),
        restir.reuseTextures[texIdx]
    );

    bool neighborUsable = false;
    float w_neighbor = 0.0f;
    float m_canonical = 1.0f;

    float3 shifted_F = f3(0.0f);
    float  shifted_phat = 0.0f;
    float  shifted_newCachedJacobian = 0.0f;

    uint32_t nb_M = 0u, nb_pathLength = 0u, nb_rcVertexIndex = 0u;
    TechniqueType nb_type = 0u;
    uint32_t nb_seed = 0u;
    uint32_t nb_rcPrimID = 0u; float2 nb_rcBary = f2(0.0f);
    float3 nb_rcWi = f3(0.0f), nb_rcRadiance = f3(0.0f);
    float nb_cachedNee = -1.0f;


    bool hasPartner = false;
    if (neighborCoord.x >= 0 && neighborCoord.y >= 0) {
        const uint32_t partnerIdx = neighborCoord.y * common.w + neighborCoord.x;

        bool fwd_valid; float3 fwd_contribution; float fwd_jacobian, fwd_newCachedJacobian;
        restir.shiftResultBuffer.getResult(selfIdx, fwd_valid, fwd_contribution, fwd_jacobian, fwd_newCachedJacobian);

        bool bwd_valid; float3 bwd_contribution; float bwd_jacobian, bwd_newCachedJacobian;
        restir.shiftResultBuffer.getResult(partnerIdx, bwd_valid, bwd_contribution, bwd_jacobian, bwd_newCachedJacobian);

        reservoirIn.getPathFlags(partnerIdx, nb_M, nb_pathLength, nb_rcVertexIndex, nb_type);
        const float3 nb_F = reservoirIn.getF_globalLoad(partnerIdx);
        const float  nb_W = reservoirIn.getW_globalLoad(partnerIdx);
        const float  nb_phat = targetFunction(nb_F);

        if (fwd_valid && nb_M > 0u) {
            shifted_F = fwd_contribution;
            shifted_phat = targetFunction(shifted_F);
            shifted_newCachedJacobian = fwd_newCachedJacobian;

            const float denom = (float)self_M * shifted_phat * fwd_jacobian + (float)nb_M * nb_phat;
            if (denom > 0.0f) {
                const float m_neighbor = ((float)nb_M * nb_phat) / denom;
                w_neighbor     = m_neighbor * shifted_phat * nb_W * fwd_jacobian;
                neighborUsable = true;

                // Identity only needed if this sample ends up winning.
                nb_seed = reservoirIn.getSeed_notstreaming(partnerIdx);
                reservoirIn.getRcVertexGeometry_globalLoad(partnerIdx, nb_rcPrimID,
                                                           nb_rcBary, nb_rcWi, nb_rcRadiance);
                if (needNeePDF(nb_type)) {
                    nb_cachedNee = reservoirIn.getCachedNEE_globalLoad(partnerIdx);
                }
            }
        }

        if (bwd_valid && nb_M > 0u) {
            const float bwd_phat = targetFunction(bwd_contribution);
            const float denom = (float)self_M * self_phat
                              + (float)nb_M   * bwd_phat * bwd_jacobian;
            if (denom > 0.0f) {
                m_canonical = ((float)self_M * self_phat) / denom;
            }
        }

        const float w_canonical = m_canonical * self_phat * self_W;
        const float w_sum       = w_canonical + w_neighbor;
        const uint32_t new_M    = min(self_M + (hasPartner ? nb_M : 0u), 255u);
    }


}