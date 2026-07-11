#pragma once

#include "objects.cuh"
#include "sampling.cuh"

struct SceneContext
{
    const Material* __restrict__ materials;
    const float4* __restrict__ textures;
    const BVHnode* __restrict__ BVH;
    const int2* __restrict__ BVHindices;
    const Vertices* __restrict__ vertices; 
    const Triangle* __restrict__ scene; 
    const Triangle* __restrict__ lights;
    const Volume* __restrict__ volumes;
    LightSampler lightSampler;
    int vertNum; 
    int triNum;
    int lightNum;
};

struct BVHContext
{
    const BVHnode* __restrict__ BVH;
    const int2* __restrict__ BVHindices;
    const Vertices* __restrict__ vertices;
    const Triangle* __restrict__ scene;
    const Material* __restrict__ materials;
    const Volume* __restrict__ volumes;
};

struct ShadeContext
{
    const Material* __restrict__ materials;
    const float4* __restrict__ textures;
    const Triangle* __restrict__ lights;
    const Triangle* __restrict__ scene;
    const Vertices* __restrict__ vertices; 
    LightSampler lightSampler;
    int lightNum;
    uint32_t triNum;
};

__host__ inline BVHContext getBVHContext(const SceneContext& sc) {
    BVHContext ctx;
    ctx.BVH        = sc.BVH;
    ctx.BVHindices = sc.BVHindices;
    ctx.vertices   = sc.vertices;
    ctx.scene      = sc.scene;
    ctx.materials  = sc.materials;
    ctx.volumes  = sc.volumes;
    return ctx;
}

__host__ inline ShadeContext getShadeContext(const SceneContext& sc) {
    ShadeContext ctx;
    ctx.materials = sc.materials;
    ctx.textures  = sc.textures;
    ctx.lights    = sc.lights;
    ctx.scene     = sc.scene;
    ctx.vertices  = sc.vertices;
    ctx.lightNum  = sc.lightNum;
    ctx.lightSampler = sc.lightSampler;
    ctx.triNum = sc.triNum;
    return ctx;
}