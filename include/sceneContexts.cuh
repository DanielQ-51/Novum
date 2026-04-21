#pragma once

#include "objects.cuh"
#include "sampling.cuh"

struct SceneContext
{
    Material* materials;
    float4* textures;
    BVHnode* BVH;
    int2* BVHindices;
    Vertices* vertices; 
    Triangle* scene; 
    Triangle* lights;
    Volume* volumes;
    LightSampler lightSampler;
    int vertNum; 
    int triNum;
    int lightNum;
};

struct BVHContext
{
    BVHnode* BVH;
    int2* BVHindices;
    Vertices* vertices;
    Triangle* scene;
    Material* materials;
    Volume* volumes;
};

struct ShadeContext
{
    Material* materials;
    float4* textures;
    Triangle* lights;
    Triangle* scene;
    Vertices* vertices; 
    LightSampler lightSampler;
    int lightNum;
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
    return ctx;
}