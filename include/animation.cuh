#pragma once

#include "objects.cuh"

struct __align__(32) LinearCameraAnimation {
    float3 origin;
    float3 initRot;
    float3 dxyz;
    float3 dxyzrot;

    LinearCameraAnimation(float3 o, float3 i, float3 dpos, float3 drot) : origin(o), initRot(i), dxyz(dpos), dxyzrot(drot) {}

    __host__ inline void update(Camera& cam, uint32_t frame) {
        cam.cameraOrigin = origin + dxyz * (float) frame;

        cam.xRot = initRot.x + dxyzrot.x * (float) frame;
        cam.yRot = initRot.y + dxyzrot.y * (float) frame;
        cam.zRot = initRot.z + dxyzrot.z * (float) frame;

        cam.preCompute();
    }
};

struct __align__(32) TurntableCameraAnimation {
    float3 target;        // Where the camera looks
    float radius;         // Distance from target
    float orbitSpeed;     // Radians per frame
    float startAngle;     // Starting rotation
    float height;
    
    TurntableCameraAnimation(float3 targetPos, float orbitRadius, float speedDeg, float startAngleDeg, float camHeight) 
        : target(targetPos), radius(orbitRadius), height(camHeight) 
    {
        orbitSpeed = speedDeg * (3.14159265f / 180.0f);
        startAngle = startAngleDeg * (3.14159265f / 180.0f);
    }

    __host__ void update(Camera& cam, uint32_t frame) {
        float angle = startAngle + (float)frame * orbitSpeed;

        float3 pos = make_float3(
            target.x + radius * cosf(angle),
            height,
            target.z + radius * sinf(angle)
        );

        cam.cameraOrigin = pos;

        float3 dir = normalize(target - pos);

        // FIXED: The negative signs correctly invert the Euler extraction
        // so that your -Z forward vector correctly aligns with 'dir'
        cam.yRot = atan2f(-dir.x, -dir.z); 
        cam.xRot = asinf(dir.y);
        cam.zRot = 0.0f;
        
        cam.preCompute();
    }
};