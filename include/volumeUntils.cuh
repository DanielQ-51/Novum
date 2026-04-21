#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include "util.cuh"
#include "objects.cuh"
#include <curand_kernel.h>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/math/Ray.h>
#include <nanovdb/math/HDDA.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/cuda/DeviceBuffer.h>

Ray toNovumRay(const nanovdb::Ray<float> r) {
    const auto start = r.start();
    const auto dir = r.dir();
    return Ray(
        f4(start[0], start[1], start[2]), 
        f4(dir[0], dir[1], dir[2])
    );
}

nanovdb::Ray<float> toNanoVDB(Ray r) {
    return nanovdb::Ray<float>(
        nanovdb::Vec3f(r.origin.x, r.origin.y, r.origin.z), 
        nanovdb::Vec3f(r.direction.x, r.direction.y, r.direction.z)
    );
}