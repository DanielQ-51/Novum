#pragma once
#include "deviceCode.cuh"
#include "fastIntegrators.cuh"
#include "objects.cuh"
#include "util.cuh"
#include "volumeRendering.cuh"
#include "sceneContexts.cuh"
#include <chrono>
#include <iostream>
#include <exception>
#include <set>
#include <iomanip>
#include "imageUtil.cuh"
#include <fstream>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <iomanip>

using namespace std;

void computeInfoForBVH(Vertices& vertices, vector<Triangle>& mesh, vector<Volume>& volumes, vector<float4>& centroids, 
    vector<float4>& AABBmins, vector<float4>& AABBmaxes, vector<int>& primTypes, vector<int>& originalIndices)
{
    for (int i = 0; i < mesh.size(); i++)
    {
        Triangle tri = mesh[i];
        float4 a = vertices.positions[tri.aInd];
        float4 b = vertices.positions[tri.bInd];
        float4 c = vertices.positions[tri.cInd];
        centroids.push_back(f4((a.x+b.x+c.x)/3.0f,(a.y+b.y+c.y)/3.0f,(a.z+b.z+c.z)/3.0f));

        // Compute AABB min
        float4 minPos = f4(
            fminf(fminf(a.x, b.x), c.x),
            fminf(fminf(a.y, b.y), c.y),
            fminf(fminf(a.z, b.z), c.z)
        ) - f4(0.000001f);
        AABBmins.push_back(minPos);

        // Compute AABB max
        float4 maxPos = f4(
            fmaxf(fmaxf(a.x, b.x), c.x),
            fmaxf(fmaxf(a.y, b.y), c.y),
            fmaxf(fmaxf(a.z, b.z), c.z)
        ) + f4(0.000001f);
        AABBmaxes.push_back(maxPos);

        primTypes.push_back(TYPE_TRIANGLE);
        originalIndices.push_back(i);
    }

    for (int i = 0; i < volumes.size(); i++)
    {
        Volume& vol = volumes[i];
        
        float4 minPos = vol.aabbMIN;
        float4 maxPos = vol.aabbMAX;
        float4 centroid = f4((minPos.x + maxPos.x) * 0.5f,
                             (minPos.y + maxPos.y) * 0.5f,
                             (minPos.z + maxPos.z) * 0.5f);

        centroids.push_back(centroid);
        AABBmins.push_back(minPos);
        AABBmaxes.push_back(maxPos);

        // TRACKING: This is Volume #i
        primTypes.push_back(TYPE_VOLUME);
        originalIndices.push_back(i);
    }
}

int partitionPrimitives(vector<int>& indices, vector<float4>& centroids, int start, int end, int axis, float splitPos)
{
    int mid = start;
    for (int i = start; i < end; i++)
    {
        float4 c = centroids[indices[i]];
        if (getFloat4Component(c, axis) < splitPos)
        {
            swap(indices[i], indices[mid]);
            mid++;
        }
    }
    return mid;
}

void SAH( vector<int>& indices, vector<float4>& centroids, vector<float4>& AABBmins, vector<float4>& AABBmaxes, int start, 
    int end, int& axis, float4 minBound, float4 maxBound, float& splitPos, float& minCost, int& backup)
{
    const int numBuckets = 8;
    struct Bucket { float4 min, max; int count; };
    Bucket buckets[numBuckets];
    for (int i = 0; i < numBuckets; i++) 
    {
        buckets[i].min = f4(FLT_MAX);
        buckets[i].max = f4(-FLT_MAX);
        buckets[i].count = 0;
    }

    for (int i = start; i < end; i++) 
    {
        int idx = indices[i];
        float c = getFloat4Component(centroids[idx] , axis);
        int b = int(numBuckets * (c - getFloat4Component(minBound , axis)) / (getFloat4Component(maxBound , axis) - getFloat4Component(minBound , axis)));
        b = clamp(b, 0, numBuckets - 1);
        buckets[b].count++;
        buckets[b].min = fminf4(buckets[b].min, AABBmins[idx]);
        buckets[b].max = fmaxf4(buckets[b].max, AABBmaxes[idx]);
    }

    minCost = FLT_MAX;
    int bestSplit = -1;
    
    for (int i = 1; i < numBuckets; i++) 
    {
        float4 leftMin = buckets[0].min;
        float4 leftMax = buckets[0].max;
        int leftCount = buckets[0].count;
        for (int j = 1; j < i; j++) {
            leftMin = fminf4(leftMin, buckets[j].min);
            leftMax = fmaxf4(leftMax, buckets[j].max);
            leftCount += buckets[j].count;
        }

        float4 rightMin = buckets[i].min;
        float4 rightMax = buckets[i].max;
        int rightCount = buckets[i].count;
        for (int j = i; j < numBuckets; j++) {
            rightMin = fminf4(rightMin, buckets[j].min);
            rightMax = fmaxf4(rightMax, buckets[j].max);
            rightCount += buckets[j].count;
        }

        float cost = 1.0f + (leftCount * surfaceArea(leftMin, leftMax) + rightCount * surfaceArea(rightMin, rightMax))
                           / surfaceArea(minBound, maxBound);
        if (cost < minCost && (leftCount > 0 && rightCount > 0)) {
            minCost = cost;
            bestSplit = i;
        }
    }

    if (bestSplit == -1)
    {
        //cout << "no best split" << endl;

        int mid = (start + end) / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + end,
            [&](int a, int b) { return getFloat4Component(centroids[a], axis) < getFloat4Component(centroids[b], axis); });
        
        splitPos = getFloat4Component(centroids[indices[mid]], axis);
    }
    else
        splitPos = getFloat4Component(minBound , axis) + (getFloat4Component(maxBound , axis) - getFloat4Component(minBound , axis)) * (float(bestSplit) / float(numBuckets));
}

int buildBVH(vector<BVHnode>& nodes, vector<int>& indices, vector<float4>& centroids, 
    vector<float4>& AABBmins, vector<float4>& AABBmaxes, int start, int end, 
    int maxLeafSize, int& largestLeaf, int& backup)
{
    int nodeIndex = nodes.size();
    nodes.push_back(BVHnode());

    float4 minBound = AABBmins[indices[start]];
    float4 maxBound = AABBmaxes[indices[start]];
    for (int i = start; i < end; i++)
    {
        int idx = indices[i];
        minBound = fminf4(minBound, AABBmins[idx]);
        maxBound = fmaxf4(maxBound, AABBmaxes[idx]);
    }
    nodes[nodeIndex].aabbMIN = minBound;
    nodes[nodeIndex].aabbMAX = maxBound;

    int primCount = end - start;

    if (primCount <= maxLeafSize) {
        nodes[nodeIndex].first = start;
        nodes[nodeIndex].primCount = primCount;
        nodes[nodeIndex].left = nodes[nodeIndex].right = -1;
        largestLeaf = max(primCount, largestLeaf);
        return nodeIndex;
    }

    /*
    float xdiff = maxBound.x - minBound.x;
    float ydiff = maxBound.y - minBound.y;
    float zdiff = maxBound.z - minBound.z;
    int axis = 0;
    if (ydiff > xdiff && ydiff > zdiff)
        axis = 1;
    else if (zdiff > xdiff && zdiff > ydiff)
        axis = 2; */

    float cost = FLT_MAX;
    int axis = -1;
    float splitPos = 0.0f;

    // Loop through all 3 axes
    for (int currentAxis = 0; currentAxis < 3; ++currentAxis) 
    {
        float currentSplitPos;
        float currentCost;
        
        // Pass currentAxis into your SAH function instead of the longest axis
        SAH(indices, centroids, AABBmins, AABBmaxes, start, end, currentAxis, 
            minBound, maxBound, currentSplitPos, currentCost, backup);

        // Keep track of the best axis we've seen so far
        if (currentCost < cost) 
        {
            cost = currentCost;
            axis = currentAxis;
            splitPos = currentSplitPos;
        }
    }

    /*if (primCount <= maxLeafSize || cost >= primCount * 1) {
        // Force a leaf
        nodes[nodeIndex].first = start;
        nodes[nodeIndex].primCount = primCount;
        nodes[nodeIndex].left = nodes[nodeIndex].right = -1;
        largestLeaf = max(primCount, largestLeaf);
        return nodeIndex;
    }*/
    int mid;
    
    int numLeft = 0;
    for (int i = start; i < end; i++) {
        if (getFloat4Component(centroids[indices[i]], axis) < splitPos)
            numLeft++;
    }
    //cout << "1numLeft: " << numLeft << endl;
    if (numLeft > 0 && numLeft < primCount)
        mid = partitionPrimitives(indices, centroids, start, end, axis, splitPos);
    else
    {
        //cout << "midpoint backup" << endl;
        float sum = 0.0f;
        backup++;
        for (int i = start; i < end; i++)
        {
            sum += getFloat4Component(centroids[indices[i]], axis);
        }
        splitPos = sum/primCount;
    }
    //cout << "SECOND failed split at " << splitPos << " on the " << axis << " numbered axis" << endl; 
    
    numLeft = 0;
    for (int i = start; i < end; i++) {
        if (getFloat4Component(centroids[indices[i]], axis) < splitPos)
            numLeft++;
    }
    //cout << "2numLeft: " << numLeft << endl;
    if (numLeft > 0 && numLeft < primCount)
        mid = partitionPrimitives(indices, centroids, start, end, axis, splitPos);
    else
    {
        nodes[nodeIndex].first = start;
        nodes[nodeIndex].primCount = primCount;
        nodes[nodeIndex].left = nodes[nodeIndex].right = -1;
        largestLeaf = max(primCount, largestLeaf);
        //cout << start << " " << mid << " " << end << endl;
        //cout << "force leaf " << primCount << endl;
        return nodeIndex;
    }
    //cout << start << " " << mid << " " << end << endl;
        //cout << "Split at " << splitPos << " on the " << axis << " numbered axis" << endl; 
    //cout << "HELLLLLLLLLLLLLLLLLOOOOOOOOOOO " << endl;
    nodes[nodeIndex].left  = buildBVH(nodes, indices, centroids, AABBmins, AABBmaxes, start, mid, maxLeafSize, largestLeaf, backup);
    nodes[nodeIndex].right = buildBVH(nodes, indices, centroids, AABBmins, AABBmaxes, mid, end, maxLeafSize, largestLeaf, backup);

    nodes[nodeIndex].primCount = 0;
    nodes[nodeIndex].first = -1;
    
    return nodeIndex;
}