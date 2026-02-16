
#include "deviceCode.cuh"
#include "objects.cuh"
#include "util.cuh"
#include <vector>
#include <chrono>
#include <iostream>
#include <set>
#include <iomanip>
#include "imageUtil.cuh"
#include <fstream>
#include <cuda_fp16.h>
#include <string>

using namespace std;

void readObjSimple(string filename, vector<float4>& points, vector<float4>& normals, vector<float4>& colors, vector<float2>& uvs,vector<Triangle>& mesh, 
    vector<Triangle>& lights, float4 c, float4 e, int materialID, float4 offset = f4(0.0f));

void computeInfoForBVH(Vertices& vertices, vector<Triangle>& mesh, vector<float4>& centroids, 
    vector<float4>& AABBmins, vector<float4>& AABBmaxes)
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
    const int numBuckets = 12;
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
    float xdiff = maxBound.x - minBound.x;
    float ydiff = maxBound.y - minBound.y;
    float zdiff = maxBound.z - minBound.z;
    int axis = 0;
    if (ydiff > xdiff && ydiff > zdiff)
        axis = 1;
    else if (zdiff > xdiff && zdiff > ydiff)
        axis = 2;

    //int bestSplit = 0;
    float splitPos;
    float cost = 0;
    SAH(indices, centroids, AABBmins, AABBmaxes, start, end, axis, minBound, maxBound, splitPos, cost, backup);

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
    if (numLeft > 0 && numLeft < (primCount - 1))
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
    if (numLeft > 0 && numLeft < (primCount - 1))
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

int initRender(string configPath, int renderNumber, string animatedObjPath = "invalid")
{
    RenderConfig config;
    loadConfig(configPath, config);

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::cout << "Began render number " << renderNumber << ": \"" << config.name << "\"\n\n";

    std::cout << "Current time: " 
              << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S")
              << "\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    int w = config.width;
    int h = config.height;

    int integratorChoice = matchIntegrator(config.integratorType); 

    int sampleCount = config.sampleCount;
    int maxDepth = config.maxDepth;

    int eyePathDepth = config.bdptEyeDepth;
    int lightPathDepth = config.bdptLightDepth;
    
    int maxLeafSize = config.bvhLeafSize;
    float VCMMergeConstant = config.vcmMergeConst;
    float VCMInitialMergeRadiusMultiplier = config.vcmInitialMergeRadiusMultiplier;

    std::cout << VCMMergeConstant << " and " << VCMInitialMergeRadiusMultiplier << std::endl;

    Camera camera;
    if (config.pinholeCamera)
        camera = Camera::Pinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov);
    else
        camera = Camera::NotPinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov, 
            config.camApeture, config.camFocalDist);

    Image image = Image(w, h);
    image.postProcess = config.postProcess;

    if (integratorChoice == UNIDIRECTIONAL)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max depth of " << 
            maxDepth << ".\nIntegrating with Naive + NEE Unidirectional MIS." << 
            endl << endl;
    }
    else if (integratorChoice == BIDIRECTIONAL)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max eye depth of " << 
            eyePathDepth << ", and a max light depth of " << 
            lightPathDepth << ".\nIntegrating with Bidirectional." << 
            endl << endl;
    }
    if (integratorChoice == NAIVE_UNIDIRECTIONAL)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max depth of " << 
            maxDepth << ".\nIntergating with Naive Unidirectional" << 
            endl << endl;
    }
    else if (integratorChoice == VCM)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max eye depth of " << 
            eyePathDepth << ", and a max light depth of " << 
            lightPathDepth << ".\nIntegrating with Vertex Connection and Merging, with alpha parameter " <<
            VCMMergeConstant << " and initial merge radius multiplier " <<
            VCMInitialMergeRadiusMultiplier << 
            endl << endl;
    }
    else if (integratorChoice == SPPM)
    {
        cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount << " samples per pixel, and a maximum leaf size of " <<
            maxLeafSize << " primitives, with a max eye depth of " << 
            eyePathDepth << ", and a max light depth of " << 
            lightPathDepth << ".\nIntegrating with Stochastic Progressive Photon Mapping, with alpha parameter " <<
            VCMMergeConstant << " and initial merge radius multiplier " <<
            VCMInitialMergeRadiusMultiplier << 
            endl << endl;

        // to turn the vcm integrator into an only merging integrator
        config.bdptConnection = false;
        config.bdptNaive = false;
        config.bdptNee = false;
        config.bdptLightTrace = false;
        config.bdptDoMis = false;
        config.vcmDoMerge = true;
        config.doSPPM = true;
    }

    updateConstants(config);

    float4* out_colors;
    cudaMalloc(&out_colors, w * h * sizeof(float4));
    cudaMemset(out_colors, 0, w * h * sizeof(float4));

    float4* out_overlay;
    cudaMalloc(&out_overlay, w * h * sizeof(float4));
    cudaMemset(out_overlay, 0, w * h * sizeof(float4));

    Vertices vertices;
    vector<float4> points;
    vector<float4> normals;
    vector<float4> colors; // unused now
    vector<float2> uvs;
    vector<Triangle> mesh;
    vector<Triangle> lightsvec;
    vector<BVHnode> bvhvec;

    vector<float4> centroids;
    vector<float4> minboxes;
    vector<float4> maxboxes;

    vector<Material> mats;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading Textures
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    vector<Image> images;
    vector<float4> pixels;
    vector<int> widths;
    vector<int> heights;
    vector<int> startIndices;
    int currentStartIndex = 0;

    images.push_back(loadBMPToImage("textures/enkidutexture.bmp", false));
    images.push_back(loadBMPToImage("textures/enkiduchibitexture.bmp", false));
    images.push_back(loadBMPToImage("textures/leaftex2.bmp", false));
    images.push_back(loadBMPToImage("textures/leafautumn.bmp", false));
    images.push_back(loadBMPToImage("textures/wood.bmp", false));
    images.push_back(loadBMPToImage("textures/wall.bmp", false));

    for (Image i : images)
    {
        vector<float4> pix = i.data();

        pixels.insert(pixels.end(), pix.begin(), pix.end());

        widths.push_back(i.width);
        heights.push_back(i.height);
        startIndices.push_back(currentStartIndex);
        currentStartIndex += i.width*i.height;
    }

    float4* textures_d;

    cudaMalloc(&textures_d, pixels.size() * sizeof(float4));
    cudaMemcpy(textures_d, pixels.data(), pixels.size() * sizeof(float4), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Creating Materials
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    Material wood = Material::Leaf(startIndices[4], widths[4], heights[4], 1.5f, 0.3f, f4(), 0.00f);
    Material wall = Material::DiffuseTextured(startIndices[5], widths[5], heights[5]);
    Material lambertTextured = Material::DiffuseTextured(startIndices[0], widths[0], heights[0]);
    Material lambert2Textured = Material::DiffuseTextured(startIndices[1], widths[1], heights[1]);

    Material lambertBlue = Material::Diffuse(f4(0.4f,0.4f,0.8f));
    Material lambertGrey = Material::Diffuse(f4(0.8f,0.8f,0.8f));
    Material lambertGreyBlue = Material::Diffuse(f4(0.6f,0.6f,0.7f));
    Material lambertWhite = Material::Diffuse(f4(0.9f,0.9f,0.9f));
    Material lambertGreen = Material::Diffuse(f4(0.2f,0.6f,0.6f));
    Material lambertRed = Material::Diffuse(f4(0.90f,0.1f,0.1f));
    Material lambertVeryGreen = Material::Diffuse(f4(0.1f,0.9f,0.1f));
    Material lambertBLACK = Material::Diffuse(f4(0.0f,0.0f,0.0f));
    Material lambert95 = Material::Diffuse(f4(0.95f,0.95f,0.95f));
    Material lambert50 = Material::Diffuse(f4(0.5f,0.5f,0.5f));

    float4 eta_steel = f4(0.14f, 0.16f, 0.13f, 1.0f);   // real part (R,G,B,alpha)
    float4 k_steel   = f4(4.1f, 2.3f, 3.1f, 1.0f);     // imaginary part (absorption)


    float4 eta_gold = f4(0.17f, 0.35f, 1.5f);  // real part of refractive index
    float4 k_gold   = f4(3.1f, 2.7f, 1.9f);   // imaginary part, absorption
    float roughness_polished = 0.05f;  
    float roughness_rough = 0.15f;  
    float roughness_rougher = 0.35f;  

    Material gold = Material::Metal(eta_gold, eta_gold, roughness_polished);
    Material gold15 = Material::Metal(eta_gold, eta_gold, roughness_rough);
    Material steel = Material::Metal(eta_steel, eta_steel, roughness_rough);
    Material steelSmooth = Material::Metal(eta_steel, eta_steel, roughness_polished);
    Material steel25 = Material::Metal(eta_steel, eta_steel, 0.25f);
    Material roughSteel = Material::Metal(eta_steel, eta_steel, roughness_rougher);

    float ior = 1.5f;

    Material glass = Material::SmoothDielectric(ior, f4(0.0f), 1);
    Material diamond = Material::SmoothDielectric(2.42f, f4(0.0f), 1);

    Material water = Material::SmoothDielectric(1.333f, f4(), 2);
    Material tea = Material::SmoothDielectric(1.333f, 2.5f * f4(0.180f, 1.5f, 2.996f), 2);

    Material ice = Material::SmoothDielectric(1.31f, f4(0.2f), 0);

    Material air = Material::SmoothDielectric(1.0f, f4(0.0f), 99);

    //Material leaf = Material::Leaf(1.5f, 0.6f, f4(0.8f, 0.25f, 0.28f), 0.2f);
    Material leaf = Material::Leaf(startIndices[2], widths[2], heights[2], 1.5f, 0.10f, f4(0.22f, 0.75f, 0.28f), 0.15f);
    Material leafAutumn = Material::Leaf(startIndices[3], widths[3], heights[3], 1.5f, 0.8f, f4(0.22f, 0.75f, 0.28f), 0.6f);
    Material canopy = Material::Leaf(startIndices[2], widths[2], heights[2], 1.5f, 0.9f, f4(0.22f, 0.75f, 0.28f), 0.7f);
    Material leafStem = Material::Diffuse(f4(0.90f, 0.9f, 0.83f));
    Material sky = Material::Diffuse(f4(0.4f, 0.4f, 1.00f));

    Material mirror = Material::Mirror();
    Material thinGlass = Material::ThinDielectric(1.5f);

    mats.push_back(air); // index 0

    mats.push_back(lambertBlue); // index 1
    mats.push_back(lambertWhite); // index 2
    mats.push_back(lambertGreen); // index 3
    mats.push_back(gold); // index 4
    mats.push_back(glass); // index 5
    mats.push_back(lambertRed); // index 6
    mats.push_back(steel); // index 7
    mats.push_back(tea); // index 8
    mats.push_back(ice); // index 9
    mats.push_back(water); // index 10
    mats.push_back(lambertTextured); // index 11
    mats.push_back(lambert2Textured); // index 12
    mats.push_back(leaf); // index 13
    mats.push_back(leafStem); // index 14
    mats.push_back(sky); // index 15
    mats.push_back(leafAutumn); // index 16
    mats.push_back(lambertGrey); // index 17
    mats.push_back(diamond); // index 18
    mats.push_back(mirror); // index 19
    mats.push_back(lambertBLACK); // index 20
    mats.push_back(lambert95); // index 21
    mats.push_back(lambert50); // index 22
    mats.push_back(lambertVeryGreen); // index 23
    mats.push_back(wood); // index 24
    mats.push_back(lambertGreyBlue); // index 25
    mats.push_back(wall); // index 26
    mats.push_back(roughSteel); // index 27
    mats.push_back(thinGlass); // index 28
    mats.push_back(steelSmooth); // index 29
    mats.push_back(steel25); // index 30
    mats.push_back(gold15); // index 31

    Material* mats_d;

    cudaMalloc(&mats_d, mats.size() * sizeof(Material));
    cudaMemcpy(mats_d, mats.data(), mats.size() * sizeof(Material), cudaMemcpyHostToDevice);

    for (MeshConfig c : config.meshes)
    {
        //if (lengthSquared(c.emissionColor) > 0.0f)
        //    readObjSimple(c.path, points, normals, colors, uvs, mesh, lightsvec, f4(), 
        //        c.emissionMultiplier * c.emissionColor, c.materialID, f4(0.0f, -0.01f * renderNumber, 0.0f));
        //else
            readObjSimple(c.path, points, normals, colors, uvs, mesh, lightsvec, f4(), 
                c.emissionMultiplier * c.emissionColor, c.materialID);
    }

    if (false && animatedObjPath != "invalid" && config.name == "watersim")
    {
        readObjSimple(animatedObjPath, points, normals, colors, uvs, mesh, lightsvec, f4(), 
                f4(), 10);
    }

    Vertices* verts;
    Triangle* scene;
    Triangle* lights;

    cudaMalloc(&verts,  sizeof(Vertices));
    Vertices temp;

    cudaMalloc(&temp.positions, sizeof(float4) * points.size());
    cudaMalloc(&temp.normals, sizeof(float4) * normals.size());
    cudaMalloc(&temp.colors,  sizeof(float4) * colors.size());
    cudaMalloc(&temp.uvs,  sizeof(float2) * uvs.size());

    cudaMemcpy(temp.positions, points.data(), points.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.normals, normals.data(), normals.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.colors, colors.data(), colors.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(verts, &temp, sizeof(Vertices), cudaMemcpyHostToDevice);

    vector<int> indvec(mesh.size());
    for (int i = 0; i < mesh.size(); i++) indvec[i] = i;

    if (mesh.size() == 0) {
        cout << "Error: No triangles loaded." << endl;
        return 1;
    }
    cout << "scene data read. There are " << mesh.size() << " Triangles and " << lightsvec.size() << " +1 lights" << endl;

    auto afterRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterRead = afterRead - start;
    std::cout << "Scene construction took: " << elapsed_seconds_afterRead.count() << " seconds" << std::endl << endl;

    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Computing BVH (on CPU)
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    vertices.positions = points.data();
    vertices.normals   = normals.data();
    vertices.colors    = colors.data();
    vertices.uvs    = uvs.data();
    computeInfoForBVH(vertices, mesh, centroids, minboxes, maxboxes);

    cout << "BVH data computed" << endl;

    int failcount = 0;
    int backupCt = 0;
    int startNode = buildBVH(bvhvec, indvec, centroids, minboxes, maxboxes, 0, mesh.size(), maxLeafSize, failcount, backupCt);

    BVHnode root = bvhvec[0];
    float4 sceneCenter = (root.aabbMAX + root.aabbMIN) * 0.5f;
    float sceneRadius = length(root.aabbMAX - sceneCenter) + 0.01f;
    float4 sceneMin = root.aabbMIN;

    cout << "BVH built. Scene radius is " << sceneRadius << "." << endl;
    cout << "Largest leaf is size: " << failcount << "." << " Backup was called "<< backupCt << " times." << endl;
    //printBVH(bvhvec, indvec);
    printBVHSummary(bvhvec);

    auto afterBVH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterBVH = afterBVH - afterRead;
    std::cout << "BVH construction took: " << elapsed_seconds_afterBVH.count() << " seconds" << std::endl << endl;
    
    BVHnode* BVH;
    int* BVHindices;
    
    cudaMalloc(&scene, mesh.size() * sizeof(Triangle));
    cudaMalloc(&lights, lightsvec.size() * sizeof(Triangle));
    cudaMalloc(&BVH, bvhvec.size() * sizeof(BVHnode));
    cudaMalloc(&BVHindices, indvec.size() * sizeof(int));

    cudaMemcpy(scene, mesh.data(), mesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(lights, lightsvec.data(), lightsvec.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    cudaMemcpy(BVH, bvhvec.data(), bvhvec.size() * sizeof(BVHnode), cudaMemcpyHostToDevice);
    cudaMemcpy(BVHindices, indvec.data(), indvec.size() * sizeof(int), cudaMemcpyHostToDevice);

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Additional setup according to which integrator is used
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    
    if (integratorChoice == UNIDIRECTIONAL)
        launch_unidirectional(maxDepth, camera, mats_d, textures_d, BVH, BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, true, w, h, out_colors);
    else if (integratorChoice == BIDIRECTIONAL)
    {
        int totalEyePathVertices = w * h * eyePathDepth;
        int totalLightPathVertices = w * h * lightPathDepth;

        PathVertices* eyePath_d;
        cudaMalloc(&eyePath_d, sizeof(PathVertices));

        PathVertices tempPaths;

        cudaMalloc(&tempPaths.materialID, sizeof(int) * totalEyePathVertices);
        cudaMalloc(&tempPaths.pt,         sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.n,          sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.wo,         sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.beta,       sizeof(float4) * totalEyePathVertices);
        cudaMalloc(&tempPaths.d_vcm,     sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.d_vc,  sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.pdfFwd,  sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.misWeight,  sizeof(float) * totalEyePathVertices);
        cudaMalloc(&tempPaths.isDelta,    sizeof(bool) * totalEyePathVertices);
        cudaMalloc(&tempPaths.lightInd,   sizeof(int) * totalEyePathVertices);
        cudaMalloc(&tempPaths.uv,   sizeof(float2) * totalEyePathVertices);
        cudaMalloc(&tempPaths.backface,   sizeof(bool) * totalEyePathVertices);

        cudaMemset(tempPaths.materialID, 0, sizeof(int) * totalEyePathVertices);
        cudaMemset(tempPaths.pt,         0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.n,          0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.wo,         0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.beta,       0, sizeof(float4) * totalEyePathVertices);
        cudaMemset(tempPaths.d_vcm,     0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.d_vc,  0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.pdfFwd,  0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.misWeight,  0, sizeof(float) * totalEyePathVertices);
        cudaMemset(tempPaths.isDelta,    0, sizeof(bool) * totalEyePathVertices);
        cudaMemset(tempPaths.lightInd,   0, sizeof(int) * totalEyePathVertices);
        cudaMemset(tempPaths.uv,   0, sizeof(float2) * totalEyePathVertices);
        cudaMemset(tempPaths.backface,   0, sizeof(bool) * totalEyePathVertices);

        cudaMemcpy(eyePath_d, &tempPaths, sizeof(PathVertices), cudaMemcpyHostToDevice);

        PathVertices* lightPath_d;
        cudaMalloc(&lightPath_d, sizeof(PathVertices));

        PathVertices tempPaths1;

        cudaMalloc(&tempPaths1.materialID, sizeof(int) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.pt,         sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.n,          sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.wo,         sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.beta,       sizeof(float4) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.d_vcm,     sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.d_vc,  sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.pdfFwd,  sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.misWeight,  sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.isDelta,    sizeof(bool) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.lightInd,   sizeof(int) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.uv,   sizeof(float2) * totalLightPathVertices);
        cudaMalloc(&tempPaths1.backface,   sizeof(bool) * totalLightPathVertices);

        cudaMemset(tempPaths1.materialID, 0, sizeof(int) * totalLightPathVertices);
        cudaMemset(tempPaths1.pt,         0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.n,          0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.wo,         0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.beta,       0, sizeof(float4) * totalLightPathVertices);
        cudaMemset(tempPaths1.d_vcm,     0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.d_vc,  0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.pdfFwd,  0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.misWeight,  0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths1.isDelta,    0, sizeof(bool) * totalLightPathVertices);
        cudaMemset(tempPaths1.lightInd,   0, sizeof(int) * totalLightPathVertices);
        cudaMemset(tempPaths1.uv,   0, sizeof(float2) * totalLightPathVertices);
        cudaMemset(tempPaths1.backface,   0, sizeof(bool) * totalLightPathVertices);

        cudaMemcpy(lightPath_d, &tempPaths1, sizeof(PathVertices), cudaMemcpyHostToDevice);

        launch_bidirectional(eyePathDepth, lightPathDepth, camera, eyePath_d, lightPath_d, mats_d, textures_d, BVH, 
            BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, w, h, 
            sceneCenter, sceneRadius, out_colors, out_overlay, config.postProcess);
        cudaFree(eyePath_d);
        cudaFree(lightPath_d);

        cudaFree(tempPaths.materialID);
        cudaFree(tempPaths.pt);
        cudaFree(tempPaths.n);
        cudaFree(tempPaths.wo);
        cudaFree(tempPaths.beta);
        cudaFree(tempPaths.d_vc);
        cudaFree(tempPaths.isDelta);
        cudaFree(tempPaths.lightInd);
        cudaFree(tempPaths.uv);
        cudaFree(tempPaths.d_vcm);
        cudaFree(tempPaths.backface);
        cudaFree(tempPaths.misWeight);
        cudaFree(tempPaths.pdfFwd);

        cudaFree(tempPaths1.materialID);
        cudaFree(tempPaths1.pt);
        cudaFree(tempPaths1.n);
        cudaFree(tempPaths1.wo);
        cudaFree(tempPaths1.beta);
        cudaFree(tempPaths1.d_vc);
        cudaFree(tempPaths1.isDelta);
        cudaFree(tempPaths1.lightInd);
        cudaFree(tempPaths1.uv);
        cudaFree(tempPaths1.d_vcm);
        cudaFree(tempPaths1.backface);
        cudaFree(tempPaths1.misWeight);
        cudaFree(tempPaths1.pdfFwd);
    }
    else if (integratorChoice == NAIVE_UNIDIRECTIONAL)
    {
        launch_naive_unidirectional(maxDepth, camera, mats_d, textures_d, BVH, BVHindices, verts, points.size(), scene, mesh.size(), lights, lightsvec.size(), sampleCount, true, w, h, out_colors);
    }
    else if (integratorChoice == VCM || integratorChoice == SPPM)
    {
        int totalLightPathVertices = w * h * lightPathDepth;

        //VCMPathVertices* lightPath_d;
        //cudaMalloc(&lightPath_d, sizeof(VCMPathVertices));

        VCMPathVertices tempPaths;

        cudaMalloc(&tempPaths.pos_x, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.pos_y, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.pos_z, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.beta_x, sizeof(half) * totalLightPathVertices);
        cudaMalloc(&tempPaths.beta_y, sizeof(half) * totalLightPathVertices);
        cudaMalloc(&tempPaths.beta_z, sizeof(half) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedNormal, sizeof(unsigned int) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedWo, sizeof(unsigned int) * totalLightPathVertices);
        //cudaMalloc(&tempPaths.packedBeta, sizeof(unsigned int) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedInfo, sizeof(unsigned int) * totalLightPathVertices);
        cudaMalloc(&tempPaths.packedUV, sizeof(half2) * totalLightPathVertices);
        cudaMalloc(&tempPaths.d_vc, sizeof(float) * totalLightPathVertices);
        cudaMalloc(&tempPaths.d_vcm, sizeof(float) * totalLightPathVertices);
        //cudaMalloc(&tempPaths.d_vm, sizeof(float) * totalLightPathVertices);

        cudaMemset(tempPaths.pos_x, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.pos_y, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.pos_z, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.beta_x, 0, sizeof(half) * totalLightPathVertices);
        cudaMemset(tempPaths.beta_y, 0, sizeof(half) * totalLightPathVertices);
        cudaMemset(tempPaths.beta_z, 0, sizeof(half) * totalLightPathVertices);
        cudaMemset(tempPaths.packedNormal, 0, sizeof(unsigned int) * totalLightPathVertices);
        cudaMemset(tempPaths.packedWo, 0, sizeof(unsigned int) * totalLightPathVertices);
        //cudaMemset(tempPaths.packedBeta, 0, sizeof(unsigned int) * totalLightPathVertices);
        cudaMemset(tempPaths.packedInfo, 0, sizeof(unsigned int) * totalLightPathVertices);
        cudaMemset(tempPaths.packedUV, 0, sizeof(half2) * totalLightPathVertices);
        cudaMemset(tempPaths.d_vc, 0, sizeof(float) * totalLightPathVertices);
        cudaMemset(tempPaths.d_vcm, 0, sizeof(float) * totalLightPathVertices);
        //cudaMemset(tempPaths.d_vm, 0, sizeof(float) * totalLightPathVertices);

        //cudaMemcpy(lightPath_d, &tempPaths, sizeof(VCMPathVertices), cudaMemcpyHostToDevice);
        
        int totalPhotons = w * h * lightPathDepth;

        //Photons* photons_d;
        //cudaMalloc(&photons_d, sizeof(Photons));

        Photons tempPhotons;
        cudaMalloc(&tempPhotons.pos_x, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons.pos_y, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons.pos_z, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons.packedWi, sizeof(unsigned int) * totalPhotons);
        //cudaMalloc(&tempPhotons.packedPower, sizeof(unsigned int) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_x, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_y, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.beta_z, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons.packedNormal, sizeof(unsigned int) * totalPhotons);
        //cudaMalloc(&tempPhotons.d_vc, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons.d_vcm, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons.d_vm, sizeof(float) * totalPhotons);

        cudaMemset(tempPhotons.pos_x, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons.pos_y, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons.pos_z, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons.beta_x, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.beta_y, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.beta_z, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons.packedWi, 0, sizeof(unsigned int) * totalPhotons);
        //cudaMemset(tempPhotons.packedPower, 0, sizeof(unsigned int) * totalPhotons);
        cudaMemset(tempPhotons.packedNormal, 0, sizeof(unsigned int) * totalPhotons);
        //cudaMemset(tempPhotons.d_vc, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons.d_vcm, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons.d_vm, 0, sizeof(float) * totalPhotons);

        //cudaMemcpy(photons_d, &tempPhotons, sizeof(Photons), cudaMemcpyHostToDevice);

        //Photons* photons_sorted_d;
        //cudaMalloc(&photons_sorted_d, sizeof(Photons));

        Photons tempPhotons1;
        cudaMalloc(&tempPhotons1.pos_x, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons1.pos_y, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons1.pos_z, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons1.packedWi, sizeof(unsigned int) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_x, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_y, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.beta_z, sizeof(half) * totalPhotons);
        cudaMalloc(&tempPhotons1.packedNormal, sizeof(unsigned int) * totalPhotons);
        //cudaMalloc(&tempPhotons1.d_vc, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons1.d_vcm, sizeof(float) * totalPhotons);
        cudaMalloc(&tempPhotons1.d_vm, sizeof(float) * totalPhotons);

        cudaMemset(tempPhotons1.pos_x, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.pos_y, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.pos_z, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.packedWi, 0, sizeof(unsigned int) * totalPhotons);
        cudaMemset(tempPhotons1.beta_x, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.beta_y, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.beta_z, 0, sizeof(half) * totalPhotons);
        cudaMemset(tempPhotons1.packedNormal, 0, sizeof(unsigned int) * totalPhotons);
        //cudaMemset(tempPhotons1.d_vc, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.d_vcm, 0, sizeof(float) * totalPhotons);
        cudaMemset(tempPhotons1.d_vm, 0, sizeof(float) * totalPhotons);

        //cudaMemcpy(photons_sorted_d, &tempPhotons1, sizeof(Photons), cudaMemcpyHostToDevice);
        

        // launch kernel
        launch_VCM(
            eyePathDepth, lightPathDepth, 
            camera, 
            &tempPaths, 
            &tempPhotons, &tempPhotons1, 
            mats_d, textures_d, 
            BVH, BVHindices, 
            verts, points.size(), 
            scene, mesh.size(), 
            lights, lightsvec.size(), sampleCount, 
            w, h, 
            sceneCenter, sceneRadius, sceneMin,
            out_colors, out_overlay,
            config.postProcess, VCMMergeConstant, VCMInitialMergeRadiusMultiplier
        );

        //cudaFree(lightPath_d);

        cudaFree(tempPaths.pos_x);
        cudaFree(tempPaths.pos_y);
        cudaFree(tempPaths.pos_z);
        cudaFree(tempPaths.packedNormal);
        cudaFree(tempPaths.packedWo);
        cudaFree(tempPaths.beta_x);
        cudaFree(tempPaths.beta_y);
        cudaFree(tempPaths.beta_z);
        cudaFree(tempPaths.packedInfo);
        cudaFree(tempPaths.packedUV);
        cudaFree(tempPaths.d_vc);
        cudaFree(tempPaths.d_vcm);
        //cudaFree(tempPaths.d_vm);

        //cudaFree(photons_d);
        
        cudaFree(tempPhotons.pos_x);
        cudaFree(tempPhotons.pos_y);
        cudaFree(tempPhotons.pos_z);
        cudaFree(tempPhotons.beta_x);
        cudaFree(tempPhotons.beta_y);
        cudaFree(tempPhotons.beta_z);
        cudaFree(tempPhotons.packedWi);
        cudaFree(tempPhotons.packedNormal);
        cudaFree(tempPhotons.d_vcm);
        //cudaFree(tempPhotons.d_vc);
        cudaFree(tempPhotons.d_vm);

        //cudaFree(photons_sorted_d);
        
        cudaFree(tempPhotons1.pos_x);
        cudaFree(tempPhotons1.pos_y);
        cudaFree(tempPhotons1.pos_z);
        cudaFree(tempPhotons1.beta_x);
        cudaFree(tempPhotons1.beta_y);
        cudaFree(tempPhotons1.beta_z);
        cudaFree(tempPhotons1.packedWi);
        cudaFree(tempPhotons1.packedNormal);
        cudaFree(tempPhotons1.d_vcm);
        //cudaFree(tempPhotons1.d_vc);
        cudaFree(tempPhotons1.d_vm);
    }
    

    

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Launch GPU Code - goes to functions in deviceCode.cu
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    float4* host_colors = new float4[w * h];
    cudaMemcpy(host_colors, out_colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    float4* host_overlay = new float4[w * h];
    cudaMemcpy(host_overlay, out_overlay, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < w * h; i++)
    {
        host_colors[i] /= (float)sampleCount;

        if (isnan(host_colors[i].x) || isnan(host_colors[i].y) || isnan(host_colors[i].z)) {
            host_colors[i] = f4(1.0f, 0.0f, 1.0f); // Bright Pink for NaN
        }
        if (isinf(host_colors[i].x) || isinf(host_colors[i].y) || isinf(host_colors[i].z)) {
            host_colors[i] = f4(0.0f, 1.0f, 0.0f); // Bright Green for Inf
        }
    }

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            
            if (host_colors[image.toIndex(i, j)].x < 0 || host_colors[image.toIndex(i, j)].y < 0 || host_colors[image.toIndex(i, j)].z < 0)
                cout << i << ", " << j << " Negative color written: <" << host_colors[image.toIndex(i, j)].x << ", " << host_colors[image.toIndex(i, j)].y << ", " 
                    << host_colors[image.toIndex(i, j)].z << ">"<< endl;

            if (host_overlay[image.toIndex(i, j)].x == 0 && host_overlay[image.toIndex(i, j)].y == 0 && host_overlay[image.toIndex(i, j)].z == 0)
                image.setColor(i, j, host_colors[image.toIndex(i, j)]);
            else
                image.setColor(i, j, host_overlay[image.toIndex(i, j)]);
        }
    }
    
    // memory freeing
    cudaFree(out_colors);
    cudaFree(out_overlay);
    cudaFree(verts);
    cudaFree(scene);
    cudaFree(lights);
    cudaFree(BVH);
    cudaFree(BVHindices);
    cudaFree(mats_d);
    cudaFree(textures_d);

    cudaFree(temp.positions);
    cudaFree(temp.normals);
    cudaFree(temp.colors);
    cudaFree(temp.uvs);
    delete[] host_colors;
    delete[] host_overlay;

    std::string filename = "renders/wateranim3/" + config.name + "" + std::to_string(renderNumber) + ".bmp";
    image.saveImageBMP(filename);
    filename = "render.bmp";
    image.saveImageBMP(filename);
    image.saveImageCSV_MONO(0);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds_render = end - afterBVH;
    std::cout << "Render took: " << elapsed_seconds_render.count() << " seconds" << std::endl << endl;


    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Total Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Total Elapsed time (ms): " << elapsed_ms.count() << " milliseconds" << std::endl;

    return 0;
}

int main ()
{
    string configName = "configs/lightbulbinterreflect.rendertron";
    //for (int i = 0; i <= 150; i++)
    //    initRender(configName, i);

    initRender(configName, 0); 

    for (int i = 125; i <= 250; ++i) {
        char buf[128];
        snprintf(buf, sizeof(buf), "scenedata/watersim/tenbillionobj/wateranim%04d.obj", i);
        initRender(configName, i, buf); 
    }

    cout << "All Renders Finished" << endl;
    return 0;
}


void readObjSimple(string filename, vector<float4>& points, vector<float4>& normals, vector<float4>& colors, vector<float2>& uvs,vector<Triangle>& mesh, 
    vector<Triangle>& lights, float4 c, float4 e, int materialID, float4 offset)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open OBJ file with path " << filename << endl;;
        return;
    }
    int startIndex = points.size();
    int normalStartIndex = normals.size();
    int uvStartIndex = uvs.size();

    int nextLightIndex = lights.size();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 's') continue; // skip comments

        std::istringstream iss(line);
        std::string prefix;
        
        iss >> prefix;
        

        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            float4 p = make_float4(x, y, z, 0.0f) + offset;
            points.push_back(p);
        }
        else if (prefix == "vt") 
        {
            double u, v;
            iss >> u >> v;

            float2 uv = f2(u,1.0f-v);
            uvs.push_back(uv);
        }
        else if (prefix == "vn") {
            double x, y, z;
            iss >> x >> y >> z;

            if (iss.fail() || std::isnan(x) || std::isnan(y) || std::isnan(z)) {
                normals.push_back(make_float4(0.0f, 1.0f, 0.0f, 0.0f)); // Safe dummy default
                continue;
            }
            float4 n = make_float4((float)x, (float)y, (float)z, 0.0f);
    
            float lenSq = lengthSquared(n);
            if (lenSq < 1e-12f) {
                n = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
            }
            normals.push_back(n);
        }
        else if (prefix == "f") {
            vector<string> items;

            string vertinfo;
            vector<int> vertexIndices;
            vector<int> normalIndices;
            vector<int> uvIndices;
            while (iss >> vertinfo) 
            {
                istringstream vss(vertinfo);
                string idx;

                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        vertexIndices.push_back(stoi(idx) - 1);
                }
                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        uvIndices.push_back(stoi(idx) - 1);
                }
                if (getline(vss, idx, '/'))
                {
                    if (!idx.empty())
                        normalIndices.push_back(stoi(idx) - 1);
                }
            }
            bool hasUV = uvIndices.size() == vertexIndices.size();
            bool hasN  = normalIndices.size() == vertexIndices.size();
            int n = vertexIndices.size();
            // Triangulate the polygon as a fan from the first vertex
            for (int i = 1; i < n - 1; ++i) {
                bool isLight = lengthSquared(e) > 0;

                int idx0 = vertexIndices[0] + startIndex;
                int idx1 = vertexIndices[i] + startIndex;
                int idx2 = vertexIndices[i + 1] + startIndex;

                float4 p0 = points[idx0];
                float4 p1 = points[idx1];
                float4 p2 = points[idx2];

                float4 e1 = f4(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
                float4 e2 = f4(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
                
                float4 cp = cross3(e1, e2);
                float areaSq = dot(cp, cp);

                if (areaSq < 1e-18f) {
                    continue; 
                }

                int uv_idx0 = hasUV ? uvIndices[0] + uvStartIndex : -1;
                int uv_idx1 = hasUV ? uvIndices[i] + uvStartIndex : -1;
                int uv_idx2 = hasUV ? uvIndices[i + 1] + uvStartIndex : -1;

                int n_idx0  = hasN ? normalIndices[0] + normalStartIndex : -1;
                int n_idx1  = hasN ? normalIndices[i] + normalStartIndex : -1;
                int n_idx2  = hasN ? normalIndices[i + 1] + normalStartIndex : -1;

                Triangle tri;
                if (isLight)
                    tri = Triangle(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, uv_idx0, uv_idx1, uv_idx2, e, nextLightIndex, mesh.size());
                else
                    tri = Triangle(idx0, idx1, idx2, n_idx0, n_idx1, n_idx2, materialID, uv_idx0, uv_idx1, uv_idx2, e, -51, mesh.size());
                mesh.push_back(tri);

                if (isLight) {
                    lights.push_back(tri);
                    nextLightIndex++;
                }
            }
        }
    }

    file.close();
}

