#pragma once

#include "objects.cuh"
#include "util.cuh"
#include "volumeRendering.cuh"
#include "sceneContexts.cuh"
#include "optixStructs.cuh"
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
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>


#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

#ifndef PTX_DIR
#define PTX_DIR "" 
#endif

__host__ std::string read_file_to_string(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

struct OptixEngineState {
    OptixDeviceContext context = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    OptixProgramGroup raygenProgramGroup = nullptr;
    OptixProgramGroup missProgramGroup = nullptr;
    OptixProgramGroup hitgroupProgramGroup = nullptr;
    OptixModule module = nullptr;
    
    CUdeviceptr d_rgRecord = 0;
    CUdeviceptr d_msRecord = 0;
    CUdeviceptr d_hgRecord = 0;
};

__host__ int initOptixSystem(OptixEngineState& engineState) {
    cudaFree(0); 
    CUcontext cuCtx = 0;

    if (optixInit() != OPTIX_SUCCESS) {
        std::cerr << "Failed to initialize OptiX!" << std::endl;
        return -1;
    }

    OptixDeviceContextOptions options = {};
    options.logCallbackLevel = 4;
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void*) {
        std::cerr << "[" << level << "][" << tag << "]: " << message << std::endl;
    };

    OptixDeviceContext context = nullptr;
    if (optixDeviceContextCreate(cuCtx, &options, &context) != OPTIX_SUCCESS) {
        std::cerr << "Failed to create OptiX context!" << std::endl;
        return -1;
    }

    std::cout << "OptiX Context Created Successfully. Ready to build." << std::endl;

    std::string ptxCode;
    try {
        std::string ptxPath = std::string(PTX_DIR) + "/renderer.ptx";
        
        std::cout << "Loading PTX from: " << ptxPath << std::endl;
        ptxCode = read_file_to_string(ptxPath);
        
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return -1;
    }

    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineOptions = {}; 
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineOptions.numPayloadValues = 5; 
    pipelineOptions.numAttributeValues = 2; // For triangle barycentrics
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    
    pipelineOptions.pipelineLaunchParamsVariableName = "params"; 

    OptixModule module = nullptr;
    optixModuleCreate(
        context, 
        &moduleOptions, 
        &pipelineOptions, 
        ptxCode.c_str(), 
        ptxCode.size(), 
        nullptr, nullptr, 
        &module
    );

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module            = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__unidirectional"; 

    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__gather";

    OptixProgramGroupDesc hitgroupDesc = {};
    hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupDesc.hitgroup.moduleCH = module;
    hitgroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__gather";

    OptixProgramGroupDesc programGroupDescs[] = { raygenDesc, missDesc, hitgroupDesc };
    OptixProgramGroup programGroups[3];

    optixProgramGroupCreate(context, programGroupDescs, 3, &pgOptions, nullptr, nullptr, programGroups);

    OptixProgramGroup raygenProgramGroup   = programGroups[0];
    OptixProgramGroup missProgramGroup     = programGroups[1];
    OptixProgramGroup hitgroupProgramGroup = programGroups[2];

    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;

    optixPipelineCreate(
        context, 
        &pipelineOptions, 
        &linkOptions, 
        programGroups, 3, 
        nullptr, nullptr, 
        &pipeline
    );

    struct RaygenRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    RaygenRecord rgRecord;
    optixSbtRecordPackHeader(raygenProgramGroup, &rgRecord); 

    CUdeviceptr d_rgRecord;
    cudaMalloc(reinterpret_cast<void**>(&d_rgRecord), sizeof(RaygenRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_rgRecord), &rgRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    RaygenRecord msRecord;
    optixSbtRecordPackHeader(missProgramGroup, &msRecord); 
    CUdeviceptr d_msRecord;
    cudaMalloc(reinterpret_cast<void**>(&d_msRecord), sizeof(RaygenRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_msRecord), &msRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    RaygenRecord hgRecord;
    optixSbtRecordPackHeader(hitgroupProgramGroup, &hgRecord); 
    CUdeviceptr d_hgRecord;
    cudaMalloc(reinterpret_cast<void**>(&d_hgRecord), sizeof(RaygenRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_hgRecord), &hgRecord, sizeof(RaygenRecord), cudaMemcpyHostToDevice);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord                = d_rgRecord;

    sbt.missRecordBase              = d_msRecord;
    sbt.missRecordStrideInBytes     = sizeof(RaygenRecord);
    sbt.missRecordCount             = 1;

    sbt.hitgroupRecordBase          = d_hgRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(RaygenRecord);
    sbt.hitgroupRecordCount         = 1;

    engineState.context = context;
    engineState.pipeline = pipeline;
    engineState.sbt = sbt;
    engineState.raygenProgramGroup = raygenProgramGroup;
    engineState.missProgramGroup = missProgramGroup;
    engineState.hitgroupProgramGroup = hitgroupProgramGroup;
    engineState.module = module;
    engineState.d_rgRecord = d_rgRecord;
    engineState.d_msRecord = d_msRecord;
    engineState.d_hgRecord = d_hgRecord;

    std::cout << "OptiX engine setup complete." << std::endl;
    return 0;
}

__host__ int optixEngineCleanup (OptixEngineState& engineState) {
    cudaFree(reinterpret_cast<void*>(engineState.d_rgRecord));
    cudaFree(reinterpret_cast<void*>(engineState.d_msRecord));
    cudaFree(reinterpret_cast<void*>(engineState.d_hgRecord));
    optixPipelineDestroy(engineState.pipeline);
    optixProgramGroupDestroy(engineState.raygenProgramGroup);
    optixProgramGroupDestroy(engineState.missProgramGroup);
    optixProgramGroupDestroy(engineState.hitgroupProgramGroup);
    optixModuleDestroy(engineState.module);
    optixDeviceContextDestroy(engineState.context);

    std::cout << "OptiX engine cleanup complete." << std::endl;
    return 0;
}

using namespace std;

void readObjSimple(string filename, vector<float4>& points, vector<float4>& normals, vector<float4>& colors, vector<float2>& uvs,vector<Triangle>& mesh, 
    vector<Triangle>& lights, vector<LightDescriptor>& lightDescriptors, float4 c, float4 e, int materialID, float4 offset = f4(0.0f));

__host__ OptixTraversableHandle buildOptixGAS(
    OptixDeviceContext context,
    const std::vector<float3>& vertices,
    const std::vector<uint3>& indices,
    CUdeviceptr& out_d_gas_output_buffer // We keep this to free it later
) {
    std::cout << "Begin OptiX GAS build." << std::endl;
    // 1. Upload Vertices to GPU
    CUdeviceptr d_vertices;
    size_t vertices_size = vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size);
    cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice);

    // 2. Upload Indices to GPU
    CUdeviceptr d_indices;
    size_t indices_size = indices.size() * sizeof(uint3);
    cudaMalloc(reinterpret_cast<void**>(&d_indices), indices_size);
    cudaMemcpy(reinterpret_cast<void*>(d_indices), indices.data(), indices_size, cudaMemcpyHostToDevice);

    // 3. Describe the geometry to OptiX
    uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    // Vertex data
    triangle_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices         = static_cast<uint32_t>(vertices.size());
    triangle_input.triangleArray.vertexBuffers       = &d_vertices;

    // Index data
    triangle_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes  = sizeof(uint3);
    triangle_input.triangleArray.numIndexTriplets    = static_cast<uint32_t>(indices.size());
    triangle_input.triangleArray.indexBuffer         = d_indices;

    triangle_input.triangleArray.flags               = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords       = 1;

    // 4. Set up build options (Fast trace speed, allow compaction if desired)
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // 5. Ask OptiX how much memory it needs
    OptixAccelBufferSizes gas_buffer_sizes;
    optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes);

    // 6. Allocate memory for the build process
    CUdeviceptr d_temp_buffer_gas;
    cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&out_d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes);

    // 7. Execute the build
    OptixTraversableHandle gas_handle = 0;
    optixAccelBuild(
        context,
        0,                  // CUDA stream
        &accel_options,
        &triangle_input,
        1,                  // Number of build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        out_d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr,            // Emitted properties (used for compaction)
        0                   // Num emitted properties
    );

    cudaDeviceSynchronize();

    cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas));
    cudaFree(reinterpret_cast<void*>(d_vertices));
    cudaFree(reinterpret_cast<void*>(d_indices));

    std::cout << "OptiX GAS build complete." << std::endl;
    return gas_handle;
}

int initRender(OptixEngineState& engineState, string configPath, int renderNumber)
{
    RenderConfig config;
    loadConfig(configPath, config);

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::cout << "------------------------------------------------------------------------------------------------------ \n" << 
        "Began render number " << renderNumber << ": \"" << config.name << "\"\n\n";

    std::cout << "Current time: " 
              << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S")
              << "\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    int w = config.width;
    int h = config.height;

    int integratorChoice = matchIntegrator(config.integratorType); 

    int sampleCount = config.sampleCount;
    int maxDepth = config.maxDepth;

    Camera camera;
    if (config.pinholeCamera)
        camera = Camera::Pinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov);
    else
        camera = Camera::NotPinhole(config.camPos, w, h, config.camRot.x, config.camRot.y, config.camRot.z, config.camFov, 
            config.camApeture, config.camFocalDist);

    camera.preCompute();

    Image image = Image(w, h);
    image.postProcess = config.postProcess;

    if (integratorChoice == OPTIX_NORMAL)
    {
        std::cout << "Rendering at " << w << " by " << h << " pixels, with " << 
            sampleCount <<" with a max depth of " << 
            maxDepth << ".\nIntegrating with Optix Naive + NEE Unidirectional MIS." << 
            endl << endl;
    }

    vector<float4> points;
    vector<float4> normals;
    vector<float4> colors; // unused now
    vector<float2> uvs;
    vector<Triangle> mesh;
    vector<Triangle> lightsvec;
    vector<Material> mats;

    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading environment map
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    EnvironmentMapManager envManager(ASSET_PATH("assets/environment/lakeside_sunrise_2k.exr"));
    //envManager.setRotation(70.0f + (float)renderNumber);
    envManager.setRotation(130.0f);
    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Loading Textures
    //---------------------------------------------------------------------------------------------------------------------------------------------------

    vector<Image> images;
    vector<float4> pixels;
    vector<int> widths;
    vector<int> heights;
    vector<int> startIndices;
    int currentStartIndex = 0;

    images.push_back(loadBMPToImage(ASSET_PATH("assets/textures/enkidutexture.bmp"), false));
    images.push_back(loadBMPToImage(ASSET_PATH("assets/textures/enkiduchibitexture.bmp"), false));
    images.push_back(loadBMPToImage(ASSET_PATH("assets/textures/leaftex2.bmp"), false));
    images.push_back(loadBMPToImage(ASSET_PATH("assets/textures/leafautumn.bmp"), false));
    images.push_back(loadBMPToImage(ASSET_PATH("assets/textures/wood.bmp"), false));
    images.push_back(loadBMPToImage(ASSET_PATH("assets/textures/wall.bmp"), false));
    images.push_back(loadBMPToImage(ASSET_PATH("assets/textures/Material.006_baseColor.bmp"), false));

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

    Material wood = Material::Leaf(4, startIndices[4], widths[4], heights[4], 1.5f, 0.3f, f4(), 0.00f);
    Material wall = Material::DiffuseTextured(5, startIndices[5], widths[5], heights[5]);
    Material lambertTextured = Material::DiffuseTextured(0, startIndices[0], widths[0], heights[0]);
    Material lambert2Textured = Material::DiffuseTextured(1, startIndices[1], widths[1], heights[1]);

    Material lambertBlue = Material::Diffuse(f4(0.4f,0.4f,0.8f));
    Material lambertGrey = Material::Diffuse(f4(0.8f,0.8f,0.8f));
    Material lambertGreyBlue = Material::Diffuse(f4(0.6f,0.6f,0.7f));
    Material lambertWhite = Material::Diffuse(f4(0.9f,0.9f,0.9f));
    Material lambertGreen = Material::Diffuse(f4(0.2f,0.6f,0.6f));
    Material lambertRed = Material::Diffuse(f4(0.90f,0.1f,0.1f));
    Material lambertVeryGreen = Material::Diffuse(f4(0.1f,0.9f,0.1f));
    Material lambertBLACK = Material::Diffuse(f4(0.0f,0.0f,0.0f));
    Material lambert95 = Material::Diffuse(f4(0.95f,0.95f,0.95f));
    Material lambert1 = Material::Diffuse(f4(1.0f));
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
    Material leaf = Material::Leaf(2, startIndices[2], widths[2], heights[2], 1.5f, 0.10f, f4(0.22f, 0.75f, 0.28f), 0.15f);
    Material leafAutumn = Material::Leaf(3, startIndices[3], widths[3], heights[3], 1.5f, 0.8f, f4(0.22f, 0.75f, 0.28f), 0.6f);
    Material canopy = Material::Leaf(2, startIndices[2], widths[2], heights[2], 1.5f, 0.9f, f4(0.22f, 0.75f, 0.28f), 0.7f);
    Material leafStem = Material::Diffuse(f4(0.90f, 0.9f, 0.83f));
    Material sky = Material::Diffuse(f4(0.4f, 0.4f, 1.00f));

    Material mirror = Material::Mirror();
    Material thinGlass = Material::ThinDielectric(1.5f);


    Material blade = Material::Metal(f4(2.88f, 2.49f, 2.12f), f4(3.05f, 2.97f, 2.76f), 0.15f);

    Material liners = Material::Metal(f4(1.80f, 1.40f, 0.40f), f4(2.10f, 2.80f, 4.20f), 0.35f);

    Material hardware = Material::Metal(eta_steel, k_steel, 0.45f);

    float4 cf_albedo = f4(0.03f, 0.03f, 0.03f, 0.0f);
    Material handles = Material::Diffuse(cf_albedo);

    Material glove = Material::Leaf(6, startIndices[6], widths[6], heights[6], 1.5f, 0.4f, f4(), 0.00f);

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
    mats.push_back(lambert1); // index 32
    mats.push_back(blade); // index 33
    mats.push_back(liners); // index 34
    mats.push_back(hardware); // index 35
    mats.push_back(handles); // index 36
    mats.push_back(glove); // index 37

    Material* mats_d;

    cudaMalloc(&mats_d, mats.size() * sizeof(Material));
    cudaMemcpy(mats_d, mats.data(), mats.size() * sizeof(Material), cudaMemcpyHostToDevice);

    vector<LightDescriptor> lightDesc;

    for (MeshConfig c : config.meshes)
    {
        readObjSimple(ASSET_PATH(c.path), points, normals, colors, uvs, mesh, lightsvec, lightDesc, f4(), 
                c.emissionMultiplier * c.emissionColor, c.materialID);
    }

    Vertices* verts;
    Triangle* scene;

    cudaMalloc(&verts,  sizeof(Vertices));
    Vertices temp;

    cudaMalloc(&temp.positions, sizeof(float4) * points.size());
    cudaMalloc(&temp.normals, sizeof(float4) * normals.size());
    cudaMalloc(&temp.uvs,  sizeof(float2) * uvs.size());

    cudaMemcpy(temp.positions, points.data(), points.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.normals, normals.data(), normals.size() * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(temp.uvs, uvs.data(), uvs.size() * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(verts, &temp, sizeof(Vertices), cudaMemcpyHostToDevice);

    if (mesh.size() == 0) {
        cout << "Error: No triangles loaded." << endl;
        return 1;
    }
    cout << "scene data read. There are " << mesh.size() << " Triangles and " << lightsvec.size() << " +1 lights" << endl;

    auto afterRead = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterRead = afterRead - start;
    std::cout << "Scene read took: " << elapsed_seconds_afterRead.count() << " seconds" << std::endl << endl;
    
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    // Computing BVH
    //---------------------------------------------------------------------------------------------------------------------------------------------------
    
    vector<float3> positions;
    vector<uint3> indices;

    positions.reserve(points.size());

    std::transform(points.begin(), points.end(), std::back_inserter(positions), 
        [](const float4& v) {
            return make_float3(v.x, v.y, v.z);
        }
    );

    for (Triangle& t : mesh) {
        indices.push_back(make_uint3(t.aInd, t.bInd, t.cInd));
    }

    CUdeviceptr out_d_gas_output_buffer;
    OptixTraversableHandle bvhHandle = buildOptixGAS(
        engineState.context,
        positions,
        indices,
        out_d_gas_output_buffer
    );

    auto afterBVH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds_afterBVH = afterBVH - afterRead;
    std::cout << "BVH construction took: " << elapsed_seconds_afterBVH.count() << " seconds" << std::endl << endl;

    Triangle* lights;

    // allocates and deallocates device pointer for lights (device light array)
    LightSamplerManager lightManager(lightDesc, lightsvec, points, lights, envManager.getView());
    
    cudaMalloc(&scene, mesh.size() * sizeof(Triangle));
    cudaMemcpy(scene, mesh.data(), mesh.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    float4* out_colors;
    cudaMalloc(&out_colors, w * h * sizeof(float4));
    cudaMemset(out_colors, 0, w * h * sizeof(float4));

    float4* d_finalOutput;
    cudaMalloc(&d_finalOutput, w * h * sizeof(float4));
    cudaMemset(d_finalOutput, 0, w * h * sizeof(float4));

    float4* host_colors = new float4[w * h];

    ShadeContext sc = {};

    sc.lightNum = lightsvec.size();
    sc.lights = lights;
    sc.scene = scene;
    sc.vertices = verts;
    sc.materials = mats_d;
    sc.textures = textures_d;
    sc.lightSampler = lightManager.getSampler();

    lightManager.getSampler().printDebugState();

    PipelineParams params = {};
    params.w = w;
    params.h = h;
    params.frame_index = 0;
    params.bvh_handle = bvhHandle; 
    params.accum_buffer = out_colors; 
    params.camera = camera;
    params.shadeContext = sc;
    params.max_depth = maxDepth;

    CUdeviceptr d_params;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(PipelineParams));
    cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(PipelineParams), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);  
    dim3 gridSize((w+15)/16, (h+15)/16);

    CUstream stream;
    cudaStreamCreate(&stream);

    auto renderStartTime = std::chrono::steady_clock::now();
    for (int sample = 0; sample < sampleCount; sample++) {

        optixLaunch(
            engineState.pipeline,
            stream,
            d_params,               // The GPU pointer we just allocated
            sizeof(PipelineParams), // FIXED: Was sizeof(Params)
            &engineState.sbt,           
            w,                   // Launch X
            h,                   // Launch Y
            1                       // Launch Z
        );

        cudaDeviceSynchronize();

        if ((sample % 30000 == 0 || sample == sampleCount-1) && DO_PROGRESSIVERENDER) 
        {
            cleanAndFormatImageNoOverlay<<<gridSize, blockSize>>>(
                params.accum_buffer, d_finalOutput, w, h, sample
            );

            cudaMemcpy(host_colors, d_finalOutput, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

            #pragma omp parallel for
            for (int i = 0; i < w * h; i++) {
                int x = i % w;
                int y = i / w;
                image.setColor(x, y, host_colors[i]);
            }
            std::string filename = "render.bmp";
            image.saveImageBMP(filename);
            image.saveImageCSV_MONO(0);
            

            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> elapsed = currentTime - renderStartTime;
            double avgTimeMs = elapsed.count() / (sample + 1);
            
            printf("\rSample %d/%d | Avg Time/Frame: %.2f ms", sample + 1, sampleCount, avgTimeMs);
            fflush(stdout);
        }

        params.frame_index++;
        cudaMemcpyAsync(
            reinterpret_cast<void*>(d_params), 
            &params, 
            sizeof(PipelineParams), 
            cudaMemcpyHostToDevice, 
            stream
        );
    }
    
    cudaMemcpy(host_colors, out_colors, w * h * sizeof(float4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            image.setColor(i, j, host_colors[image.toIndex(i, j)]/sampleCount);
        }
    }

    // memory freeing
    cudaFree(reinterpret_cast<void*>(out_d_gas_output_buffer));
    cudaFree(reinterpret_cast<void*>(d_params));
    cudaFree(out_colors);
    cudaFree(d_finalOutput);
    cudaFree(verts);
    cudaFree(scene);
    cudaFree(mats_d);
    cudaFree(textures_d);

    cudaFree(temp.positions);
    cudaFree(temp.normals);
    cudaFree(temp.uvs);
    delete[] host_colors;

    std::string filename = std::string(ROOT_DIR) + "/renders/optix/" + config.name + "" + std::to_string(renderNumber) + ".bmp";
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


void readObjSimple(
    string filename, 
    vector<float4>& points, 
    vector<float4>& normals, 
    vector<float4>& colors, 
    vector<float2>& uvs, 
    vector<Triangle>& mesh, 
    vector<Triangle>& lights, 
    vector<LightDescriptor>& lightDescriptors, 
    float4 c, float4 e, 
    int materialID, 
    float4 offset
)
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

    LightDescriptor ld;
    if (lengthSquared(e) > 0.0f) {
        ld.startInd = nextLightIndex;
        ld.totalPower = 0.0f;
    }

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
                float area = 0.5f * length(cp);

                if (area < 1e-18f) {
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
                    ld.totalPower += luminance(e) * h_PI * area;
                    nextLightIndex++;
                }
            }
        }
    }

    if (lengthSquared(e) > 0.0f) {
        ld.numPrim = lights.size() - ld.startInd;
        lightDescriptors.push_back(ld);
    }

    file.close();
}