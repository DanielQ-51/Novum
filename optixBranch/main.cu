#include <iostream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include "optixStructs.cuh"
#include "optixSetup.cuh"
#include "hostSetup.cuh"

#define ASSET_PATH(path) (std::string(ROOT_DIR) + "/" + path)

#ifndef PTX_DIR
#define PTX_DIR "" 
#endif

int main() {
    std::cout << "Novum Experimental Optix Branch Launched\n ----------------------------------------------------------------------------------------------\n";

    OptixEngineState engineState;
    initOptixSystem(engineState);

    initRender(engineState,  ASSET_PATH("configs/config.rendertron"), 0);

    optixEngineCleanup(engineState);
    std::cout << "goodbye\n";
    return 0;
}
