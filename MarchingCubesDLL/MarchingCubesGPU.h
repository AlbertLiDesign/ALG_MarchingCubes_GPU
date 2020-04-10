#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <SDKDDKVer.h>

typedef int uint;
typedef unsigned char uchar;

using namespace std;

extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint * voxelVerts, uint * voxelOccupied, uint3 gridSize,
    uint numVoxels, float3 basePoint, float3 voxelSize, float isoValue);

extern "C" void launch_compactVoxels(dim3 grid, dim3 threads, uint * compactedVoxelArray, uint * voxelOccupied,
    uint * voxelOccupiedScan, uint numVoxels);

extern "C" void launch_extractIsosurface(dim3 grid, dim3 threads,
    float3 * result, uint * compactedVoxelArray, uint * numVertsScanned,
    uint3 gridSize, float3 basePoint, float3 voxelSize, float isoValue);

extern "C" void exclusiveSumScan(uint * output, uint * input, uint numElements);
struct cfloat3
{
    float x, y, z;
};
float3 Convert(cfloat3 a)
{
    return make_float3(a.x, a.y, a.z);
}
cfloat3 Convert(float3 a)
{
    cfloat3 cf;
    cf.x = a.x;
    cf.y = a.y;
    cf.z = a.z;
    return cf;
}

// constants
float3 basePoint;
uint3 gridSize;

float3 voxelSize;
uint numVoxels = 0;
uint num_activeVoxels = 0;
uint num_resultVertices = 0;

float isoValue = 0.2f;

// device data
float3* d_result = 0;

uint* d_voxelVerts = 0;
uint* d_voxelVertsScan = 0;
uint* d_voxelOccupied = 0;
uint* d_voxelOccupiedScan = 0;
uint* d_compVoxelArray;

// output
float3* resultPts;

// forward declarations
void initMC();
void cleanup();