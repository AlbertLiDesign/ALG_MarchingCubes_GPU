#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <SDKDDKVer.h>


typedef unsigned int uint;
typedef unsigned char uchar;

using namespace std;

extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint * voxelVerts, uint * voxelOccupied, uint3 gridSize,
    uint numVoxels, float3 basePoint, float3 voxelSize,
    float isoValue, float3 * samplePts, uint sampleLength);

extern "C" void launch_compactVoxels(dim3 grid, dim3 threads, uint * compactedVoxelArray, uint * voxelOccupied,
    uint * voxelOccupiedScan, uint numVoxels);

extern "C" void launch_extractIsosurface(dim3 grid, dim3 threads,
    float3 * result, uint * compactedVoxelArray, uint * numVertsScanned,
    uint3 gridSize, float3 basePoint, float3 voxelSize, float isoValue, float scale,
    float3 * samplePts, uint sampleLength);

extern "C" void allocateTextures(uint * *d_edgeTable, uint * *d_triTable, uint * *d_numVertsTable);
extern "C" void exclusiveSumScan(unsigned int* output, unsigned int* input, unsigned int numElements);

extern "C" __declspec(dllexport)  float3 * marchingcubesGPU(int sampleCount, float3 bP, float3 vS, int xCount, int yCount, int zCount, float3 * samplePoints);


// constants
const string filePath = "E:\\pts.txt";
const string outputPath = "E:\\result.txt";

float3 basePoint;
uint3 gridSize;

float3 voxelSize;
uint numVoxels = 0;
uint num_activeVoxels = 0;
uint num_resultVertices = 0;
uint sampleLength = 0;

float isoValue = 0.2f;
float dIsoValue = 0.005f;
float scale = 0.0f;


// device data
float3* samplePts;
float3* d_samplePts;

float3* d_result = 0;

uchar* d_volume = 0;
uint* d_voxelVerts = 0;
uint* d_voxelVertsScan = 0;
uint* d_voxelOccupied = 0;
uint* d_voxelOccupiedScan = 0;
uint* d_compVoxelArray;

// tables
uint* d_numVertsTable = 0;
uint* d_edgeTable = 0;
uint* d_triTable = 0;

// output
float3* resultPts;

// forward declarations
float3* loadFile(string filename);
void initMC();
void cleanup();
void runComputeIsosurface();
void writeFile(string filename);