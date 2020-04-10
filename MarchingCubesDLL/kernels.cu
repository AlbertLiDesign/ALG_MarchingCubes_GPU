#include <stdio.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include "tables.h"


// compute values of each corner point
__device__ float computeValue(float x, float y, float z)
{
    return cos(x) * sin(y) + cos(y) * sin(z) + cos(z) * sin(x);
}
__device__ float computeValue(float3 p)
{
    return computeValue(p.x,p.y,p.z);
}
__device__ float4 computeValue4(float3 p)
{
    float v = computeValue(p.x, p.y, p.z);
    const float d = 0.001f;
    float dx = computeValue(p.x + d, p.y, p.z) - v;
    float dy = computeValue(p.x, p.y + d, p.z) - v;
    float dz = computeValue(p.x, p.y, p.z + d) - v;

    return make_float4(dx, dy, dz, v);
}
// compute 3d index in the grid from 1d index
__device__ uint3 calcGridPos(uint i, uint3 gridSize)
{
    uint3 gridPos;

    gridPos.z = i / (gridSize.x * gridSize.y);
    gridPos.y = i % (gridSize.x * gridSize.y) / gridSize.x;
    gridPos.x = i % (gridSize.x * gridSize.y) % gridSize.x;
    return gridPos;
}
__device__ void calcOffsetValue(float isolevel, float3 p0, float3 p1, float4 f0, float4 f1, float3& p)
{
    float t = (isolevel - f0.w) / (f1.w - f0.w);
    p = p0 + t * (p1 - p0);
}


// classify voxel
__global__ void classifyVoxel(uint* voxelVerts, uint* voxelOccupied, uint3 gridSize,
    uint numVoxels, float3 basePoint, float3 voxelSize, float isoValue)
{
    uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
    uint i = blockId * blockDim.x + threadIdx.x;

    uint3 gridPos = calcGridPos(i, gridSize);

    float3 p;
    p.x = basePoint.x + gridPos.x * voxelSize.x;
    p.y = basePoint.y + gridPos.y * voxelSize.y;
    p.z = basePoint.z + gridPos.z * voxelSize.z;

    float field0 = computeValue(p);
    float field1 = computeValue(make_float3(voxelSize.x + p.x, 0.0f + p.y, 0.0f + p.z));
    float field2 = computeValue(make_float3(voxelSize.x + p.x, voxelSize.y + p.y, 0.0f + p.z));
    float field3 = computeValue(make_float3(0.0f + p.x, voxelSize.y + p.y, 0.0f + p.z));
    float field4 = computeValue(make_float3(0.0f + p.x, 0.0f + p.y, voxelSize.z + p.z));
    float field5 = computeValue(make_float3(voxelSize.x + p.x, 0.0f + p.y, voxelSize.z + p.z));
    float field6 = computeValue(make_float3(voxelSize.x + p.x, voxelSize.y + p.y, voxelSize.z + p.z));
    float field7 = computeValue(make_float3(0.0f + p.x, voxelSize.y + p.y, voxelSize.z + p.z));

    // calculate flag indicating if each vertex is inside or outside isosurface
    uint cubeindex;
    cubeindex = uint(field0 < isoValue);
    cubeindex += uint(field1 < isoValue) * 2;
    cubeindex += uint(field2 < isoValue) * 4;
    cubeindex += uint(field3 < isoValue) * 8;
    cubeindex += uint(field4 < isoValue) * 16;
    cubeindex += uint(field5 < isoValue) * 32;
    cubeindex += uint(field6 < isoValue) * 64;
    cubeindex += uint(field7 < isoValue) * 128;

    // read number of vertices from texture
    uint numVerts = numVertsTable[cubeindex];

    voxelVerts[i] = numVerts;
    if ((numVerts > 0))
    {
        voxelOccupied[i] = 1;
    }
}

// compact voxel array
__global__ void compactVoxels(uint* compactedVoxelArray, uint* voxelOccupied, uint* voxelOccupiedScan, uint numVoxels)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    if (voxelOccupied[i] && (i < numVoxels))
    {
        compactedVoxelArray[voxelOccupiedScan[i]] = i;
    }
}

__global__ void extractIsosurface(float3* result, uint* compactedVoxelArray, uint* numVertsScanned,
    uint3 gridSize, float3 basePoint, float3 voxelSize, float isoValue)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    // compute position in 3d grid
    uint3 gridPos = calcGridPos(compactedVoxelArray[i], gridSize);

    float3 p;
    p.x = basePoint.x + gridPos.x * voxelSize.x;
    p.y = basePoint.y + gridPos.y * voxelSize.y;
    p.z = basePoint.z + gridPos.z * voxelSize.z;

    float3 v[8];
    v[0] = p;
    v[1] = make_float3(voxelSize.x + p.x, 0.0f + p.y, 0.0f + p.z);
    v[2] = make_float3(voxelSize.x + p.x, voxelSize.y + p.y, 0.0f + p.z);
    v[3] = make_float3(0.0f + p.x, voxelSize.y + p.y, 0.0f + p.z);
    v[4] = make_float3(0.0f + p.x, 0.0f + p.y, voxelSize.z + p.z);
    v[5] = make_float3(voxelSize.x + p.x, 0.0f + p.y, voxelSize.z + p.z);
    v[6] = make_float3(voxelSize.x + p.x, voxelSize.y + p.y, voxelSize.z + p.z);
    v[7] = make_float3(0.0f + p.x, voxelSize.y + p.y, voxelSize.z + p.z);

    float4 field[8];
    field[0] = computeValue4(v[0]);
    field[1] = computeValue4(v[1]);
    field[2] = computeValue4(v[2]);
    field[3] = computeValue4(v[3]);
    field[4] = computeValue4(v[4]);
    field[5] = computeValue4(v[5]);
    field[6] = computeValue4(v[6]);
    field[7] = computeValue4(v[7]);

    // calculate flag indicating if each vertex is inside or outside isosurface
    uint cubeindex;
    cubeindex = uint(field[0].w < isoValue);
    cubeindex += uint(field[1].w < isoValue) * 2;
    cubeindex += uint(field[2].w < isoValue) * 4;
    cubeindex += uint(field[3].w < isoValue) * 8;
    cubeindex += uint(field[4].w < isoValue) * 16;
    cubeindex += uint(field[5].w < isoValue) * 32;
    cubeindex += uint(field[6].w < isoValue) * 64;
    cubeindex += uint(field[7].w < isoValue) * 128;

    float3 vertlist[12];
    float offsetV[12];

    // compute the position of all vertices
    calcOffsetValue(isoValue, v[0], v[1], field[0], field[1], vertlist[0]);
    calcOffsetValue(isoValue, v[1], v[2], field[1], field[2], vertlist[1]);
    calcOffsetValue(isoValue, v[2], v[3], field[2], field[3], vertlist[2]);
    calcOffsetValue(isoValue, v[3], v[0], field[3], field[0], vertlist[3]);
    calcOffsetValue(isoValue, v[4], v[5], field[4], field[5], vertlist[4]);
    calcOffsetValue(isoValue, v[5], v[6], field[5], field[6], vertlist[5]);
    calcOffsetValue(isoValue, v[6], v[7], field[6], field[7], vertlist[6]);
    calcOffsetValue(isoValue, v[7], v[4], field[7], field[4], vertlist[7]);
    calcOffsetValue(isoValue, v[0], v[4], field[0], field[4], vertlist[8]);
    calcOffsetValue(isoValue, v[1], v[5], field[1], field[5], vertlist[9]);
    calcOffsetValue(isoValue, v[2], v[6], field[2], field[6], vertlist[10]);
    calcOffsetValue(isoValue, v[3], v[7], field[3], field[7], vertlist[11]);

    // read number of vertices from texture
    uint numVerts = numVertsTable[cubeindex];

    for (int j = 0; j < numVerts; j++)
    {
        //find out which edge intersects the isosurface
        uint edge = triTable[cubeindex * 16 + j];
        uint index = numVertsScanned[compactedVoxelArray[i]] + j;

        result[index] = vertlist[edge];
    }
}

#pragma region pass methods


extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint * voxelVerts, uint * voxelOccupied, uint3 gridSize,
    uint numVoxels, float3 basePoint, float3 voxelSize, float isoValue)
{
    // calculate number of vertices need per voxel
    classifyVoxel << <grid, threads >> > (voxelVerts, voxelOccupied, gridSize,
        numVoxels, basePoint, voxelSize, isoValue);
    getLastCudaError("classifyVoxel failed");
}

extern "C" void launch_compactVoxels(dim3 grid, dim3 threads, uint * compactedVoxelArray, uint * voxelOccupied, uint * voxelOccupiedScan, uint numVoxels)
{
    compactVoxels << <grid, threads >> > (compactedVoxelArray, voxelOccupied,
        voxelOccupiedScan, numVoxels);
    getLastCudaError("compactVoxels failed");
}

extern "C" void exclusiveSumScan(uint * output, uint * input, uint numElements)
{
    thrust::exclusive_scan(thrust::device_ptr<uint>(input),
        thrust::device_ptr<uint>(input + numElements),
        thrust::device_ptr<uint>(output));
}

extern "C" void launch_extractIsosurface(dim3 grid, dim3 threads,
    float3 * result, uint * compactedVoxelArray, uint * numVertsScanned,
    uint3 gridSize, float3 basePoint, float3 voxelSize, float isoValue)
{
    extractIsosurface << <grid, threads >> > (result, compactedVoxelArray, numVertsScanned,
        gridSize, basePoint, voxelSize, isoValue);
    getLastCudaError("extract Isosurface failed");
}
#pragma endregion
