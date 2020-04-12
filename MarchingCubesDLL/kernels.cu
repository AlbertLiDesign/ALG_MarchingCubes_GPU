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

    float3 v0 = p;
    float3 v1 = make_float3(voxelSize.x + p.x, 0.0f + p.y, 0.0f + p.z);
    float3 v2 = make_float3(voxelSize.x + p.x, voxelSize.y + p.y, 0.0f + p.z);
    float3 v3 = make_float3(0.0f + p.x, voxelSize.y + p.y, 0.0f + p.z);
    float3 v4 = make_float3(0.0f + p.x, 0.0f + p.y, voxelSize.z + p.z);
    float3 v5 = make_float3(voxelSize.x + p.x, 0.0f + p.y, voxelSize.z + p.z);
    float3 v6 = make_float3(voxelSize.x + p.x, voxelSize.y + p.y, voxelSize.z + p.z);
    float3 v7 = make_float3(0.0f + p.x, voxelSize.y + p.y, voxelSize.z + p.z);

   float4 field0 = computeValue4(v0);
   float4 field1 = computeValue4(v1);
   float4 field2 = computeValue4(v2);
   float4 field3 = computeValue4(v3);
   float4 field4 = computeValue4(v4);
   float4 field5 = computeValue4(v5);
   float4 field6 = computeValue4(v6);
   float4 field7 = computeValue4(v7);

    // calculate flag indicating if each vertex is inside or outside isosurface
    uint cubeindex;
    cubeindex = uint(field0.w < isoValue);
    cubeindex += uint(field1.w < isoValue) * 2;
    cubeindex += uint(field2.w < isoValue) * 4;
    cubeindex += uint(field3.w < isoValue) * 8;
    cubeindex += uint(field4.w < isoValue) * 16;
    cubeindex += uint(field5.w < isoValue) * 32;
    cubeindex += uint(field6.w < isoValue) * 64;
    cubeindex += uint(field7.w < isoValue) * 128;

    float3 vertlist[12];
    float offsetV[12];

    // compute the position of all vertices
    calcOffsetValue(isoValue, v0, v1, field0, field1, vertlist[0]);
    calcOffsetValue(isoValue, v1, v2, field1, field2, vertlist[1]);
    calcOffsetValue(isoValue, v2, v3, field2, field3, vertlist[2]);
    calcOffsetValue(isoValue, v3, v0, field3, field0, vertlist[3]);
    calcOffsetValue(isoValue, v4, v5, field4, field5, vertlist[4]);
    calcOffsetValue(isoValue, v5, v6, field5, field6, vertlist[5]);
    calcOffsetValue(isoValue, v6, v7, field6, field7, vertlist[6]);
    calcOffsetValue(isoValue, v7, v4, field7, field4, vertlist[7]);
    calcOffsetValue(isoValue, v0, v4, field0, field4, vertlist[8]);
    calcOffsetValue(isoValue, v1, v5, field1, field5, vertlist[9]);
    calcOffsetValue(isoValue, v2, v6, field2, field6, vertlist[10]);
    calcOffsetValue(isoValue, v3, v7, field3, field7, vertlist[11]);

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
