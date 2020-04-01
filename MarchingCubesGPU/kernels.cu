#include <stdio.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "tables.h"


// textures containing look-up tables
texture<uint, 1, cudaReadModeElementType> edgeTexture;
texture<uint, 1, cudaReadModeElementType> faceTexture;
texture<uint, 1, cudaReadModeElementType> vertexTexture;


extern "C"
void allocateTextures(uint * *d_edgeTable, uint * *d_triTable, uint * *d_numVertsTable)
{
    checkCudaErrors(cudaMalloc((void**)d_edgeTable, 256 * sizeof(uint)));
    checkCudaErrors(cudaMemcpy((void*)*d_edgeTable, (void*)edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaBindTexture(0, edgeTexture, *d_edgeTable, channelDesc));

    checkCudaErrors(cudaMalloc((void**)d_triTable, 256 * 16 * sizeof(uint)));
    checkCudaErrors(cudaMemcpy((void*)*d_triTable, (void*)triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaBindTexture(0, faceTexture, *d_triTable, channelDesc));

    checkCudaErrors(cudaMalloc((void**)d_numVertsTable, 256 * sizeof(uint)));
    checkCudaErrors(cudaMemcpy((void*)*d_numVertsTable, (void*)numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaBindTexture(0, vertexTexture, *d_numVertsTable, channelDesc));
}

// compute values of each corner point
__device__ float computeValue(float3* samplePts, float3 testP, uint sampleLength)
{
    float result = 0.0f;
    float Dx, Dy, Dz;

    for (int j = 0; j < sampleLength; j++)
    {
        Dx = testP.x - samplePts[j].x;
        Dy = testP.y - samplePts[j].y;
        Dz = testP.z - samplePts[j].z;

        result += 1 / (Dx * Dx + Dy * Dy + Dz * Dz);
    }
    return result;
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
__device__ float calcOffsetValue(float Value1, float Value2, float ValueDesired)
{
    if ((Value2 - Value1) == 0.0f)
        return 0.5f;

    return (ValueDesired - Value1) / (Value2 - Value1);
}

// classify voxel
__global__ void classifyVoxel(uint* voxelVerts, uint* voxelOccupied, uint3 gridSize,
    uint numVoxels, float3 basePoint, float3 voxelSize,
    float isoValue, float3* samplePts, uint sampleLength)
{
    uint blockId = blockIdx.y * gridDim.x + blockIdx.x;
    uint i = blockId * blockDim.x + threadIdx.x;

    uint3 gridPos = calcGridPos(i, gridSize);

    float3 p;
    p.x = basePoint.x + gridPos.x * voxelSize.x;
    p.y = basePoint.y + gridPos.y * voxelSize.y;
    p.z = basePoint.z + gridPos.z * voxelSize.z;

    float field[8];
    field[0] = computeValue(samplePts, p, sampleLength);
    field[1] = computeValue(samplePts, make_float3(voxelSize.x + p.x, 0 + p.y, 0 + p.z), sampleLength);
    field[2] = computeValue(samplePts, make_float3(voxelSize.x + p.x, voxelSize.y + p.y, 0 + p.z), sampleLength);
    field[3] = computeValue(samplePts, make_float3(0 + p.x, voxelSize.y + p.y, 0 + p.z), sampleLength);
    field[4] = computeValue(samplePts, make_float3(0 + p.x, 0 + p.y, voxelSize.z + p.z), sampleLength);
    field[5] = computeValue(samplePts, make_float3(voxelSize.x + p.x, 0 + p.y, voxelSize.z + p.z), sampleLength);
    field[6] = computeValue(samplePts, make_float3(voxelSize.x + p.x, voxelSize.y + p.y, voxelSize.z + p.z), sampleLength);
    field[7] = computeValue(samplePts, make_float3(0 + p.x, voxelSize.y + p.y, voxelSize.z + p.z), sampleLength);

    // calculate flag indicating if each vertex is inside or outside isosurface
    uint cubeindex;
    cubeindex = uint(field[0] < isoValue);
    cubeindex += uint(field[1] < isoValue) * 2;
    cubeindex += uint(field[2] < isoValue) * 4;
    cubeindex += uint(field[3] < isoValue) * 8;
    cubeindex += uint(field[4] < isoValue) * 16;
    cubeindex += uint(field[5] < isoValue) * 32;
    cubeindex += uint(field[6] < isoValue) * 64;
    cubeindex += uint(field[7] < isoValue) * 128;

    // read number of vertices from texture
    uint numVerts = tex1Dfetch(vertexTexture, cubeindex);

    if (i < numVoxels)
    {
        voxelVerts[i] = numVerts;
        voxelOccupied[i] = (numVerts > 0);
    }
}

extern "C" void launch_classifyVoxel(dim3 grid, dim3 threads, uint * voxelVerts, uint * voxelOccupied, uint3 gridSize,
    uint numVoxels, float3 basePoint, float3 voxelSize,
    float isoValue, float3 * samplePts, uint sampleLength)
{
    // calculate number of vertices need per voxel
    classifyVoxel <<<grid, threads >>> (voxelVerts, voxelOccupied, gridSize,
        numVoxels, basePoint, voxelSize,
        isoValue, samplePts, sampleLength);
    getLastCudaError("classifyVoxel failed");
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

extern "C" void launch_compactVoxels(dim3 grid, dim3 threads, uint * compactedVoxelArray, uint * voxelOccupied, uint * voxelOccupiedScan, uint numVoxels)
{
    compactVoxels << <grid, threads >> > (compactedVoxelArray, voxelOccupied,
        voxelOccupiedScan, numVoxels);
    getLastCudaError("compactVoxels failed");
}

extern "C" void exclusiveSumScan(unsigned int* output, unsigned int* input, unsigned int numElements)
{
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
        thrust::device_ptr<unsigned int>(input + numElements),
        thrust::device_ptr<unsigned int>(output));
}

__global__ void extractIsosurface(float3* result, uint* compactedVoxelArray, uint* numVertsScanned,
    uint3 gridSize, float3 basePoint,  float3 voxelSize, float isoValue,float scale, 
    float3* samplePts, uint sampleLength)
{
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

    // compute position in 3d grid
    uint3 gridPos = calcGridPos(i, gridSize);

    float3 p;
    p.x = basePoint.x + gridPos.x * voxelSize.x;
    p.y = basePoint.y + gridPos.y * voxelSize.y;
    p.z = basePoint.z + gridPos.z * voxelSize.z;

    float field[8];
    field[0] = computeValue(samplePts, p, sampleLength);
    field[1] = computeValue(samplePts, make_float3(voxelSize.x + p.x, 0 + p.y, 0 + p.z), sampleLength);
    field[2] = computeValue(samplePts, make_float3(voxelSize.x + p.x, voxelSize.y + p.y, 0 + p.z), sampleLength);
    field[3] = computeValue(samplePts, make_float3(0 + p.x, voxelSize.y + p.y, 0 + p.z), sampleLength);
    field[4] = computeValue(samplePts, make_float3(0 + p.x, 0 + p.y, voxelSize.z + p.z), sampleLength);
    field[5] = computeValue(samplePts, make_float3(voxelSize.x + p.x, 0 + p.y, voxelSize.z + p.z), sampleLength);
    field[6] = computeValue(samplePts, make_float3(voxelSize.x + p.x, voxelSize.y + p.y, voxelSize.z + p.z), sampleLength);
    field[7] = computeValue(samplePts, make_float3(0 + p.x, voxelSize.y + p.y, voxelSize.z + p.z), sampleLength);

    // calculate flag indicating if each vertex is inside or outside isosurface
    uint cubeindex;
    cubeindex = uint(field[0] < isoValue);
    cubeindex += uint(field[1] < isoValue) * 2;
    cubeindex += uint(field[2] < isoValue) * 4;
    cubeindex += uint(field[3] < isoValue) * 8;
    cubeindex += uint(field[4] < isoValue) * 16;
    cubeindex += uint(field[5] < isoValue) * 32;
    cubeindex += uint(field[6] < isoValue) * 64;
    cubeindex += uint(field[7] < isoValue) * 128;

    float3 vertlist[12];
    float offsetV[12];

    //compute t values from two end points on each edge
    offsetV[0] = calcOffsetValue(field[0], field[1], isoValue);
    offsetV[1] = calcOffsetValue(field[1], field[2], isoValue);
    offsetV[2] = calcOffsetValue(field[2], field[3], isoValue);
    offsetV[3] = calcOffsetValue(field[3], field[0], isoValue);
    offsetV[4] = calcOffsetValue(field[4], field[5], isoValue);
    offsetV[5] = calcOffsetValue(field[5], field[6], isoValue);
    offsetV[6] = calcOffsetValue(field[6], field[7], isoValue);
    offsetV[7] = calcOffsetValue(field[7], field[4], isoValue);
    offsetV[8] = calcOffsetValue(field[0], field[4], isoValue);
    offsetV[9] = calcOffsetValue(field[1], field[5], isoValue);
    offsetV[10] = calcOffsetValue(field[2], field[6], isoValue);
    offsetV[11] = calcOffsetValue(field[3], field[7], isoValue);

    // compute the position of all vertices
    vertlist[0].x = basePoint.x + (gridPos.x + 0.0f + offsetV[0] * 1.0f) * scale;
    vertlist[0].y = basePoint.y + (gridPos.y + 0.0f + offsetV[0] * 0.0f) * scale;
    vertlist[0].z = basePoint.z + (gridPos.z + 0.0f + offsetV[0] * 0.0f) * scale;

    vertlist[1].x = basePoint.x + (gridPos.x + 1.0f + offsetV[1] * 0.0f) * scale;
    vertlist[1].y = basePoint.y + (gridPos.y + 0.0f + offsetV[1] * 1.0f) * scale;
    vertlist[1].z = basePoint.z + (gridPos.z + 0.0f + offsetV[1] * 0.0f) * scale;

    vertlist[2].x = basePoint.x + (gridPos.x + 1.0f + offsetV[2] * -1.0f) * scale;
    vertlist[2].y = basePoint.y + (gridPos.y + 1.0f + offsetV[2] * 0.0f) * scale;
    vertlist[2].z = basePoint.z + (gridPos.z + 0.0f + offsetV[2] * 0.0f) * scale;

    vertlist[3].x = basePoint.x + (gridPos.x + 0.0f + offsetV[3] * 0.0f) * scale;
    vertlist[3].y = basePoint.y + (gridPos.y + 1.0f + offsetV[3] * -1.0f) * scale;
    vertlist[3].z = basePoint.z + (gridPos.z + 0.0f + offsetV[3] * 0.0f) * scale;

    vertlist[4].x = basePoint.x + (gridPos.x + 0.0f + offsetV[4] * 1.0f) * scale;
    vertlist[4].y = basePoint.y + (gridPos.y + 0.0f + offsetV[4] * 0.0f) * scale;
    vertlist[4].z = basePoint.z + (gridPos.z + 1.0f + offsetV[4] * 0.0f) * scale;

    vertlist[5].x = basePoint.x + (gridPos.x + 1.0f + offsetV[5] * 0.0f) * scale;
    vertlist[5].y = basePoint.y + (gridPos.y + 0.0f + offsetV[5] * 1.0f) * scale;
    vertlist[5].z = basePoint.z + (gridPos.z + 1.0f + offsetV[5] * 0.0f) * scale;

    vertlist[6].x = basePoint.x + (gridPos.x + 1.0f + offsetV[6] * -1.0f) * scale;
    vertlist[6].y = basePoint.y + (gridPos.y + 1.0f + offsetV[6] * 0.0f) * scale;
    vertlist[6].z = basePoint.z + (gridPos.z + 1.0f + offsetV[6] * 0.0f) * scale;

    vertlist[7].x = basePoint.x + (gridPos.x + 0.0f + offsetV[7] * 0.0f) * scale;
    vertlist[7].y = basePoint.y + (gridPos.y + 1.0f + offsetV[7] * -1.0f) * scale;
    vertlist[7].z = basePoint.z + (gridPos.z + 1.0f + offsetV[7] * 0.0f) * scale;

    vertlist[8].x = basePoint.x + (gridPos.x + 0.0f + offsetV[8] * 0.0f) * scale;
    vertlist[8].y = basePoint.y + (gridPos.y + 0.0f + offsetV[8] * 0.0f) * scale;
    vertlist[8].z = basePoint.z + (gridPos.z + 0.0f + offsetV[8] * 1.0f) * scale;

    vertlist[9].x = basePoint.x + (gridPos.x + 1.0f + offsetV[9] * 0.0f) * scale;
    vertlist[9].y = basePoint.y + (gridPos.y + 0.0f + offsetV[9] * 0.0f) * scale;
    vertlist[9].z = basePoint.z + (gridPos.z + 0.0f + offsetV[9] * 1.0f) * scale;

    vertlist[10].x = basePoint.x + (gridPos.x + 1.0f + offsetV[10] * 0.0f) * scale;
    vertlist[10].y = basePoint.y + (gridPos.y + 1.0f + offsetV[10] * 0.0f) * scale;
    vertlist[10].z = basePoint.z + (gridPos.z + 0.0f + offsetV[10] * 1.0f) * scale;

    vertlist[11].x = basePoint.x + (gridPos.x + 0.0f + offsetV[11] * 0.0f) * scale;
    vertlist[11].y = basePoint.y + (gridPos.y + 1.0f + offsetV[11] * 0.0f) * scale;
    vertlist[11].z = basePoint.z + (gridPos.z + 0.0f + offsetV[11] * 1.0f) * scale;

    // read number of vertices from texture
    uint numVerts = tex1Dfetch(vertexTexture, cubeindex);

    for (int j = 0; j < numVerts; j++)
    {
        //find out which edge intersects the isosurface
        uint edge = tex1Dfetch(faceTexture, cubeindex*16+j);
        uint index = numVertsScanned[i] + j;

        result[index] = vertlist[edge];
    }
}

extern "C" void launch_extractIsosurface(dim3 grid, dim3 threads,
    float3 * result, uint * compactedVoxelArray, uint * numVertsScanned,
    uint3 gridSize, float3 basePoint, float3 voxelSize, float isoValue, float scale,
    float3 * samplePts, uint sampleLength)
{
    extractIsosurface <<<grid, threads >>> (result, compactedVoxelArray, numVertsScanned,
        gridSize, basePoint, voxelSize, isoValue, scale,
        samplePts, sampleLength);
    getLastCudaError("extract Isosurface failed");
}
