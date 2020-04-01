#include <stdlib.h>
# include<iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#include "MarchingCubesGPU.h"

int main()
{
    initMC();

    runComputeIsosurface();

    cleanup();

    writeFile(outputPath);
    system("PAUSE");
}

// Load arguments
float3* loadFile(string filename)
{
    ifstream inFile;

    inFile.open(filename);
    if (inFile) {
        cout << "pts.txt open scessful" << endl;

        string s;
        getline(inFile, s);
        sampleLength = stoi(s);

        float bf1, bf2, bf3;
        inFile >> bf1 >> bf2 >> bf3;
        basePoint = make_float3(bf1, bf2, bf3);

        inFile >> scale >> isoValue;
        voxelSize = make_float3(scale, scale, scale);

        int xCount, yCount, zCount;
        inFile >> xCount >> yCount >> zCount;
        gridSize = make_uint3(xCount, yCount, zCount);
        numVoxels = xCount * yCount * zCount;

        samplePts = new float3[sampleLength * sizeof(float3)];
        int i = 0;
        float f1, f2, f3;
        while (inFile >> f1 >> f2 >> f3)
        {
            samplePts[i] = make_float3(f1, f2, f3);
            i++;
        }

        inFile.close();
        return samplePts;
    }
    else
    {
        cout << "endless.txt doesn't exist" << endl;
        float3* samplePts = new float3[sampleLength];
        return samplePts;
    }
}

void writeFile(string filename)
{
    ofstream outFile;
    outFile.open(filename);
    if(outFile.is_open())
    {
        for (size_t i = 0; i < num_resultVertices; i++)
        {
            outFile << resultPts[i].x << '\t' << resultPts[i].y << '\t' << resultPts[i].z << endl;
        }
        
        outFile.close();
    }
}

void initMC()
{
    loadFile(filePath);

    cout << "Arguments: " << endl;
    cout << gridSize.x << ' ' << gridSize.y << ' ' << gridSize.z << endl;
    cout << "isoVaule = " << isoValue << '\t' << "Scale = " << scale << endl;
    cout << sampleLength << endl;

    // allocate textures
    allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

    // allocate device memory
    unsigned int memSize = sizeof(uint) * numVoxels;
    checkCudaErrors(cudaMalloc((void**)&d_voxelVerts, memSize));
    checkCudaErrors(cudaMalloc((void**)&d_voxelVertsScan, memSize));
    checkCudaErrors(cudaMalloc((void**)&d_voxelOccupied, memSize));
    checkCudaErrors(cudaMalloc((void**)&d_voxelOccupiedScan, memSize));
    checkCudaErrors(cudaMalloc((void**)&d_compVoxelArray, memSize));

    // allocate sample points and copy them to device
    checkCudaErrors(cudaMalloc((void**)&d_samplePts, sampleLength * sizeof(float3)));
    cudaMemcpy(d_samplePts, samplePts, sampleLength * sizeof(float3), cudaMemcpyHostToDevice);
}
void cleanup()
{
    checkCudaErrors(cudaFree(d_result));

    checkCudaErrors(cudaFree(d_edgeTable));
    checkCudaErrors(cudaFree(d_triTable));
    checkCudaErrors(cudaFree(d_numVertsTable));

    checkCudaErrors(cudaFree(d_voxelVerts));
    checkCudaErrors(cudaFree(d_voxelVertsScan));
    checkCudaErrors(cudaFree(d_voxelOccupied));
    checkCudaErrors(cudaFree(d_voxelOccupiedScan));
    checkCudaErrors(cudaFree(d_compVoxelArray));

    checkCudaErrors(cudaFree(d_samplePts));
}
void runComputeIsosurface()
{
    int threads = 128;
    dim3 grid((numVoxels+threads -1) / threads, 1, 1);

    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535)
    {
        grid.y = grid.x / 32768;
        grid.x = 32768;
    }

    // calculate number of vertices need per voxel
    launch_classifyVoxel(grid, threads,
        d_voxelVerts, d_voxelOccupied, gridSize,
        numVoxels, basePoint, voxelSize, isoValue, d_samplePts, sampleLength);

    // after classifying voxels, we will get a lot of empty voxels.
    // in order to cull them, we have to use exclusive sum scan to get a new array
    // the last element in this new array plus the last element in the array before scan 
    // equals the number of active voxels
    exclusiveSumScan(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

    uint lastElement, lastScanElement;
    // only copy the last elements from two arrays on the device
    checkCudaErrors(cudaMemcpy((void*)&lastElement,
        (void*)(d_voxelOccupied + numVoxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)&lastScanElement,
        (void*)(d_voxelOccupiedScan + numVoxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost));

    // comput the number of active voxels
    num_activeVoxels = lastElement + lastScanElement;

    cout << num_activeVoxels << endl;

    if (num_activeVoxels == 0)
    {
        // return if there are no full voxels
        num_resultVertices = 0;
        return;
    }

    // compact voxel index array
    launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);

    // compute the number of output vertices
    exclusiveSumScan(d_voxelVertsScan, d_voxelVerts, numVoxels);

    uint lastElement2, lastScanElement2;
    checkCudaErrors(cudaMemcpy((void*)&lastElement2,
        (void*)(d_voxelVerts + numVoxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)&lastScanElement2,
        (void*)(d_voxelVertsScan + numVoxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost));
    num_resultVertices = lastElement2 + lastScanElement2;

    cout << num_resultVertices << endl;

    checkCudaErrors(cudaMalloc((void**)&(d_result), num_resultVertices * sizeof(float3)));

    dim3 grid2((num_activeVoxels + threads - 1) / threads, 1, 1);

    // get around maximum grid size of 65535 in each dimension
    if (grid.x > 65535)
    {
        grid2.x /= 2;
        grid2.y *= 2;
    }

    launch_extractIsosurface(grid2, threads, d_result, d_compVoxelArray, d_voxelVertsScan,
        gridSize, basePoint, voxelSize,
        isoValue, scale, d_samplePts, sampleLength);

    resultPts = new float3[num_resultVertices * sizeof(float3)];

    checkCudaErrors(cudaMemcpy(resultPts,
        d_result, num_resultVertices * sizeof(float3),
        cudaMemcpyDeviceToHost));
}

