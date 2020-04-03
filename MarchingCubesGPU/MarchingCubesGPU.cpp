#include <stdlib.h>
# include<iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_launch_parameters.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#include "MarchingCubesGPU.h"

#include<time.h>

extern "C" __declspec(dllexport)  void computMC(cfloat3 bP, cfloat3 vS,
    int xCount, int yCount, int zCount, float s, float iso, cfloat3 * samplePoints, 
    int sampleCount, size_t& resultLength);
extern "C" __declspec(dllexport)  void getResult(cfloat3 * result);
extern "C" __declspec(dllexport)  void freeMemory(cfloat3 * a);


//int main()
//{
//    ifstream inFile;
//
//    inFile.open(filePath);
//    cout << "pts.txt open scessful" << endl;
//
//    string s;
//    getline(inFile, s);
//    int sampleCount = stoi(s);
//
//    float bf1, bf2, bf3;
//    inFile >> bf1 >> bf2 >> bf3;
//    cfloat3 bp = Convert(make_float3(bf1, bf2, bf3));
//
//    float ss, iso;
//    inFile >> ss >> iso;
//    cfloat3 vS = Convert(make_float3(1, 1, 1));
//
//    int xCount, yCount, zCount;
//    inFile >> xCount >> yCount >> zCount;
//
//    cfloat3* samplePoints = new cfloat3[sampleCount*sizeof(cfloat3)];
//    int i = 0;
//    float f1, f2, f3;
//    while (inFile >> f1 >> f2 >> f3)
//    {
//        samplePoints[i] = Convert(make_float3(f1, f2, f3));
//        i++;
//    }
//
//    inFile.close();
//
//    size_t resultLength = 0;
//    cfloat3* result = marchingcubesGPU(bp, vS, xCount, yCount, zCount, ss, iso, samplePoints, sampleCount, resultLength);
//
//    //ofstream outFile;
//    //outFile.open(outputPath);
//    //if (outFile.is_open())
//    //{
//    //    for (size_t i = 0; i < num_resultVertices; i++)
//    //    {
//    //        outFile << result[i].x << '\t' << result[i].y << '\t' << result[i].z << endl;
//    //    }
//    //    outFile.close();
//    //}
//}

void computMC(cfloat3 bP, cfloat3 vS, int xCount, int yCount, int zCount,
    float s, float iso, cfloat3* samplePoints, int sampleCount, size_t& resultLength)
{

    sampleLength = sampleCount;
    basePoint = Convert(bP);
    voxelSize = Convert(vS);
    gridSize = make_uint3(xCount, yCount, zCount);
    numVoxels = xCount * yCount * zCount;
    scale = s;
    isoValue = iso;

    samplePts = new float3[sampleCount];
    for (int i = 0; i < sampleCount; i++)
    {
        samplePts[i] = Convert(samplePoints[i]);
    }

    initMC();

    runComputeIsosurface();

    cleanup();

    resultLength = (size_t)num_resultVertices;
}
void getResult(cfloat3* results)
{
    for (int i = 0; i < num_resultVertices; i++)
    {
        results[i] = Convert(resultPts[i]);
    }
    delete[] resultPts;
}

void freeMemory(cfloat3* a)
{
    delete[] a;
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
void writeScan()
{
    ofstream outFile;
    outFile.open("E:\\scanresult.txt");
    if (outFile.is_open())
    {
        for (size_t i = 0; i < numVoxels; i++)
        {
            outFile << voxelOccupiedScan[i] << endl;
        }

        outFile.close();
    }
}
void initMC()
{
    clock_t start2 = clock();

    // allocate textures
    allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

    clock_t end2 = clock();
    cout << "allocateTextures: " << (double)(end2 - start2) / CLOCKS_PER_SEC * 1000 << endl;

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
    delete[] samplePts;
    delete[] voxelOccupiedScan;
}
void runComputeIsosurface()
{
    clock_t start3 = clock();
    #pragma region Classify all voxels
    int threads = 256;
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
    #pragma endregion
    clock_t end3 = clock();
    cout << "launch_classifyVoxel: " << (double)(end3 - start3) / CLOCKS_PER_SEC * 1000 << endl;

    clock_t start4 = clock();
    #pragma region Scan occupied voxels
    // after classifying voxels, we will get a lot of empty voxels.
    // in order to cull them, we have to use exclusive sum scan to get a new array
    // the last element in this new array plus the last element in the array before scan 
    // equals the number of active voxels
    exclusiveSumScan(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

    //launch_scan(grid, threads,d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
    #pragma endregion
    clock_t end4 = clock();
    cout << "exclusiveSumScan1: " << (double)(end4 - start4) / CLOCKS_PER_SEC * 1000 << endl;

    //voxelOccupiedScan = new uint[numVoxels];

    //checkCudaErrors(cudaMemcpy(voxelOccupiedScan,
    //    d_voxelOccupiedScan, numVoxels * sizeof(uint),
    //    cudaMemcpyDeviceToHost));
    //writeScan();
    
    #pragma region Find the number of active voxels
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
    
    if (num_activeVoxels == 0)
    {
        // return if there are no full voxels
        num_resultVertices = 0;
        return;
    }

    #pragma endregion

    #pragma region Compact voxels
    // compact voxel index array
    launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
    #pragma endregion

    clock_t start43 = clock();
    #pragma region Scan the number of vertices each voxel has
    // compute the number of output vertices
    exclusiveSumScan(d_voxelVertsScan, d_voxelVerts, numVoxels);
    #pragma endregion
    clock_t end43 = clock();
    cout << "exclusiveSumScan2: " << (double)(end43 - start43) / CLOCKS_PER_SEC * 1000 << endl;

    #pragma region Find the number of sum vertices
    uint lastElement2, lastScanElement2;
    checkCudaErrors(cudaMemcpy((void*)&lastElement2,
        (void*)(d_voxelVerts + numVoxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy((void*)&lastScanElement2,
        (void*)(d_voxelVertsScan + numVoxels - 1),
        sizeof(uint), cudaMemcpyDeviceToHost));
    num_resultVertices = lastElement2 + lastScanElement2;
    #pragma endregion

    clock_t start5 = clock();
    #pragma region Generate isosurface
    
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

    resultPts = new float3[num_resultVertices];
    checkCudaErrors(cudaMemcpy(resultPts,
        d_result, num_resultVertices * sizeof(float3),
        cudaMemcpyDeviceToHost));
    #pragma endregion
    clock_t end5 = clock();
    cout << "launch_extractIsosurface: " << (double)(end5 - start5) / CLOCKS_PER_SEC * 1000 << endl;
}


