## Introduction

ALG_MarchingCubes_GPU is an isosurface extraction plug-in for Grasshopper using [Marching Cubes algorithm](https://en.wikipedia.org/wiki/Marching_cubes) on GPU. 
![](https://albertlidesign.github.io/post-images/1586082938627.png)

## Algorithm

1. Classify voxels: Mark all the active voxels to get an active voxels array and calculate the number of vertices in each voxel by looking up vertices table. It is executed using one thread per voxel.

2. Exclusive sum scan:  Get the total number of active voxels and the total number of resulting vertices using exclusive scan algorithm. They are obtained by the sum of the last value of the exclusive scan and the last value of input array.

3. Compact voxels: This compacts the active voxels array to get rid of empty voxels. This allows us to execute Isosurface Extraction on only the active voxels.

4. Isosurface extraction: Calculate the position of the points in the active voxel and obtain all the result points by looking up triangle table.

5. Generate a mesh model from result points.

## Reference

[1] Dyken, C., Ziegler, G., Theobalt, C., & Seidel, H. P. (2008, December). High‐speed marching cubes using histopyramids. In Computer Graphics Forum (Vol. 27, No. 8, pp. 2028-2039). Oxford, UK: Blackwell Publishing Ltd.

[2] Lorensen W E, Cline H E. Marching cubes: A high resolution 3D surface construction algorithm. ACM SIGGRAPH Computer Graphics. 1987;21(4)

[3] C. Dyken, G. Ziegler, C. Theobalt, and H.-P. Seidel. High-speed Marching Cubes using HistoPyramids. Computer Graphics Forum, 27(8):2028–2039, Dec. 2008.

[4] The algorithm and lookup tables by Paul Bourke httppaulbourke.netgeometrypolygonise：http://paulbourke.net/geometry/polygonise/

[5] Marching Cubes implementation using OpenCL and OpenGL：https://www.eriksmistad.no/marching-cubes-implementation-using-opencl-and-opengl/

[6] A sample extracts a geometric isosurface from a volume dataset using the marching cubes algorithm.: https://github.com/tpn/cuda-samples/tree/master/v10.2/2_Graphics/marchingCubes

[7] The introduction of marching cubes: http://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html

[8] The introduction of marching cubes: https://medium.com/zeg-ai/voxel-to-mesh-conversion-marching-cube-algorithm-43dbb0801359
