## Introduction

ALG_MarchingCubes_GPU is an isosurface extraction plug-in for Grasshopper run on GPU. It provides two versions(C# and C++) to implement Marching Cubes on GPU, the C# version is based on [Alea GPU](http://www.aleagpu.com/release/3_0_4/doc/)) and the C++ version is based on [CUDA v10.2](https://developer.nvidia.com/cuda-downloads). This project is an important learning experience for me, I hope it can be a parallel programming reference case to help GPU programming beginners and Grasshopper developers.

At present, its computational performance can still be optimized (e.g. 99% of computing time is spent on the memory copy), I will continue to improve it in the future work.

![](https://albertlidesign.github.io/post-images/1586082938627.png)

## Installation

### Alea GPU

To compile the c# version, you need to install [Alea GPU](http://www.aleagpu.com/release/3_0_4/doc/).  Alea GPU requires a CUDA-capable GPU with **compute capability 2.0** or higher. Alea GPU consists of several assemblies, tools and resources, organized in multiple [NuGet packages](http://www.nuget.org/profiles/quantalea). So you can install them directly through Visual Studio. 

It is important to note that you also need to install **FSharp.Core** package and **CUDA v9.0** (CUDA v10.2 is not supported) before using Alea GPU in C#.

If you want to learn more about Alea GPU, please check the web site [Alea GPU](http://www.aleagpu.com/release/3_0_4/doc/)

![](https://albertlidesign.github.io/post-images/1586082600760.png)

### CUDA

To compile the c++ version, you need to install [CUDA v10.2](https://developer.nvidia.com/cuda-downloads). 

## Performance

![](https://albertlidesign.github.io/post-images/1586082652606.png)

## Reference

[1] Dyken, C., Ziegler, G., Theobalt, C., & Seidel, H. P. (2008, December). High‐speed marching cubes using histopyramids. In Computer Graphics Forum (Vol. 27, No. 8, pp. 2028-2039). Oxford, UK: Blackwell Publishing Ltd.

[2] Lorensen W E, Cline H E. Marching cubes: A high resolution 3D surface construction algorithm. ACM SIGGRAPH Computer Graphics. 1987;21(4)

[3] C. Dyken, G. Ziegler, C. Theobalt, and H.-P. Seidel. High-speed Marching Cubes using HistoPyramids. Computer Graphics Forum, 27(8):2028–2039, Dec. 2008.

[4] The algorithm and lookup tables by Paul Bourke httppaulbourke.netgeometrypolygonise：http://paulbourke.net/geometry/polygonise/

[5] Marching Cubes implementation using OpenCL and OpenGL：https://www.eriksmistad.no/marching-cubes-implementation-using-opencl-and-opengl/

[6] Samples for CUDA Developers which demonstrates features in CUDA Toolkit: https://github.com/NVIDIA/cuda-samples

[7] Samples and C# GPU Programming tutorials for Alea GPU Developers: http://www.aleagpu.com/release/3_0_4/doc/

