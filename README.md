# ALG_MarchingCubes_GPU

## Introduction

ALG_MarchingCubes_GPU is an isosurface extraction plug-in for Grasshopper built on GPU. It provides C# ([Alea GPU](http://www.aleagpu.com/release/3_0_4/doc/)) and C++ ([CUDA v10.2](https://developer.nvidia.com/cuda-downloads)) to implement GPU parallel acceleration. Since this project is preparing for a Grasshopper component, it also provided a method to pass C++ functions to C# (using IntPtr to receive structural Pointers from C++). As this project is an important learning experience for me, I hope it can be a good parallel programming case to help GPU programming beginners and Grasshopper developers.

At present, its computational efficiency can still be optimized (e.g. 99% of computing time is spent on the memory copy), I will continue to improve it in the future work.



## Installation

### Alea GPU

If you want to compile the c# version, you should install Alea GPU.  Alea GPU requires a CUDA-capable GPU with **compute capability 2.0** or higher. 

Alea GPU consists of several assemblies, tools and resources, organized in multiple [NuGet packages](http://www.nuget.org/profiles/quantalea).

The package [Alea](http://www.nuget.org/packages/Alea) installs the Alea GPU JIT compiler that translates IL code or F# quotations to GPU code.

```

```





If you want to learn more about Alea GPU, please check the web site [Alea GPU](http://www.aleagpu.com/release/3_0_4/doc/)



## Reference

[1]Dyken, C., Ziegler, G., Theobalt, C., & Seidel, H. P. (2008, December). High‐speed marching cubes using histopyramids. In Computer Graphics Forum (Vol. 27, No. 8, pp. 2028-2039). Oxford, UK: Blackwell Publishing Ltd.

[2] Lorensen W E, Cline H E. Marching cubes: A high resolution 3D surface construction algorithm. ACM SIGGRAPH Computer Graphics. 1987;21(4)

[3] C. Dyken, G. Ziegler, C. Theobalt, and H.-P. Seidel. High-speed Marching Cubes using HistoPyramids. Computer Graphics Forum, 27(8):2028–2039, Dec. 2008.

[4] The algorithm and lookup tables by Paul Bourke httppaulbourke.netgeometrypolygonise：http://paulbourke.net/geometry/polygonise/

[5] Marching Cubes implementation using OpenCL and OpenGL：https://www.eriksmistad.no/marching-cubes-implementation-using-opencl-and-opengl/

[6] Samples for CUDA Developers which demonstrates features in CUDA Toolkit: https://github.com/NVIDIA/cuda-samples

[7] Samples and C# GPU Programming tutorials for Alea GPU Developers: http://www.aleagpu.com/release/3_0_4/doc/

