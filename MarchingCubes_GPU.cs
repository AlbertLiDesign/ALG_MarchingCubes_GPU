using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Grasshopper.Kernel.Types.Transforms;
using Rhino.Geometry;
using Alea.CudaToolkit;
using Alea;
using Alea.CSharp;
using Alea.Parallel;

namespace ALG_MarchingCubes
{
    public class MarchingCubes_GPU
    {
        // constants
        public  int numVoxels = 0;
        public  int maxVerts = 0;

        public double3 voxelSize;
        public double isoValue = 0.2;

        public  Alea.int3 gridSize;

        // data
        public int[] cudaIndex;
        public Point3d[] samplePoints = null;
        double4[] pos = null, norm = null;
        public int[] voxelVerts = null;
        public int[] voxelVertsScan = null;
        public int[] voxelOccupied = null;
        public int[] voxelOccupiedScan = null;
        public int[] compactedVoxelArray = null;

        #region basic functions
        private double3[] ConvertPointsToDouble3(Point3d[] pts)
        {
            double3[] d = new double3[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                d[i].x = pts[i].X;
                d[i].y = pts[i].Y;
                d[i].z = pts[i].Z;
            }
            return d;
        }
        public List<Point3d> ConvertDouble3ToPoint3d(double3[] array)
        {
            List<Point3d> pts = new List<Point3d>();
            for (int i = 0; i < array.Length; i++)
            {
                pts.Add(new Point3d(array[i].x, array[i].y, array[i].z));
            }
            return pts;
        }
        public int Compact(double a, double b)
        {
            if (a < b)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
        public int Sum(int a, int b) { return a + b; }

        public List<Point3d> ConvertDouble4ToPoint3d(double4[] array)
        {
            List<Point3d> pts = new List<Point3d>();
            for (int i = 0; i < array.Length; i++)
            {
                pts.Add(new Point3d(array[i].x, array[i].y, array[i].z));
            }
            return pts;
        }

        //点的线性插值函数
        private double3 lerp(double3 a, double3 b, double t)
        {
            return CreateDouble3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z));
        }
        //浮点数的线性插值函数
        private double lerp(double a, double b, double t)
        {
            return a + t * (b - a);
        }
        //MC顶点的线性插值
        private double3 vertexInterp(double isolevel, double3 p0, double3 p1, float f0, float f1)
        {
            double t = (isolevel - f0) / (f1 - f0);
            return lerp(p0, p1, t);
        }

        // 计算边上的线性插值顶点
        private void vertexInterp2(double isolevel, double3 p0, double3 p1, double4 f0, double4 f1, ref double3 p, ref double3 n)
        {
            double t = (isolevel - f0.w) / (f1.w - f0.w);
            p = lerp(p0, p1, t);
            n.x = lerp(f0.x, f1.x, t);
            n.y = lerp(f0.y, f1.y, t);
            n.z = lerp(f0.z, f1.z, t);
        }
        //定义一个场函数，输入xyz坐标，返回一个值
        //v = ((3x)^4 - 5(3x)^2 - 5(3y)^2 + (3z)^4 - 5(z)^2 + 11.8) * 0.2 + 0.5
        private double tangle(double3[] samplePts, double x, double y, double z)
        {
            double result = 0.0;
            double Dx, Dy, Dz;

            for (int j = 0; j < samplePts.Length; j++)
            {
                Dx = x - samplePts[j].x;
                Dy = y - samplePts[j].y;
                Dz = z - samplePts[j].z;

                result += 1 / (Dx * Dx + Dy * Dy + Dz * Dz);
            }
            return result;
        }

        //根据一维索引计算在三维grid中的位置
        private Alea.int3 calcGridPos(int i, Alea.int3 gridSize)
        {
            Alea.int3 gridPos;

            gridPos.z = i / (gridSize.x * gridSize.y);
            gridPos.y = i % (gridSize.x * gridSize.y) / gridSize.x;
            gridPos.x = i % (gridSize.x * gridSize.y) % gridSize.x;

            return gridPos;
        }
        public double3 CreateDouble3(double x, double y, double z)
        {
            double3 p = new double3();
            p.x = x;
            p.y = y;
            p.z = z;
            return p;
        }

        public double ComputeValue(double3[] samplePts, double3 testP)
        {
            double result = 0.0;
            double Dx, Dy, Dz;
            double sum = samplePts.Length;

            for (int j = 0; j < samplePts.Length; j++)
            {
                Dx = testP.x - samplePts[j].x;
                Dy = testP.y - samplePts[j].y;
                Dz = testP.z - samplePts[j].z;

                result += (sum * (1 / sum)) / (Dx * Dx + Dy * Dy + Dz * Dz);
            }
            return result;
        }
        public double4 fieldFunc4(double3[] samplePts, double3 p)
        {
            double4 d4 = new double4();
            double v = ComputeValue(samplePts, p);
            const double d = 0.001;
            d4.x = tangle(samplePts, p.x + d, p.y, p.z) - v;
            d4.y = tangle(samplePts, p.x, p.y + d, p.z) - v;
            d4.z = tangle(samplePts, p.x, p.y, p.z + d) - v;
            d4.w = v;

            return d4;
        }
        #endregion
        #region classifyVoxel
        public void classifyVoxel(double3[] voxelV, int[] voxelVerts, int[] voxelOccupied, Alea.int3 gridSize,
            int numVoxels, double3 voxelSize, double isoValue, double3[] samplePts,int[] VertsTable)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x; //block在grid中的位置
            int i = blockId * blockDim.x + threadIdx.x; //线程索引

            //计算grid中的位置
            Alea.int3 gridPos = calcGridPos(i, gridSize);

            double3 p = new double3();

            p.x = gridPos.x * voxelSize.x;
            p.y = gridPos.y * voxelSize.y;
            p.z = gridPos.z * voxelSize.z;

            //输出所有顶点
            voxelV[i*8] = p;
            voxelV[i*8+1] = CreateDouble3(voxelSize.x + p.x, 0 + p.y, 0 + p.z);
            voxelV[i*8+2] = CreateDouble3(voxelSize.x + p.x, voxelSize.y + p.y, 0 + p.z);
            voxelV[i*8+3] = CreateDouble3(0 + p.x, voxelSize.y + p.y, 0 + p.z);
            voxelV[i*8+4] = CreateDouble3(0 + p.x, 0 + p.y, voxelSize.z + p.z);
            voxelV[i*8+5] = CreateDouble3(voxelSize.x + p.x, 0 + p.y, voxelSize.z + p.z);
            voxelV[i*8+6] = CreateDouble3(voxelSize.x + p.x, voxelSize.y + p.y, voxelSize.z + p.z);
            voxelV[i*8+7] = CreateDouble3(0 + p.x, voxelSize.y + p.y, voxelSize.z + p.z);

            //计算cube中的8个点对应的value
            double d0 = ComputeValue(samplePts, voxelV[i * 8]);
            double d1 = ComputeValue(samplePts, voxelV[i * 8 + 1]);
            double d2 = ComputeValue(samplePts, voxelV[i * 8 + 2]);
            double d3 = ComputeValue(samplePts, voxelV[i * 8 + 3]);
            double d4 = ComputeValue(samplePts, voxelV[i * 8 + 4]);
            double d5 = ComputeValue(samplePts, voxelV[i * 8 + 5]);
            double d6 = ComputeValue(samplePts, voxelV[i * 8 + 6]);
            double d7 = ComputeValue(samplePts, voxelV[i * 8 + 7]);

            //判定它们的状态
            int cubeindex;
            cubeindex = Compact(d0, isoValue);
            cubeindex += Compact(d1, isoValue) * 2;
            cubeindex += Compact(d2, isoValue) * 4;
            cubeindex += Compact(d3, isoValue) * 8;
            cubeindex += Compact(d4, isoValue) * 16;
            cubeindex += Compact(d5, isoValue) * 32;
            cubeindex += Compact(d6, isoValue) * 64;
            cubeindex += Compact(d7, isoValue) * 128;

            //根据表来查出该体素的顶点数
            int numVerts = VertsTable[cubeindex];

            if (i < numVoxels)
            {
                voxelVerts[i] = numVerts;
                if (numVerts > 0)
                {
                    voxelOccupied[i] = 1;
                }
                
            }
        }
        #endregion
        #region compactVoxels
        private void compactVoxels(int[] compactedVoxelArray, int[] voxelOccupied, int[] voxelOccupiedScan, int numVoxels)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            if ((voxelOccupied[i] == 1) && (i < numVoxels))
            {
                compactedVoxelArray[voxelOccupiedScan[i]] = i;
            }
        }
        #endregion
        #region generateTriangles
        private void generateTriangles(double4[] pos, double4[] norm, double3[] model_voxelActive, double3[] vertlist, double3[] normlist,
             int[] verts_voxelActive, double3[] samplePts, double isoValue, int[] VertsTable, int[,] TriangleConnectionTable)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            //计算其场值
            double4 d0 = fieldFunc4(samplePts, model_voxelActive[i * 8]);
            double4 d1 = fieldFunc4(samplePts, model_voxelActive[i * 8 + 1]);
            double4 d2 = fieldFunc4(samplePts, model_voxelActive[i * 8 + 2]);
            double4 d3 = fieldFunc4(samplePts, model_voxelActive[i * 8 + 3]);
            double4 d4 = fieldFunc4(samplePts, model_voxelActive[i * 8 + 4]);
            double4 d5 = fieldFunc4(samplePts, model_voxelActive[i * 8 + 5]);
            double4 d6 = fieldFunc4(samplePts, model_voxelActive[i * 8 + 6]);
            double4 d7 = fieldFunc4(samplePts, model_voxelActive[i * 8 + 7]);

            // 计算判定状态
            int cubeindex;
            cubeindex = Compact(d0.w, isoValue);
            cubeindex += Compact(d1.w, isoValue) * 2;
            cubeindex += Compact(d2.w, isoValue) * 4;
            cubeindex += Compact(d3.w, isoValue) * 8;
            cubeindex += Compact(d4.w, isoValue) * 16;
            cubeindex += Compact(d5.w, isoValue) * 32;
            cubeindex += Compact(d6.w, isoValue) * 64;
            cubeindex += Compact(d7.w, isoValue) * 128;

            //找到位于isosurface的交点，计算顶点在各个边的位置
            vertexInterp2(isoValue, model_voxelActive[i * 8], model_voxelActive[i * 8+1], d0, d1, ref vertlist[0], ref normlist[0]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 1], model_voxelActive[i * 8 + 2], d1, d2, ref vertlist[1], ref normlist[1]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 2], model_voxelActive[i * 8 + 3], d2, d3, ref vertlist[2], ref normlist[2]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 3], model_voxelActive[i * 8], d3, d0, ref vertlist[3], ref normlist[3]);

            vertexInterp2(isoValue, model_voxelActive[i * 8+4], model_voxelActive[i * 8 + 5], d4, d5, ref vertlist[4], ref normlist[4]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 5], model_voxelActive[i * 8 + 6], d5, d6, ref vertlist[5], ref normlist[5]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 6], model_voxelActive[i * 8 + 7], d6, d7, ref vertlist[6], ref normlist[6]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 7], model_voxelActive[i * 8 + 4], d7, d4, ref vertlist[7], ref normlist[7]);

            vertexInterp2(isoValue, model_voxelActive[i * 8], model_voxelActive[i * 8 + 4], d0, d4, ref vertlist[8], ref normlist[8]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 1], model_voxelActive[i * 8 + 5], d1, d5, ref vertlist[9], ref normlist[9]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 2], model_voxelActive[i * 8 + 6], d2, d6, ref vertlist[10], ref normlist[10]);
            vertexInterp2(isoValue, model_voxelActive[i * 8 + 3], model_voxelActive[i * 8 + 7], d3, d7, ref vertlist[11], ref normlist[11]);

            int numVerts = VertsTable[cubeindex];

            //每个voxel有numVerts个顶点
            for (int j = 0; j < numVerts; j++)
            {
                //根据边表找到这些顶点在哪个边上
                int edge = TriangleConnectionTable[cubeindex, j];

                int index = verts_voxelActive[i] + j;

                double4 a = new double4();
                a.x = vertlist[edge].x;
                a.y = vertlist[edge].y;
                a.z = vertlist[edge].z;
                a.w = 1.0;
                pos[index] = a;

                double4 b = new double4();
                b.x = normlist[edge].x;
                b.y = normlist[edge].y;
                b.z = normlist[edge].z;
                b.w = 0.0;
                norm[index] = b;

            }
        }
        #endregion
        #region computeIsosurface

        public List<Point3d> computeIsosurface()
        {
            #region 计算所有voxel的活跃度
            //多kernel的通用线程管理
            int threads = 256;
            Alea.dim3 grid = new Alea.dim3((numVoxels + threads - 1) / threads, 1, 1); //block的数量，维度
            Alea.dim3 block = new Alea.dim3(threads, 1, 1); //thread的数量，维度

            if (grid.x > 65535)
            {
                grid.y = grid.x / 32768;
                grid.x = 32768;
            }

            var deviceId = Device.Default.Id;
            var gpu = Gpu.Get(deviceId);

            var lp = new LaunchParam(grid, block);

            double3[] samplePts = ConvertPointsToDouble3(samplePoints);

            int[] d_voxelVerts = Gpu.Default.Allocate<int>(voxelVerts);
            int[] d_voxelOccupied = Gpu.Default.Allocate<int>(voxelOccupied);
           
            double3[] d_voxelV = Gpu.Default.Allocate<double3>(numVoxels*8);
            double3[] d_samplePts = Gpu.Default.Allocate<double3>(samplePts);

            gpu.Launch(classifyVoxel, lp, d_voxelV,d_voxelVerts, d_voxelOccupied,
                gridSize, numVoxels, voxelSize, isoValue, d_samplePts, Tables.VertsTable);

            voxelVerts = Gpu.CopyToHost(d_voxelVerts);

            //所有单元
            double3[] result_voxelV = Gpu.CopyToHost(d_voxelV);
            //活跃voxel的判定集合
            voxelOccupied = Gpu.CopyToHost(d_voxelOccupied);
            //每个voxel中存在的顶点数
            voxelVerts = Gpu.CopyToHost(d_voxelVerts); 

            Gpu.Free(d_voxelVerts);
            Gpu.Free(d_voxelOccupied);
            Gpu.Free(d_voxelV);
            Gpu.Free(d_samplePts);
            #endregion
            #region 压缩voxel，提取活跃voxel
            //计算活跃voxel的个数
            List<int> index_voxelActiveList = new List<int>();
            for (int i = 0; i < voxelOccupied.Length; i++)
            {
                if (voxelOccupied[i] > 0)
                {
                    index_voxelActiveList.Add(i);
                }
            }

            //活跃voxel索引
            int[] index_voxelActive = index_voxelActiveList.ToArray();
            //活跃voxel数量
            int num_voxelActive = index_voxelActive.Length;

            //活跃voxel模型
            double3[] model_voxelActive = new double3[8*num_voxelActive];
            //活跃voxel中的顶点数
            int[] verts_voxelActive = new int[num_voxelActive];
            //总顶点数
            int sum_Verts = 0;

            for (int i = 0; i < num_voxelActive; i++)
            {
                model_voxelActive[8*i] = result_voxelV[8*index_voxelActive[i]];
                model_voxelActive[8 * i + 1] = result_voxelV[8 * index_voxelActive[i] + 1];
                model_voxelActive[8 * i + 2] = result_voxelV[8 * index_voxelActive[i] + 2];
                model_voxelActive[8 * i + 3] = result_voxelV[8 * index_voxelActive[i] + 3];
                model_voxelActive[8 * i + 4] = result_voxelV[8 * index_voxelActive[i] + 4];
                model_voxelActive[8 * i + 5] = result_voxelV[8 * index_voxelActive[i] + 5];
                model_voxelActive[8 * i + 6] = result_voxelV[8 * index_voxelActive[i] + 6];
                model_voxelActive[8 * i + 7] = result_voxelV[8 * index_voxelActive[i] + 7];
                verts_voxelActive[i] = voxelVerts[index_voxelActive[i]];
                sum_Verts+= voxelVerts[index_voxelActive[i]];
            }
            #endregion
            #region 从活跃voxel提取isosurface
            //多kernel的通用线程管理
            int threads2 = 256;
            Alea.dim3 grid2 = new Alea.dim3((num_voxelActive + threads2 - 1) / threads2, 1, 1); //block的数量，维度
            Alea.dim3 block2 = new Alea.dim3(threads2, 1, 1); //thread的数量，维度

            if (grid2.x > 65535)
            {
                grid2.y = grid2.x / 32768;
                grid2.x = 32768;
            }

            var lp2 = new LaunchParam(grid2, block2);

            pos = new double4[sum_Verts];
            norm = new double4[sum_Verts];
            double3[] vertlist = new double3[12];
            double3[] normlist = new double3[12];

            double3[] d_model_voxelActive = Gpu.Default.Allocate<double3>(model_voxelActive);
            int[] d_verts_voxelActive = Gpu.Default.Allocate<int>(verts_voxelActive);

            double3[] d_samplePts2 = Gpu.Default.Allocate<double3>(samplePts);

            double4[] d_pos = Gpu.Default.Allocate<double4>(pos);
            double4[] d_norm = Gpu.Default.Allocate<double4>(norm);
            double3[] d_vertlist = Gpu.Default.Allocate<double3>(vertlist);
            double3[] d_normlist = Gpu.Default.Allocate<double3>(normlist);

            gpu.Launch(generateTriangles, lp2, d_pos, d_norm, d_model_voxelActive, d_vertlist, d_normlist,  d_verts_voxelActive, 
                d_samplePts2, isoValue, Tables.VertsTable, Tables.TriangleConnectionTable);

            gpu.Synchronize();

            var result = Gpu.CopyToHost(d_pos);

            Gpu.Free(d_vertlist);
            Gpu.Free(d_normlist);
            Gpu.Free(d_samplePts);
            Gpu.Free(d_model_voxelActive);
            Gpu.Free(d_verts_voxelActive);
            Gpu.Free(d_pos);
            Gpu.Free(d_norm);

            #endregion

            List<Point3d> pts = ConvertDouble4ToPoint3d(result);

            return pts;

        }
        #endregion
    }
}
