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
using System.Threading;
using System.Diagnostics;

namespace ALG_MarchingCubes
{
    public class MarchingCubes_GPU
    {
        public Box SourceBox;
        public Box TargetBox;
        public int[] cubeSum;

        // constants
        public  int numVoxels = 0;
        public  int maxVerts = 0;

        public double3 voxelSize;
        public double isoValue = 0.2;

        public  Alea.int3 gridSize;
        public int[,] gridIndex3d;

        //采样点
        double3[] samplePts;

        //voxel在grid中的索引
        public Alea.int3[] gridIdx;
        //活跃voxel在grid中的索引
        public Alea.int3[] index3d_voxelActive;
        //所有voxel顶点坐标
        public double3[] result_voxelV;

        //活跃voxel的顶点总数
        public int sumVerts;
        //活跃voxel数量
        public int num_voxelActive;
        //活跃voxel的顶点坐标
        public double3[] model_voxelActive;

        //所有voxel的状态
        public int[] cubeindices;
        //voxel缩放倍率
        public double scale;

        public List<Point3d> pp;
        public double[] cubeValues;
        public int[] cudaIndex;
        public Point3d[] samplePoints = null;
        public int[] voxelVerts = null;
        public int[] verts_voxelActive = null;
        public int[] verts_scanIdx = null;
        public int[] voxelOccupied = null;
        public int[] voxelOccupiedScan = null;
        public int[] compactedVoxelArray = null;
        public double[] offsetV;
        public List<double> offsets = new List<double>();
        public int[] edgeFlags;

        private static double[,] Vertices = new double[8, 3]
         {
             {0.0, 0.0, 0.0},{1.0, 0.0, 0.0},{1.0, 1.0, 0.0},{0.0, 1.0, 0.0},
             {0.0, 0.0, 1.0},{1.0, 0.0, 1.0},{1.0, 1.0, 1.0},{0.0, 1.0, 1.0}
         };
        private int[,] EdgeConnection = new int[12, 2]
        {
             {0,1}, {1,2}, {2,3}, {3,0},
             {4,5}, {5,6}, {6,7}, {7,4},
             {0,4}, {1,5}, {2,6}, {3,7}
         };
        private double[,] EdgeDirection = new double[12, 3]
         {
            {1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
            {1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
            {0.0, 0.0, 1.0},{0.0, 0.0, 1.0},{ 0.0, 0.0, 1.0},{0.0, 0.0, 1.0}
         };

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
        public int[,] ConvertInt3ToIntArray(Alea.int3[] a)
        {
            int[,] b = new int[a.Length, 3];
            for (int i = 0; i < a.Length; i++)
            {
                b[i, 0] = a[i].x;
                b[i, 1] = a[i].y;
                b[i, 2] = a[i].z;
            }
            return b;

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
        public List<Point3d> ConvertDouble4ToPoint3d(double4[] array)
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
        private double3 vertexInterp2(double isolevel, double3 p0, double3 p1, double4 f0, double4 f1)
        {
            
            double t = (isolevel - f0.w) / (f1.w - f0.w);
            double3 p = lerp(p0, p1, t);
            //n.x = lerp(f0.x, f1.x, t);
            //n.y = lerp(f0.y, f1.y, t);
            //n.z = lerp(f0.z, f1.z, t);

            return p;
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

            for (int j = 0; j < samplePts.Length; j++)
            {
                Dx = testP.x - samplePts[j].x;
                Dy = testP.y - samplePts[j].y;
                Dz = testP.z - samplePts[j].z;

                result +=1 / (Dx * Dx + Dy * Dy + Dz * Dz);
            }
            return result;
        }
        public double GetOffset(double Value1, double Value2, double ValueDesired)
        {
            if ((Value2 - Value1) == 0.0)
                return 0.5;

            return (ValueDesired - Value1) / (Value2 - Value1);
        }
        #endregion
        public void classifyVoxel(double3[] voxelV, int[] voxelVerts, int[] voxelOccupied, Alea.int3 gridSize,
            int numVoxels, double3 voxelSize, double isoValue,double scale, double3[] samplePts,int[] VertsTable, Alea.int3[] gridIdx)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x; //block在grid中的位置
            int i = blockId * blockDim.x + threadIdx.x; //线程索引

            //计算grid中的位置
            Alea.int3 gridPos = calcGridPos(i, gridSize);
            gridIdx[i] = gridPos;
            double3 p = new double3();

            p.x = gridPos.x * voxelSize.x* scale;
            p.y = gridPos.y * voxelSize.y* scale;
            p.z = gridPos.z * voxelSize.z* scale;

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
        private void computeEdgeFlags(double[] cubeValues, int[] edgeFlags, double3[] model_voxelActive, double3[] samplePts, double isoValue,int[] EdgeTable)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            //判定顶点状态，与用户指定的iso值比对
            cubeValues[i * 8] = ComputeValue(samplePts, model_voxelActive[i * 8]);
            cubeValues[i * 8 + 1] = ComputeValue(samplePts, model_voxelActive[i * 8 + 1]);
            cubeValues[i * 8 + 2] = ComputeValue(samplePts, model_voxelActive[i * 8 + 2]); 
            cubeValues[i * 8 + 3] = ComputeValue(samplePts, model_voxelActive[i * 8 + 3]);
            cubeValues[i * 8 + 4] = ComputeValue(samplePts, model_voxelActive[i * 8 + 4]);
            cubeValues[i * 8 + 5] = ComputeValue(samplePts, model_voxelActive[i * 8 + 5]);
            cubeValues[i * 8 + 6] = ComputeValue(samplePts, model_voxelActive[i * 8 + 6]);
            cubeValues[i * 8 + 7] = ComputeValue(samplePts, model_voxelActive[i * 8 + 7]);

            int flag = Compact(cubeValues[i * 8] , isoValue);
            flag += Compact(cubeValues[i * 8 + 1], isoValue) * 2;
            flag += Compact(cubeValues[i * 8 + 2], isoValue) * 4;
            flag += Compact(cubeValues[i * 8 + 3], isoValue) * 8;
            flag += Compact(cubeValues[i * 8 + 4], isoValue) * 16;
            flag += Compact(cubeValues[i * 8 + 5], isoValue) * 32;
            flag += Compact(cubeValues[i * 8 + 6], isoValue) * 64;
            flag += Compact(cubeValues[i * 8 + 7], isoValue) * 128;

            //找到哪些几条边和边界相交
            edgeFlags[i] = EdgeTable[flag];
        }
        private void generateTriangles(double3[] pos, int[] edgeFlags, Alea.int3[] d_index3d_voxelActive, int[] d_verts_scanIdx, 
         int[,] EdgeConnection, double[,] EdgeDirection, double[,] Vertices,
         double isoValue, double scale, int[,] TriangleConnectionTable, double[] cubeValues, int[] d_k)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            int scanIdx = d_verts_scanIdx[i];
            int k = 0;

            //for (int j = 0; j < 12; i++)
            //{
                
            //    if ((edgeFlags[i] & (1 << j)) != 0) //如果在这条边上有交点
            //    {
            //        double Offset = GetOffset(cubeValues[i * 8 + EdgeConnection[j, 0]], cubeValues[i * 8 + EdgeConnection[j, 1]], isoValue);//获得所在边的点的位置的系数

            //        //获取边上顶点的坐标
            //        pos[scanIdx+k].x = d_index3d_voxelActive[i].x + (Vertices[EdgeConnection[j, 0], 0] + Offset * EdgeDirection[j, 0]) * scale;
            //        pos[scanIdx+k].y = d_index3d_voxelActive[i].y + (Vertices[EdgeConnection[j, 0], 1] + Offset * EdgeDirection[j, 1]) * scale;
            //        pos[scanIdx+k].z = d_index3d_voxelActive[i].z + (Vertices[EdgeConnection[j, 0], 2] + Offset * EdgeDirection[j, 2]) * scale;
            //        k++;
            //    }
            //}
            //d_k[i] = k;
        }

        //计算所有voxel的活跃度
        public void runClassifyVoxel()
        { 
            //多kernel的通用线程管理
            int threads = 128;
            Alea.dim3 grid = new Alea.dim3((numVoxels + threads - 1) / threads, 1, 1); //block的数量，维度
            Alea.dim3 block = new Alea.dim3(threads, 1, 1); //thread的数量，维度

            if (grid.x > 65535)
            {
                grid.y = grid.x / 32768;
                grid.x = 32768;
            }

            var gpu = Gpu.Default;
            var lp = new LaunchParam(grid, block);

            samplePts = ConvertPointsToDouble3(samplePoints);

            int[] d_voxelVerts = gpu.Allocate<int>(numVoxels);
            int[] d_voxelOccupied = gpu.Allocate<int>(numVoxels);
            double3[] d_voxelV = gpu.Allocate<double3>(numVoxels*8);
            double3[] d_samplePts = gpu.Allocate<double3>(samplePts);
            Alea.int3[] d_gridIdx = gpu.Allocate<Alea.int3>(numVoxels);
            int[] d_VertsTable = gpu.Allocate<int>(Tables.VertsTable);

            gpu.Launch(classifyVoxel, lp,d_voxelV, d_voxelVerts, d_voxelOccupied,
                gridSize, numVoxels, voxelSize, isoValue,scale, d_samplePts, d_VertsTable, d_gridIdx);
            
            //所有单元
            result_voxelV = Gpu.CopyToHost(d_voxelV);
            //活跃voxel的判定集合
            voxelOccupied = Gpu.CopyToHost(d_voxelOccupied);
            //每个voxel中存在的顶点数
            voxelVerts = Gpu.CopyToHost(d_voxelVerts);
            //每个voxel的三维索引
            gridIdx = Gpu.CopyToHost(d_gridIdx);

            Gpu.Free(d_voxelV);
            Gpu.Free(d_voxelVerts);
            Gpu.Free(d_voxelOccupied);
            Gpu.Free(d_samplePts);
            Gpu.Free(d_VertsTable);
            Gpu.Free(d_gridIdx);

            gpu.Synchronize();
        }
        //压缩voxel，提取活跃voxel
        public void runExtractActiveVoxels()
        {
            var gpu = Gpu.Default;
            //计算活跃voxel的个数
            List<int> index_voxelActiveList = new List<int>();
            List<Alea.int3> index3d_voxelActiveList = new List<Alea.int3>();
            for (int i = 0; i < voxelOccupied.Length; i++)
            {
                if (voxelOccupied[i] > 0)
                {
                    index_voxelActiveList.Add(i);
                    index3d_voxelActiveList.Add(gridIdx[i]);
                }
            }

            //活跃voxel索引
            int[] index_voxelActive = index_voxelActiveList.ToArray();
            num_voxelActive = index_voxelActive.Length;
            //活跃Voxel在grid中的索引
            index3d_voxelActive = index3d_voxelActiveList.ToArray();
            gridIndex3d = ConvertInt3ToIntArray(index3d_voxelActive);
            //活跃voxel模型
            model_voxelActive = new double3[8 * num_voxelActive];
            //活跃voxel中的顶点数
            verts_voxelActive = new int[num_voxelActive];
            //总顶点数
            sumVerts = 0;


            Parallel.For(0, num_voxelActive, i =>
             {
                 model_voxelActive[8 * i] = result_voxelV[8 * index_voxelActive[i]];
                 model_voxelActive[8 * i + 1] = result_voxelV[8 * index_voxelActive[i] + 1];
                 model_voxelActive[8 * i + 2] = result_voxelV[8 * index_voxelActive[i] + 2];
                 model_voxelActive[8 * i + 3] = result_voxelV[8 * index_voxelActive[i] + 3];
                 model_voxelActive[8 * i + 4] = result_voxelV[8 * index_voxelActive[i] + 4];
                 model_voxelActive[8 * i + 5] = result_voxelV[8 * index_voxelActive[i] + 5];
                 model_voxelActive[8 * i + 6] = result_voxelV[8 * index_voxelActive[i] + 6];
                 model_voxelActive[8 * i + 7] = result_voxelV[8 * index_voxelActive[i] + 7];
                 verts_voxelActive[i] = voxelVerts[index_voxelActive[i]];
             });
            
            //扫描以获得最终点索引
            Func<int, int, int> op = Sum;
            Alea.Session session = new Alea.Session(gpu);
            int[] d_verts_voxelActive = Gpu.Default.Allocate<int>(verts_voxelActive);
            int[] d_voxelVertsScan = Gpu.Default.Allocate<int>(verts_voxelActive.Length);

            Alea.Parallel.GpuExtension.Scan<int>(session, d_voxelVertsScan, d_verts_voxelActive, 0, Sum, num_voxelActive);

            var result_Scan = Gpu.CopyToHost(d_voxelVertsScan);

            verts_scanIdx = new int[num_voxelActive];

            for (int i = 1; i < num_voxelActive; i++)
            {
                verts_scanIdx[i] = result_Scan[i - 1];
            }
            verts_scanIdx[0] = 0;

            sumVerts = verts_scanIdx.ElementAt(verts_scanIdx.Length-1) + verts_voxelActive.ElementAt(verts_voxelActive.Length-1);

            Gpu.Free(d_verts_voxelActive);
            Gpu.Free(d_voxelVertsScan);

            gpu.Synchronize();
            
        }
        public List<Point3d> runExtractIsoSurfaceCPU()
        {
            cubeValues = new double[8 * num_voxelActive];
            edgeFlags = new int[num_voxelActive];

            Point3d[] Apts = new Point3d[sumVerts];

            Parallel.For(0, num_voxelActive, i =>
            //for (int i = 0; i < num_voxelActive; i++)
            {
                Point3d[] pts = new Point3d[12];
                //判定顶点状态，与用户指定的iso值比对
                cubeValues[i * 8] = ComputeValue(samplePts, model_voxelActive[i * 8]);
                cubeValues[i * 8 + 1] = ComputeValue(samplePts, model_voxelActive[i * 8 + 1]);
                cubeValues[i * 8 + 2] = ComputeValue(samplePts, model_voxelActive[i * 8 + 2]);
                cubeValues[i * 8 + 3] = ComputeValue(samplePts, model_voxelActive[i * 8 + 3]);
                cubeValues[i * 8 + 4] = ComputeValue(samplePts, model_voxelActive[i * 8 + 4]);
                cubeValues[i * 8 + 5] = ComputeValue(samplePts, model_voxelActive[i * 8 + 5]);
                cubeValues[i * 8 + 6] = ComputeValue(samplePts, model_voxelActive[i * 8 + 6]);
                cubeValues[i * 8 + 7] = ComputeValue(samplePts, model_voxelActive[i * 8 + 7]);

                int flag = Compact(cubeValues[i * 8], isoValue);
                flag += Compact(cubeValues[i * 8 + 1], isoValue) * 2;
                flag += Compact(cubeValues[i * 8 + 2], isoValue) * 4;
                flag += Compact(cubeValues[i * 8 + 3], isoValue) * 8;
                flag += Compact(cubeValues[i * 8 + 4], isoValue) * 16;
                flag += Compact(cubeValues[i * 8 + 5], isoValue) * 32;
                flag += Compact(cubeValues[i * 8 + 6], isoValue) * 64;
                flag += Compact(cubeValues[i * 8 + 7], isoValue) * 128;

                //找到哪些几条边和边界相交
                int EdgeFlag = Tables.CubeEdgeFlags[flag];
                edgeFlags[i] = EdgeFlag;

                //找出每条边和边界的相交点，找出在这些交点处的法线量
                for (int j = 0; j < 12; j++)
                {
                    if ((EdgeFlag & (1 << j)) != 0) //如果在这条边上有交点
                    {
                        double Offset = GetOffset(cubeValues[i * 8 + EdgeConnection[j, 0]], cubeValues[i * 8 + EdgeConnection[j, 1]], isoValue);//获得所在边的点的位置的系数
                        Point3d pt = new Point3d();
                        //获取边上顶点的坐标
                        pt.X = index3d_voxelActive[i].x + (Vertices[EdgeConnection[j, 0], 0] + Offset * EdgeDirection[j, 0]) * scale;
                        pt.Y = index3d_voxelActive[i].y + (Vertices[EdgeConnection[j, 0], 1] + Offset * EdgeDirection[j, 1]) * scale;
                        pt.Z = index3d_voxelActive[i].z + (Vertices[EdgeConnection[j, 0], 2] + Offset * EdgeDirection[j, 2]) * scale;
                        pts[j] = pt;
                    }
                }

                int num = 0;
                //画出找到的三角形
                for (int Triangle = 0; Triangle < 5; Triangle++)
                {
                    if (Tables.TriangleConnectionTable[flag, 3 * Triangle] < 0)
                        break;


                    for (int Corner = 0; Corner < 3; Corner++)
                    {
                        int Vertex = Tables.TriangleConnectionTable[flag, 3 * Triangle + Corner];
                        Point3d pd = new Point3d(pts[Vertex].X, pts[Vertex].Y, pts[Vertex].Z);
                        Apts[verts_scanIdx[i] + num] = pd;
                        num++;
                    }
                }
            });
            return Apts.ToList() ;
        }
        public List<Point3d> runExtractIsoSurfaceGPU()
        {
            var gpu = Gpu.Default;

            edgeFlags = new int[num_voxelActive];
            double3[] pts = new double3[12 * num_voxelActive];
            double3[] d_pts = Gpu.Default.Allocate<double3>(pts);
            double3[] Apts = Gpu.Default.Allocate<double3>(sumVerts);
            int[] d_edgeFlags = Gpu.Default.Allocate<int>(edgeFlags);

            Alea.int3[] d_index3d_voxelActive = Gpu.Default.Allocate<Alea.int3>(index3d_voxelActive);
            double3[] d_model_voxelActive = Gpu.Default.Allocate<double3>(model_voxelActive);
            double3[] d_samplePts = Gpu.Default.Allocate<double3>(samplePts);
            double[] d_cubeValues = Gpu.Default.Allocate<double>(8 * num_voxelActive);
            int[] d_verts_scanIdx = Gpu.Default.Allocate<int>(verts_scanIdx);

            double[,] d_Vertices = Gpu.Default.Allocate<double>(Vertices);
            double[,] d_EdgeDirection = Gpu.Default.Allocate<double>(EdgeDirection);
            int[,] d_EdgeConnection = Gpu.Default.Allocate<int>(EdgeConnection);
            int[,] d_TriTable = Gpu.Default.Allocate<int>(Tables.TriangleConnectionTable);
            int[] d_EdgeTable = Gpu.Default.Allocate<int>(Tables.CubeEdgeFlags);

            double[] numbers = new double[2];
            numbers[0] = isoValue;
            numbers[1] = scale;
            double[] d_numbers = Gpu.Default.Allocate<double>(numbers);

            gpu.For(0, num_voxelActive, i =>
             {
                 //判定顶点状态，与用户指定的iso值比对
                 d_cubeValues[i * 8] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8]) * d_numbers[1];
                 d_cubeValues[i * 8 + 1] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 1]) * d_numbers[1];
                 d_cubeValues[i * 8 + 2] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 2]) * d_numbers[1];
                 d_cubeValues[i * 8 + 3] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 3]) * d_numbers[1];
                 d_cubeValues[i * 8 + 4] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 4]) * d_numbers[1];
                 d_cubeValues[i * 8 + 5] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 5]) * d_numbers[1];
                 d_cubeValues[i * 8 + 6] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 6]) * d_numbers[1];
                 d_cubeValues[i * 8 + 7] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 7]) * d_numbers[1];

                 int flag = Compact(d_cubeValues[i * 8], d_numbers[0]);
                 flag += Compact(d_cubeValues[i * 8 + 1], d_numbers[0]) * 2;
                 flag += Compact(d_cubeValues[i * 8 + 2], d_numbers[0]) * 4;
                 flag += Compact(d_cubeValues[i * 8 + 3], d_numbers[0]) * 8;
                 flag += Compact(d_cubeValues[i * 8 + 4], d_numbers[0]) * 16;
                 flag += Compact(d_cubeValues[i * 8 + 5], d_numbers[0]) * 32;
                 flag += Compact(d_cubeValues[i * 8 + 6], d_numbers[0]) * 64;
                 flag += Compact(d_cubeValues[i * 8 + 7], d_numbers[0]) * 128;

                 //找到哪些几条边和边界相交
                 int EdgeFlag = d_EdgeTable[flag];
                 d_edgeFlags[i] = EdgeFlag;

                 //找出每条边和边界的相交点，找出在这些交点处的法线量
                 for (int j = 0; j < 12; j++)
                 {
                     if ((EdgeFlag & (1 << j)) != 0) //如果在这条边上有交点
                    {
                         double Offset = GetOffset(d_cubeValues[i * 8 + d_EdgeConnection[j, 0]], d_cubeValues[i * 8 + d_EdgeConnection[j, 1]], d_numbers[0]);//获得所在边的点的位置的系数
                        double3 pt = new double3();
                        //获取边上顶点的坐标
                        pt.x = d_index3d_voxelActive[i].x + (d_Vertices[d_EdgeConnection[j, 0], 0] + Offset * d_EdgeDirection[j, 0]) * d_numbers[1];
                         pt.y = d_index3d_voxelActive[i].y + (d_Vertices[d_EdgeConnection[j, 0], 1] + Offset * d_EdgeDirection[j, 1]) * d_numbers[1];
                         pt.z = d_index3d_voxelActive[i].z + (d_Vertices[d_EdgeConnection[j, 0], 2] + Offset * d_EdgeDirection[j, 2]) * d_numbers[1];
                         d_pts[12*i+j] = pt;
                     }
                 }
                 int num = 0;
                //画出找到的三角形
                for (int Triangle = 0; Triangle < 5; Triangle++)
                 {
                     if (d_TriTable[flag, 3 * Triangle] < 0)
                         break;

                     for (int Corner = 0; Corner < 3; Corner++)
                     { 
                         int Vertex = d_TriTable[flag, 3 * Triangle + Corner];
                         double3 pd = CreateDouble3(d_pts[12*i+Vertex].x, d_pts[12 * i + Vertex].y, d_pts[12 * i + Vertex].z);
                         Apts[d_verts_scanIdx[i] + num] = pd;
                         num++;
                     }
                 }
             });
            var result_Scan = Gpu.CopyToHost(Apts);
            edgeFlags = Gpu.CopyToHost(d_edgeFlags);
            return ConvertDouble3ToPoint3d(result_Scan);
        }

        public void runComputeEdgeFlags()
        {
            var gpu = Gpu.Default;
            //多kernel的通用线程管理
            int threads2 = 128;
            Alea.dim3 grid2 = new Alea.dim3((num_voxelActive + threads2 - 1) / threads2, 1, 1); //block的数量，维度
            Alea.dim3 block2 = new Alea.dim3(threads2, 1, 1); //thread的数量，维度

            if (grid2.x > 65535)
            {
                grid2.y = grid2.x / 32768;
                grid2.x = 32768;
            }

            var lp2 = new LaunchParam(grid2, block2);

            cubeValues = new double[8 * num_voxelActive];
            edgeFlags = new int[num_voxelActive];

            //声明输出参数
            double[] d_cubeValues = Gpu.Default.Allocate<double>(cubeValues);
            int[] d_edgeFlags = Gpu.Default.Allocate<int>(edgeFlags);

            //拷贝输入参数
            double3[] d_model_voxelActive = Gpu.Default.Allocate<double3>(model_voxelActive);
            int[] d_CubeEdgeFlags = Gpu.Default.Allocate<int>(Tables.CubeEdgeFlags);
            double3[] d_samplePts = gpu.Allocate<double3>(samplePts);
            //共5个设备端参数
            gpu.Launch(computeEdgeFlags, lp2, d_cubeValues, d_edgeFlags, d_model_voxelActive, d_samplePts, isoValue, d_CubeEdgeFlags);
            model_voxelActive = Gpu.CopyToHost(d_model_voxelActive);//输出活跃box坐标
            edgeFlags = Gpu.CopyToHost(d_edgeFlags);//输出每个活跃体素的边状态
            cubeValues = Gpu.CopyToHost(d_cubeValues);//每个cube的值

            Gpu.Free(d_model_voxelActive);
            Gpu.Free(d_CubeEdgeFlags);
            Gpu.Free(d_samplePts);
        }
        //从活跃voxel提取isosurface
        public void runComputeIsosurface()
        {
            var gpu = Gpu.Default;
            //多kernel的通用线程管理
            int threads2 = 128;
            Alea.dim3 grid2 = new Alea.dim3((num_voxelActive + threads2 - 1) / threads2, 1, 1); //block的数量，维度
            Alea.dim3 block2 = new Alea.dim3(threads2, 1, 1); //thread的数量，维度

            if (grid2.x > 65535)
            {
                grid2.y = grid2.x / 32768;
                grid2.x = 32768;
            }

            var lp2 = new LaunchParam(grid2, block2);

            //输出参数
            double3[] d_pos = Gpu.Default.Allocate<double3>(sumVerts);

            int[] d_k = Gpu.Default.Allocate<int>(num_voxelActive);

            //输入参数
            Alea.int3[] d_index3d_voxelActive = Gpu.Default.Allocate<Alea.int3>(index3d_voxelActive);
            int[] d_verts_scanIdx = Gpu.Default.Allocate<int>(verts_scanIdx);
            int[,] d_EdgeConnection = Gpu.Default.Allocate<int>(EdgeConnection);
            double[,] d_EdgeDirection = Gpu.Default.Allocate<double>(EdgeDirection);
            double[,] d_Vertices = Gpu.Default.Allocate<double>(Vertices);
            int[,] d_TriTable = Gpu.Default.Allocate<int>(Tables.TriangleConnectionTable);
            int[] d_edgeFlags = Gpu.Default.Allocate<int>(edgeFlags);
            double[] d_cubeValues = Gpu.Default.Allocate<double>(cubeValues);

            //共9个参数
            gpu.Launch(generateTriangles, lp2, d_pos, d_edgeFlags, d_index3d_voxelActive, d_verts_scanIdx,
                d_EdgeConnection, d_EdgeDirection, d_Vertices, isoValue, scale, d_TriTable, d_cubeValues, d_k);
            var result2 = Gpu.CopyToHost(d_k);//输出最终坐标
            //pp = ConvertDouble3ToPoint3d(result2);

        }

        public List<Point3d> runGPU_MC(ref List<double> time)
        {
            
            Stopwatch sw = new Stopwatch();
            sw.Start();
            runClassifyVoxel();
            sw.Stop();
            double ta = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            runExtractActiveVoxels();
            sw.Stop();
            double tb = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            //runComputeEdgeFlags();
            //runComputeIsosurface();
            //List<Point3d> pts = ConvertDouble3ToPoint3d(model_voxelActive);
            List<Point3d> pts = runExtractIsoSurfaceCPU();
            //List<Point3d> pts = new List<Point3d>();
            sw.Stop();
            double tc = sw.Elapsed.TotalMilliseconds;

            time.Add(ta);
            time.Add(tb);
            time.Add(tc);

            return pts;
        }
    }
}
