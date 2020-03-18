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
        public int[] cubeSum;

        // constants
        public  int numVoxels = 0;
        public  int maxVerts = 0;

        public double3 voxelSize;
        public double isoValue = 0.2;

        public  Alea.int3 gridSize;

        // data
        public double[] cubeValues;
        public int[] cudaIndex;
        public Point3d[] samplePoints = null;
        double3[] pos = null;
        public int[] voxelVerts = null;
        public int[] verts_voxelActive = null;
        public int[] verts_scanIdx = null;
        public int[] voxelOccupied = null;
        public int[] voxelOccupiedScan = null;
        public int[] compactedVoxelArray = null;
        public double scale = 1.0;
        public double[] offsetV;
        public int[] edge_Flags;

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
        public double GetOffset(double Value1, double Value2, double ValueDesired)
        {
            if ((Value2 - Value1) == 0.0)
                return 0.5;

            return (ValueDesired - Value1) / (Value2 - Value1);
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
        private void generateTriangles(double[] d_offset, int[] d_verts_scanIdx, int[] edgeFlags, double3[] pos, double3[] model_voxelActive, int[,] EdgeConnection, int[,] EdgeDirection, double[,] Vertices,
            double3[] samplePts, double isoValue,double scale, int[] EdgeTable, int[,] TriangleConnectionTable,double[] cubeValues)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;
            int flag = 0;
            int EdgeFlag = 0;
            double Offset = 0.0;


            //判定顶点状态，与用户指定的iso值比对
            for (int j = 0; j < 8; j++)
            {
                cubeValues[i * 8 + j] = ComputeValue(samplePts, model_voxelActive[i * 8 + j]);
                if (cubeValues[i * 8 + j] <= isoValue)
                {
                    flag = 0;
                    flag |= 1 << j;//左移相当于乘，这里相当于乘2的j次方
                    edgeFlags[i * 8 + j] = flag;
                }
            }
            DeviceFunction.SyncThreads();


            //找到哪些几条边和边界相交
            EdgeFlag = EdgeTable[flag];
            //edgeFlags[i] = EdgeFlag;

            for (int j = 0; j < 12; j++)
            {
                int num = 0;
                if ((EdgeFlag & (1 << j)) != 0) //如果在这条边上有交点
                {
                    Offset = GetOffset(cubeValues[8 * i + EdgeConnection[j, 0]], cubeValues[8 * i + EdgeConnection[j, 1]], isoValue);//获得所在边的点的位置的系数

                    //获取边上顶点的坐标
                    pos[d_verts_scanIdx[i] + num].x = model_voxelActive[i].x * scale + (Vertices[EdgeConnection[j, 0], 0] + Offset * EdgeDirection[j, 0]) * scale;
                    pos[d_verts_scanIdx[i] + num].y = model_voxelActive[i].y * scale + (Vertices[EdgeConnection[j, 0], 1] + Offset * EdgeDirection[j, 1]) * scale;
                    pos[d_verts_scanIdx[i] + num].z = model_voxelActive[i].z * scale + (Vertices[EdgeConnection[j, 0], 2] + Offset * EdgeDirection[j, 2]) * scale;
                    num++;

                    d_offset[d_verts_scanIdx[i] + num] = Offset;
                }
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
            verts_voxelActive = new int[num_voxelActive];
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

            //扫描以获得最终点索引
            Func<int, int, int> op = Sum;
            Alea.Session session = new Alea.Session(gpu);
            int[] d_verts_voxelActive = Gpu.Default.Allocate<int>(verts_voxelActive);
            int[] d_voxelVertsScan = Gpu.Default.Allocate<int>(verts_voxelActive.Length);

            Alea.Parallel.GpuExtension.Scan<int>(session, d_voxelVertsScan, d_verts_voxelActive, 0, Sum, num_voxelActive);

            var result_Scan = Gpu.CopyToHost(d_voxelVertsScan);

            Gpu.Free(d_verts_voxelActive);
            Gpu.Free(d_voxelVertsScan);

            verts_scanIdx = new int[num_voxelActive];

            for (int i = 1; i < num_voxelActive-1; i++)
            {
                verts_scanIdx[i] = result_Scan[i - 1];
            }
            verts_scanIdx[0] = 0;
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

            pos = new double3[sum_Verts];

            int[] d_verts_scanIdx = Gpu.Default.Allocate<int>(verts_scanIdx);
            double3[] d_pos = Gpu.Default.Allocate<double3>(pos);
            double3[] d_model_voxelActive = Gpu.Default.Allocate<double3>(model_voxelActive);
            int[,] d_EdgeConnection = Gpu.Default.Allocate<int>(12,2);
            int[,] d_EdgeDirection = Gpu.Default.Allocate<int>(12, 3);
            double[,] d_Vertices = Gpu.Default.Allocate<double>(8, 3);
            double3[] d_samplePts2 = Gpu.Default.Allocate<double3>(samplePts);
            double[] d_cubeValues = Gpu.Default.Allocate<double>(8*num_voxelActive);
            int[] d_edge_Flags = Gpu.Default.Allocate<int>(8*num_voxelActive);
            double[] d_offset = Gpu.Default.Allocate<double>(sum_Verts);

            gpu.Launch(generateTriangles, lp2, d_offset, d_verts_scanIdx, d_edge_Flags, d_pos, d_model_voxelActive, d_EdgeConnection, d_EdgeDirection, d_Vertices,
                d_samplePts2, isoValue, 1.0, Tables.CubeEdgeFlags, Tables.TriangleConnectionTable, d_cubeValues);

            var result = Gpu.CopyToHost(d_model_voxelActive);
            edge_Flags = Gpu.CopyToHost(d_edge_Flags);
            offsetV = Gpu.CopyToHost(d_offset);
            cubeValues = Gpu.CopyToHost(d_cubeValues);

            Gpu.Free(edge_Flags);
            Gpu.Free(d_offset);

            Gpu.Free(d_pos);
            Gpu.Free(d_model_voxelActive);
            Gpu.Free(d_cubeValues);
            Gpu.Free(d_EdgeConnection);
            Gpu.Free(d_EdgeDirection);
            Gpu.Free(d_Vertices);
            Gpu.Free(d_samplePts);
            #endregion

            List<Point3d> pts = ConvertDouble3ToPoint3d(result);

            return pts;

        }
        #endregion
    }
}
