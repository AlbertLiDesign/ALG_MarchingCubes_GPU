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
namespace ALG_MarchingCubes
{
    public class MarchingCubes_GPU
    {
        private static int Num = 128;
        private static double[,] Vertices = new double[8, 3]
          {
             {0.0, 0.0, 0.0},{1.0, 0.0, 0.0},{1.0, 1.0, 0.0},{0.0, 1.0, 0.0},
             {0.0, 0.0, 1.0},{1.0, 0.0, 1.0},{1.0, 1.0, 1.0},{0.0, 1.0, 1.0}
           };
        private static int[,] EdgeConnection = new int[12, 2]
          {
             {0,1}, {1,2}, {2,3}, {3,0},
             {4,5}, {5,6}, {6,7}, {7,4},
             {0,4}, {1,5}, {2,6}, {3,7}
          };
        private static double[,] EdgeDirection = new double[12, 3]
          {
            {1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
            {1.0, 0.0, 0.0},{0.0, 1.0, 0.0},{-1.0, 0.0, 0.0},{0.0, -1.0, 0.0},
            {0.0, 0.0, 1.0},{0.0, 0.0, 1.0},{ 0.0, 0.0, 1.0},{0.0, 0.0, 1.0}
          };
        private static Point3d[] EdgeVertex = new Point3d[12];

        private static double GetVolumeElement(int ld, double[] Volumes, int blockId, int xCount, int yCount, int zCount)
        {
            var globalIdx = globalRow * ld + globalCol;

            if (globalIdx < Volumes.Length)
                return Volumes[globalIdx];
            else
                return 0.0;
        }
        [GpuManaged]
        public static void RunGpuPacked(double[,] a, double[,] b, double[,] c)
        {
            //声明三个二维数组
            var lp = VolumeExtract();
            var aFlat = Pack(a);
            var bFlat = Pack(b);
            var cFlat = new double[c.Length];
            Gpu.Default.Launch(KernelPacked, lp, aFlat, bFlat, cFlat, a.GetLength(1), b.GetLength(1), c.GetLength(1));
            Unpack(cFlat, c);
        }
        private static int DivUp(int num, int den)
        {
            return (num + den - 1) / den;
        }
        private static LaunchParam VolumeExtract(int N)
        {
            //假设原始数据中体素总数为 N,按照每个线程块包含Num个线程的容量定义三维线程grid
            var grid = new Alea.dim3(DivUp(N,Num));
            //定义三维block
            var block = new Alea.dim3(N);
            
            return new LaunchParam(grid, block);
        }
        [GpuManaged]
        private static void VolumeExtract_Kernel(double isovalue, List<Point3d> samplePoints, int xCount, int yCount, int zCount)
        {
            List<Point3d> pts = new List<Point3d>();

            var blockX = blockIdx.x;

            var threadX = threadIdx.x;
            var threadY = threadIdx.y;
            var threadZ = threadIdx.z;

            int N = xCount * yCount * zCount;   //体素总数

            for (int X = 0; X < xCount; X++)
            {
                for (int Y = 0; Y < yCount; Y++)
                {
                    for (int Z = 0; Z < zCount; Z++)
                    {

                    }
                }
            }
            for (int m = 0; m < DivUp(N, Num); m++)
            {
                
            }

            double[] CubeValues = new double[8];
            int flag = 0;

            //计算box的8个顶点的cubeValue，判断是否为活跃volume
            for (int i = 0; i < 8; i++)
            {
                //计算CubeValue，即每个box的8个顶点的iso值
                CubeValues[i] = Dist(fx + Vertices[i, 0] * Scale,
                  fy + Vertices[i, 1] * Scale,
                  fz + Vertices[i, 2] * Scale, samplePoints, Weights);

                //判定顶点状态，与用户指定的iso值比对
                if (CubeValues[i] <= isovalue)
                {
                    flag |= 1 << i;
                }
            }
        }

        private static double Dist(double X, double Y, double Z, List<Point3d> SamplePoints, List<double> Weights)
        {
            double result = 0.0;
            double Dx, Dy, Dz;
            double sum = 0.0;
            foreach (var item in Weights)
            {
                sum += item;
            }

            for (int i = 0; i < SamplePoints.Count; i++)
            {
                Dx = X - SamplePoints[i].X;
                Dy = Y - SamplePoints[i].Y;
                Dz = Z - SamplePoints[i].Z;

                result += (sum * (Weights[i] / sum)) / (Dx * Dx + Dy * Dy + Dz * Dz);
            }
            return result;
        }

        public static double GetOffset(double Value1, double Value2, double ValueDesired)
        {
            if ((Value2 - Value1) == 0.0)
                return 0.5;

            return (ValueDesired - Value1) / (Value2 - Value1);
        }

        public static List<Point3d> MarchCube(double isovalue, double fx, double fy, double fz, double Scale, List<Point3d> SamplePoints, List<double> Weights)
        {
            //检查权重
            if (Weights.Count < SamplePoints.Count)
            {
                List<double> average = new List<double>();
                for (int i = 0; i < SamplePoints.Count; i++)
                {
                    average.Add(1);
                }
                Weights = average;
            }

            List<Point3d> pts = new List<Point3d>();
            double[] CubeValues = new double[8];
            double Offset = 0.0;
            int flag = 0;
            int EdgeFlag = 0;

            //生成每个Box的模型
            for (int i = 0; i < 8; i++)
            {
                //计算CubeValue，即每个box的8个顶点的iso值
                CubeValues[i] = Dist(fx + Vertices[i, 0] * Scale,
                  fy + Vertices[i, 1] * Scale,
                  fz + Vertices[i, 2] * Scale, SamplePoints, Weights);

                //判定顶点状态，与用户指定的iso值比对
                if (CubeValues[i] <= isovalue)
                {
                    flag |= 1 << i;
                }
            }
            //找到哪些几条边和边界相交
            EdgeFlag = Tables.CubeEdgeFlags[flag];


            //如果整个立方体都在边界内，则没有交点
            if (EdgeFlag == 0) return null;

            //找出每条边和边界的相交点，找出在这些交点处的法线量
            for (int i = 0; i < 12; i++)
            {
                if ((EdgeFlag & (1 << i)) != 0) //如果在这条边上有交点
                {
                    Offset = GetOffset(CubeValues[EdgeConnection[i, 0]], CubeValues[EdgeConnection[i, 1]], isovalue);//获得所在边的点的位置的系数

                    //获取边上顶点的坐标
                    EdgeVertex[i].X = fx + (Vertices[EdgeConnection[i, 0], 0] + Offset * EdgeDirection[i, 0]) * Scale;
                    EdgeVertex[i].Y = fy + (Vertices[EdgeConnection[i, 0], 1] + Offset * EdgeDirection[i, 1]) * Scale;
                    EdgeVertex[i].Z = fz + (Vertices[EdgeConnection[i, 0], 2] + Offset * EdgeDirection[i, 2]) * Scale;
                }
            }

            //画出找到的三角形
            for (int Triangle = 0; Triangle < 5; Triangle++)
            {
                if (Tables.TriangleConnectionTable[flag, 3 * Triangle] < 0)
                    break;


                for (int Corner = 0; Corner < 3; Corner++)
                {
                    int Vertex = Tables.TriangleConnectionTable[flag, 3 * Triangle + Corner];
                    Point3d pd = new Point3d(EdgeVertex[Vertex].X, EdgeVertex[Vertex].Y, EdgeVertex[Vertex].Z);
                    pts.Add(pd);
                }
            }
            return pts;
        }
    }
}
