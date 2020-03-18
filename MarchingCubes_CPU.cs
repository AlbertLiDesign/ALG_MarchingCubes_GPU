using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Grasshopper.Kernel.Types.Transforms;
using Rhino.Geometry;

namespace ALG_MarchingCubes
{
        public class MarchingCubes_CPU
    {
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
        private static Point3d[] EdgeNorm = new Point3d[12];


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

        private static double GetOffset(double Value1, double Value2, double ValueDesired)
        {
            if ((Value2 - Value1) == 0.0)
                return 0.5;

            return (ValueDesired - Value1) / (Value2 - Value1);
        }

        public static List<Point3d> MarchCube(double isovalue, double fx, double fy, double fz, double Scale, List<Point3d> SamplePoints, List<double> Weights,ref List<double> ddd)
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
                    ddd.Add(Offset);
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
