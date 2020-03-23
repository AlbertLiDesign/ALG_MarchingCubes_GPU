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

        public static List<Point3d> MarchCube(double isovalue, double fx, double fy, double fz, double Scale, List<Point3d> SamplePoints, List<double> Weights, ref List<double> cubeValue)
        {
            //check weights
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

            for (int i = 0; i < 8; i++)
            {
                //Compute cubeValues of 8 vertices
                CubeValues[i] = Dist(fx + Vertices[i, 0] * Scale,
                  fy + Vertices[i, 1] * Scale,
                  fz + Vertices[i, 2] * Scale, SamplePoints, Weights);

                //Check each vertex state
                if (CubeValues[i] <= isovalue)
                {
                    cubeValue.Add(CubeValues[i]);
                    flag |= 1 << i;
                }
            }
            //find out which edge intersects the isosurface
            EdgeFlag = Tables.CubeEdgeFlags[flag];


            //check whether this voxel is crossed by the isosurface
            if (EdgeFlag == 0) return null;

            for (int j = 0; j < 12; j++)
            {
                //check whether an edge have a point
                if ((EdgeFlag & (1 << j)) != 0) 
                {
                    //compute t values from two end points on each edge
                    Offset = GetOffset(CubeValues[EdgeConnection[j, 0]], CubeValues[EdgeConnection[j, 1]], isovalue);
                    
                    //get positions
                    EdgeVertex[j].X = fx + (Vertices[EdgeConnection[j, 0], 0] + Offset * EdgeDirection[j, 0]) * Scale;
                    EdgeVertex[j].Y = fy + (Vertices[EdgeConnection[j, 0], 1] + Offset * EdgeDirection[j, 1]) * Scale;
                    EdgeVertex[j].Z = fz + (Vertices[EdgeConnection[j, 0], 2] + Offset * EdgeDirection[j, 2]) * Scale;
                    
                }
            }

            //Find out points from each triangle
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