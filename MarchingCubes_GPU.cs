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


        #region classifyVoxel
        //定义一个场函数，输入xyz坐标，返回一个值
        //v = ((3x)^4 - 5(3x)^2 - 5(3y)^2 + (3z)^4 - 5(z)^2 + 11.8) * 0.2 + 0.5
        private static double tangle(double x, double y, double z)
        {
            x *= 3.0;
            y *= 3.0;
            z *= 3.0;
            return (x * x * x * x - 5.0 * x * x + y * y * y * y - 5.0 * y * y + z * z * z * z - 5.0 * z * z + 11.8) * 0.2 + 0.5;
        }

        //定义一个场函数，输入一个点的xyz坐标，返回一个值
        private static double fieldFunc(Point3d p)
        {
            return tangle(p.X, p.Y, p.Z);
        }
        //根据一维索引计算在三维grid中的位置
        private static int[,,] calcGridPos(int i, int[,,] gridSizeShift, int[,,] gridSizeMask)
        {
            int a = i & gridSizeMask.GetLength(0);
            int b = (i >> gridSizeShift.GetLength(1)) & gridSizeMask.GetLength(1);
            int c = (i >> gridSizeShift.GetLength(2)) & gridSizeMask.GetLength(2);
            return new int[a, b, c];
        }
        [GpuManaged]
        private static void classifyVoxel(int[] voxelVerts, int[] voxelOccupied, int[,,] gridSize,
            int[,,] gridSizeShift, int[,,] gridSizeMask, int numVoxels,
            double[,,] voxelSize, double isoValue)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            //计算grid中的位置
            int[,,] gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

            Point3d p = new Point3d();

            p.X = -1.0f + (gridPos.GetLength(0) * voxelSize.GetLength(0));
            p.Y = -1.0f + (gridPos.GetLength(1) * voxelSize.GetLength(1));
            p.Z = -1.0f + (gridPos.GetLength(2) * voxelSize.GetLength(2));

            //计算cube中的8个点对应的value
            double[] field = new double[8];
            field[0] = fieldFunc(p);
            field[1] = fieldFunc(p + new Point3d(voxelSize.GetLength(0), 0, 0));
            field[2] = fieldFunc(p + new Point3d(voxelSize.GetLength(0), voxelSize.GetLength(1), 0));
            field[3] = fieldFunc(p + new Point3d(0, voxelSize.GetLength(1), 0));
            field[4] = fieldFunc(p + new Point3d(0, 0, voxelSize.GetLength(2)));
            field[5] = fieldFunc(p + new Point3d(voxelSize.GetLength(0), 0, voxelSize.GetLength(2)));
            field[6] = fieldFunc(p + new Point3d(voxelSize.GetLength(0), voxelSize.GetLength(1), voxelSize.GetLength(2)));
            field[7] = fieldFunc(p + new Point3d(0, voxelSize.GetLength(1), voxelSize.GetLength(2)));

            //判定它们的状态
            int cubeindex;
            cubeindex = Convert.ToInt32(field[0] < isoValue);
            cubeindex += Convert.ToInt32(field[1] < isoValue) * 2;
            cubeindex += Convert.ToInt32(field[2] < isoValue) * 4;
            cubeindex += Convert.ToInt32(field[3] < isoValue) * 8;
            cubeindex += Convert.ToInt32(field[4] < isoValue) * 16;
            cubeindex += Convert.ToInt32(field[5] < isoValue) * 32;
            cubeindex += Convert.ToInt32(field[6] < isoValue) * 64;
            cubeindex += Convert.ToInt32(field[7] < isoValue) * 128;

            //根据点表查找状态
            int numVerts = Tables.VertsTable[cubeindex];

            if (i < numVoxels)
            {
                voxelVerts[i] = numVerts;
                voxelOccupied[i] = Convert.ToInt32(numVerts > 0);
            }
        }
        #endregion
        #region compactVoxels
        private static void compactVoxels(int[] compactedVoxelArray, int[] voxelOccupied, int[] voxelOccupiedScan, int numVoxels)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            if ((voxelOccupied[i] == 1) && (i < numVoxels))
            {
                compactedVoxelArray[voxelOccupiedScan[i]] = i;
            }
        }
        #endregion
    }
}
