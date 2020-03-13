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
        // constants
        public  int numVoxels = 0;
        public  int maxVerts = 0;
        public  int activeVoxels = 0;

        public double3 voxelSize;
        public double isoValue = 0.2;
        public double dIsoValue = 0.002;

        public  Alea.int3 gridSizeShift;
        public  Alea.int3 gridSize;
        public  Alea.int3 gridSizeMask;

        // device data
        double4[] d_pos = null, d_normal = null;
        public  int[] d_voxelVerts = null;
        public  int[] d_voxelVertsScan = null;
        public  int[] d_voxelOccupied = null;
        public  int[] d_voxelOccupiedScan = null;
        public  int[] d_compVoxelArray;

        #region classifyVoxel
        //定义一个场函数，输入xyz坐标，返回一个值
        //v = ((3x)^4 - 5(3x)^2 - 5(3y)^2 + (3z)^4 - 5(z)^2 + 11.8) * 0.2 + 0.5
        private double tangle(double x, double y, double z)
        {
            x *= 3.0;
            y *= 3.0;
            z *= 3.0;
            return (x * x * x * x - 5.0 * x * x + y * y * y * y - 5.0 * y * y + z * z * z * z - 5.0 * z * z + 11.8) * 0.2 + 0.5;
        }

        //定义一个场函数，输入一个点的xyz坐标，返回一个值
        private double fieldFunc(Point3d p)
        {
            return tangle(p.X, p.Y, p.Z);
        }
        //根据一维索引计算在三维grid中的位置
        private Alea.int3 calcGridPos(int i, Alea.int3 gridSizeShift, Alea.int3 gridSizeMask)
        {
            Alea.int3 gridPos;
            gridPos.x = i & gridSizeMask.x;
            gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
            gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
            return gridPos;
        }
        private Alea.int3 calcGridPos2(int i, Alea.int3 gridSize)
        {
            Alea.int3 gridPos;

            gridPos.z = i / (gridSize.x * gridSize.y);
            gridPos.y = i % (gridSize.x * gridSize.y) / gridSize.x;
            gridPos.x = i % (gridSize.x * gridSize.y) % gridSize.x;

            return gridPos;
        }
        [GpuManaged]
        private void classifyVoxel(int[] voxelVerts, int[] voxelOccupied, Alea.int3 gridSize,
            int numVoxels, double3 voxelSize, double isoValue)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x; //block在grid中的位置
            int i = blockId * blockDim.x + threadIdx.x; //线程索引

            //计算grid中的位置
            Alea.int3 gridPos = calcGridPos2(i, gridSize);

            Point3d p = new Point3d();

            p.X = -1.0f + (gridPos.x * voxelSize.x);
            p.Y = -1.0f + (gridPos.y * voxelSize.y);
            p.Z = -1.0f + (gridPos.z * voxelSize.z);

            //计算cube中的8个点对应的value
            double[] field = new double[8];
            field[0] = fieldFunc(p);
            field[1] = fieldFunc(p + new Point3d(voxelSize.x, 0, 0));
            field[2] = fieldFunc(p + new Point3d(voxelSize.x, voxelSize.y, 0));
            field[3] = fieldFunc(p + new Point3d(0, voxelSize.y, 0));
            field[4] = fieldFunc(p + new Point3d(0, 0, voxelSize.z));
            field[5] = fieldFunc(p + new Point3d(voxelSize.x, 0, voxelSize.z));
            field[6] = fieldFunc(p + new Point3d(voxelSize.x, voxelSize.y, voxelSize.z));
            field[7] = fieldFunc(p + new Point3d(0, voxelSize.y, voxelSize.z));

            //判定它们的状态
            int cubeindex;
            cubeindex =   Convert.ToInt32(field[0] < isoValue);
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
        private double4 fieldFunc4(Point3d p)
        {
            double v = tangle(p.X, p.Y, p.Z);
            const double d = 0.001f;
            double dx = tangle(p.X + d, p.Y, p.Z) - v;
            double dy = tangle(p.X, p.Y + d, p.Z) - v;
            double dz = tangle(p.X, p.Y, p.Z + d) - v;
            double4 a = new double4();
            a.x = dx;
            a.y = dy;
            a.z = dz;
            a.w = v;
            return a;
        }
        //点的线性插值函数
        private Point3d lerp(Point3d a, Point3d b, double t)
        {
            return a + t * (b - a);
        }
        //浮点数的线性插值函数
        private double lerp(double a, double b, double t)
        {
            return a + t * (b - a);
        }
        //MC顶点的线性插值
        private Point3d vertexInterp(double isolevel, Point3d p0, Point3d p1, float f0, float f1)
        {
            double t = (isolevel - f0) / (f1 - f0);
            return lerp(p0, p1, t);
        }

        // 计算边上的线性插值顶点
        private void vertexInterp2(double isolevel, Point3d p0, Point3d p1, double4 f0, double4 f1, ref Point3d p, ref Point3d n)
        {
            double t = (isolevel - f0.w) / (f1.w - f0.w);
            p = lerp(p0, p1, t);
            n.X = lerp(f0.x, f1.x, t);
            n.Y = lerp(f0.y, f1.y, t);
            n.Z = lerp(f0.z, f1.z, t);
            //    n = normalize(n);
        }
        private void generateTriangles(double4[] pos, double4[] norm,
            int[] compactedVoxelArray, int[] numVertsScanned, Alea.int3 gridSize,
            Alea.int3 gridSizeShift, Alea.int3 gridSizeMask,
                  double3 voxelSize, double isoValue, int activeVoxels, int maxVerts)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            if (i > activeVoxels - 1)
            {
                i = activeVoxels - 1;
            }

            int voxel = i;

            //计算三维grid中的位置
            Alea.int3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

            Point3d p = new Point3d();
            p.X = -1.0f + (gridPos.x * voxelSize.x);
            p.Y = -1.0f + (gridPos.y * voxelSize.y);
            p.Z = -1.0f + (gridPos.z * voxelSize.z);

            //计算每个cube的位置
            Point3d[] v = new Point3d[8];
            v[0] = p;
            v[1] = p + new Point3d(voxelSize.x, 0, 0);
            v[2] = p + new Point3d(voxelSize.x, voxelSize.y, 0);
            v[3] = p + new Point3d(0, voxelSize.y, 0);
            v[4] = p + new Point3d(0, 0, voxelSize.z);
            v[5] = p + new Point3d(voxelSize.x, 0, voxelSize.z);
            v[6] = p + new Point3d(voxelSize.x, voxelSize.y, voxelSize.z);
            v[7] = p + new Point3d(0, voxelSize.y, voxelSize.z);

            // evaluate field values
            double4[] field = new double4[8];
            field[0] = fieldFunc4(v[0]);
            field[1] = fieldFunc4(v[1]);
            field[2] = fieldFunc4(v[2]);
            field[3] = fieldFunc4(v[3]);
            field[4] = fieldFunc4(v[4]);
            field[5] = fieldFunc4(v[5]);
            field[6] = fieldFunc4(v[6]);
            field[7] = fieldFunc4(v[7]);

            // recalculate flag
            // (this is faster than storing it in global memory)
            int cubeindex = 0;
            cubeindex = Convert.ToInt32(field[0].w < isoValue);
            cubeindex += Convert.ToInt32(field[1].w < isoValue) * 2;
            cubeindex += Convert.ToInt32(field[2].w < isoValue) * 4;
            cubeindex += Convert.ToInt32(field[3].w < isoValue) * 8;
            cubeindex += Convert.ToInt32(field[4].w < isoValue) * 16;
            cubeindex += Convert.ToInt32(field[5].w < isoValue) * 32;
            cubeindex += Convert.ToInt32(field[6].w < isoValue) * 64;
            cubeindex += Convert.ToInt32(field[7].w < isoValue) * 128;

            // find the vertices where the surface intersects the cube
            Point3d[] vertlist = new Point3d[12];
            Point3d[] normlist = new Point3d[12];

            vertexInterp2(isoValue, v[0], v[1], field[0], field[1], ref vertlist[0], ref normlist[0]);
            vertexInterp2(isoValue, v[1], v[2], field[1], field[2], ref vertlist[1], ref normlist[1]);
            vertexInterp2(isoValue, v[2], v[3], field[2], field[3], ref vertlist[2], ref normlist[2]);
            vertexInterp2(isoValue, v[3], v[0], field[3], field[0], ref vertlist[3], ref normlist[3]);

            vertexInterp2(isoValue, v[4], v[5], field[4], field[5], ref vertlist[4], ref normlist[4]);
            vertexInterp2(isoValue, v[5], v[6], field[5], field[6], ref vertlist[5], ref normlist[5]);
            vertexInterp2(isoValue, v[6], v[7], field[6], field[7], ref vertlist[6], ref normlist[6]);
            vertexInterp2(isoValue, v[7], v[4], field[7], field[4], ref vertlist[7], ref normlist[7]);

            vertexInterp2(isoValue, v[0], v[4], field[0], field[4], ref vertlist[8], ref normlist[8]);
            vertexInterp2(isoValue, v[1], v[5], field[1], field[5], ref vertlist[9], ref normlist[9]);
            vertexInterp2(isoValue, v[2], v[6], field[2], field[6], ref vertlist[10], ref normlist[10]);
            vertexInterp2(isoValue, v[3], v[7], field[3], field[7], ref vertlist[11], ref normlist[11]);

            int numVerts = Tables.VertsTable[cubeindex];

            for (int j = 0; j < numVerts; j++)
            {
                int edge = Tables.TriangleConnectionTable[cubeindex * 16, j];

                int index = numVertsScanned[voxel] + j;

                if (index < maxVerts)
                {
                    double4 a = new double4();
                    a.x = vertlist[edge].X;
                    a.y = vertlist[edge].Y;
                    a.z = vertlist[edge].Z;
                    a.w = 1.0;
                    pos[index] = a;

                    double4 b = new double4();
                    b.x = normlist[edge].X;
                    b.y = normlist[edge].Y;
                    b.z = normlist[edge].Z;
                    b.w = 0.0;
                    norm[index] = b;
                }
            }
        }
        #endregion
        #region computeIsosurface
        public void computeIsosurface()
        {
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

            gpu.Launch(classifyVoxel, lp, d_voxelVerts, d_voxelOccupied,
                gridSize, numVoxels, voxelSize, isoValue);

        }
        #endregion



        #region matrix
        private const int BlockSize = 32;
        //通过指定行列搜索矩阵元素：对于一维数组构造的矩阵，将矩阵行列根据block的行列所计算出的id来找到对应元素
        private static double GetMatrixElement(int ld, double[] matrix, int blockRow, int blockCol, int row, int col)
        {
            var globalRow = blockRow * BlockSize + row;
            var globalCol = blockCol * BlockSize + col;
            var globalIdx = globalRow * ld + globalCol;
            if (globalIdx < matrix.Length)
                return matrix[globalIdx];
            else
                return 0.0;
        }
        private static void SetMatrixElement(int ld, double[] matrix, int blockRow, int blockCol, int row, int col,
    double value)
        {
            var globalRow = blockRow * BlockSize + row;
            var globalCol = blockCol * BlockSize + col;
            var globalIdx = globalRow * ld + globalCol;
            if (globalIdx < matrix.Length)
                matrix[globalIdx] = value;
        }

        private static int DivUp(int num, int den)
        {
            return (num + den - 1) / den;
        }
        private static void KernelPacked(double[] a, double[] b, double[] c, int colsA, int colsB, int colsC)
        {
            var blockRow = blockIdx.x;
            var blockCol = blockIdx.y;

            var valueC = 0.0;

            var row = threadIdx.x;
            var col = threadIdx.y;

            for (var m = 0; m < DivUp(colsA, BlockSize); ++m)
            {
                var subA = __shared__.Array2D<double>(BlockSize, BlockSize);
                var subB = __shared__.Array2D<double>(BlockSize, BlockSize);

                subA[row, col] = GetMatrixElement(colsA, a, blockRow, m, row, col);
                subB[row, col] = GetMatrixElement(colsB, b, m, blockCol, row, col);
                DeviceFunction.SyncThreads();

                for (var e = 0; e < BlockSize; ++e)
                {
                    valueC += subA[row, e] * subB[e, col];
                }
                DeviceFunction.SyncThreads();
            }

            SetMatrixElement(colsC, c, blockRow, blockCol, row, col, valueC);
        }
        private static double[] Pack(double[,] a)
        {
            var flat = new double[a.Length];
            var rows = a.GetLength(0);
            var cols = a.GetLength(1);
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    flat[i * cols + j] = a[i, j];
            return flat;
        }

        [GpuManaged]
        private static void Unpack(double[] aFlat, double[,] a)
        {
            var rows = a.GetLength(0);
            var cols = a.GetLength(1);
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    a[i, j] = aFlat[i * cols + j];
        }
        private static LaunchParam LaunchParam(double[,] a, double[,] b, double[,] c)
        {
            //定义二维线程数
            var blockSize = new Alea.dim3(BlockSize, BlockSize);
            //定义二维block，这里DivUP是向上取整，相当于ceil操作。例如我们有矩阵A有33列，线程数为32，
            //那么我们需要多分配一个block用来计算，因此向上取整
            var gridSize = new Alea.dim3(DivUp(a.GetLength(0), BlockSize), DivUp(b.GetLength(1), BlockSize));
            return new LaunchParam(gridSize, blockSize);
        }
        static readonly Random rng = new Random(42);
        public static double[,] RandomMatrix(int rows, int cols)
        {
            var a = new double[rows, cols];
            for (var i = 0; i < rows; ++i)
                for (var j = 0; j < cols; ++j)
                    a[i, j] = rng.NextDouble();
            return a;
        }
        [GpuManaged]
        public static void RunGpuPacked(double[,] a, double[,] b, double[,] c)
        {
            //声明三个二维数组
            var lp = LaunchParam(a, b, c);
            var aFlat = Pack(a);
            var bFlat = Pack(b);
            var cFlat = new double[c.Length];
            Gpu.Default.Launch(KernelPacked, lp, aFlat, bFlat, cFlat, a.GetLength(1), b.GetLength(1), c.GetLength(1));
            Unpack(cFlat, c);
        }
        #endregion
    }
}
