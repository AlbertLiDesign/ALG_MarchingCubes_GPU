using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Rhino.Geometry;

namespace ALG_MarchingCubes.Based_on_CUDA
{
    [StructLayout(LayoutKind.Sequential)]
    struct cfloat3
    {
        public float x, y, z;
    }
    class ExtractIsosurface
    {
        public Point3d basePoint;
        public Point3d voxelSize;
        public int xCount;
        public int yCount;
        public int zCount;
        public float scale;
        public float isoValue;
        public List<Point3d> samplePoints;

        public ExtractIsosurface() { }
        public ExtractIsosurface(Point3d basePoint, int xCount, int yCount, int zCount, Point3d voxelSize,
            float scale, float isoValue, List<Point3d> samplePoints)
        {
            this.basePoint = basePoint;
            this.xCount = xCount;
            this.yCount = yCount;
            this.zCount = zCount;
            this.voxelSize = voxelSize;
            this.scale = scale;
            this.isoValue = isoValue;
            this.samplePoints = samplePoints;
        }

        [DllImport("MarchingCubesDLL.dll", EntryPoint = "computMC")]
        public static extern void computMC(cfloat3 bP, cfloat3 vS, int xCount, int yCount, int zCount,
            float s, float iso, cfloat3[] samplePoints, int sampleCount,  ref uint resultLength, ref uint activeVoxels);
        [DllImport("MarchingCubesDLL.dll", EntryPoint = "getResult")]
        public static extern void getResult(IntPtr result);

        [DllImport("MarchingCubesDLL.dll", EntryPoint = "freeMemory")]
        public static extern void freeMemory(IntPtr a);
        public List<Point3d> runIsosurface(ref int num_activeVoxels)
        {
            int sampleCount = samplePoints.Count();
            cfloat3 bP = ConvertPtToFloat3(basePoint);
            cfloat3 vS = ConvertPtToFloat3(voxelSize);           
            cfloat3[] smaplePts = new cfloat3[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                smaplePts[i] = ConvertPtToFloat3(samplePoints[i]);
            }

            uint resultLength = 0, activeVoxels = 0;
            computMC(bP, vS, xCount, yCount, zCount, scale, isoValue, smaplePts, sampleCount, ref resultLength, ref activeVoxels);

            num_activeVoxels = (int)activeVoxels;
            int size = Marshal.SizeOf(typeof(cfloat3)) * (int)resultLength;
            IntPtr result = Marshal.AllocHGlobal(size);

            getResult(result);
            Point3d[] pts = new Point3d[(int)resultLength];

            Parallel.For (0, (int)resultLength,i=>
            {
                IntPtr pPointor = new IntPtr(result.ToInt64() + Marshal.SizeOf(typeof(cfloat3)) * i);
                pts[i] = ConvertFloat3ToPt((cfloat3)Marshal.PtrToStructure(pPointor, typeof(cfloat3)));
            });
            Marshal.FreeHGlobal(result);
            return pts.ToList() ;
        }

        public cfloat3 ConvertPtToFloat3(Point3d p)
        {
            cfloat3 a = new cfloat3();
            a.x = (float)p.X;
            a.y = (float)p.Y;
            a.z = (float)p.Z;
            return a;
        }
        public Point3d ConvertFloat3ToPt(cfloat3 p)
        {
            Point3d a = new Point3d();
            a.X = p.x;
            a.Y = p.y;
            a.Z = p.z;
            return a;
        }
    }
}
