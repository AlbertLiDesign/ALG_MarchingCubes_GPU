using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Rhino.Geometry;

namespace ALG.MarchingCubes
{
    [StructLayout(LayoutKind.Sequential)]
    struct cfloat3
    {
        public float x, y, z;
    }
    class ExtractIsosurface
    {
        public Point3d basePoint;
        public int xCount;
        public int yCount;
        public int zCount;
        public float isoValue;
        public List<Point3d> samplePoints;

        public ExtractIsosurface() { }
        public ExtractIsosurface(Point3d basePoint, int xCount, int yCount, int zCount, float isoValue)
        {
            this.basePoint = basePoint;
            this.xCount = xCount;
            this.yCount = yCount;
            this.zCount = zCount;
            this.isoValue = isoValue;
        }

        [DllImport("MarchingCubesDLL.dll", EntryPoint = "computMC")]
        public static extern bool computMC(cfloat3 bP, cfloat3 vS, int xCount, int yCount, int zCount, float iso,  ref uint resultLength);
        [DllImport("MarchingCubesDLL.dll", EntryPoint = "getResult")]
        public static extern void getResult(IntPtr result);
        public bool runIsosurface(float x, float y, float z, ref List<Point3d> vertices, ref int num_activeVoxels)
        {
            cfloat3 bP = ConvertPtToFloat3(basePoint);
            cfloat3 vS = new cfloat3();
            vS.x = x;
            vS.y = y;
            vS.z = z;

            uint resultLength = 0;
            bool successful = computMC(bP, vS, xCount, yCount, zCount, isoValue, ref resultLength);

            if (successful == false)
            {
                return successful;
            }
            else
            {
                int size = Marshal.SizeOf(typeof(cfloat3)) * (int)resultLength;
                IntPtr result = Marshal.AllocHGlobal(size);

                getResult(result);
                Point3d[] pts = new Point3d[(int)resultLength];

                Parallel.For(0, (int)resultLength, i =>
                {
                    IntPtr pPointor = new IntPtr(result.ToInt64() + Marshal.SizeOf(typeof(cfloat3)) * i);
                    pts[i] = ConvertFloat3ToPt((cfloat3)Marshal.PtrToStructure(pPointor, typeof(cfloat3)));
                });
                Marshal.FreeHGlobal(result);
                vertices = pts.ToList();
                return successful;

            }
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
