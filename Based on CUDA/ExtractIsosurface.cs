using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Rhino.Geometry;

namespace ALG_MarchingCubes.Based_on_CUDA
{
    struct float3
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

        [DllImport("MarchingCubesDLL.dll")]
        public static extern float3[] marchingcubesGPU(int sampleCount, float3 bP, float3 vS, int xCount, int yCount, int zCount,float s, float iso, float3[] samplePoints);

        public List<Point3d> runIsosurface()
        {
            List<Point3d> pts = new List<Point3d>();
            int sampleCount = samplePoints.Count();
            float3 bP = ConvertPtToFloat3(basePoint);
            float3 vS = ConvertPtToFloat3(voxelSize);

            float3[] smPts = new float3[sampleCount];
            for (int i = 0; i < sampleCount; i++)
            {
                smPts[i] = ConvertPtToFloat3(samplePoints[i]);
            }

            float3[] result = marchingcubesGPU(sampleCount, bP, vS, xCount, yCount, zCount, scale, isoValue, smPts);
            for (int i = 0; i < result.Length; i++)
            {
                pts.Add(ConvertFloat3ToPt(result[i]));
            }
            return pts;
        }

        public float3 ConvertPtToFloat3(Point3d p)
        {
            float3 a = new float3();
            a.x = (float)p.X;
            a.y = (float)p.Y;
            a.z = (float)p.Z;
            return a;
        }
        public Point3d ConvertFloat3ToPt(float3 p)
        {
            Point3d a = new Point3d();
            a.X = p.x;
            a.Y = p.y;
            a.Z = p.z;
            return a;
        }
    }
}
