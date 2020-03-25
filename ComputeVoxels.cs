using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Rhino.Geometry;
using Grasshopper.Kernel.Types;
using System.Drawing;
using System.Diagnostics;
using Grasshopper;
using Grasshopper.Kernel.Data;
using System.Linq;

namespace ALG_MarchingCubes
{
    public class ALG_ComputeVoxels : GH_Component
    {
        public ALG_ComputeVoxels()
          : base("ComputeVoxels", "ComputeVoxels", "Compute voxels from points.", "ALG", "MarchingCubes") { }
        public override GH_Exposure Exposure => GH_Exposure.primary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGeometryParameter("Geometries", "G", "Geometries", GH_ParamAccess.list);
            pManager.AddNumberParameter("BoundaryRatio", "B", "BoundaryRatio", GH_ParamAccess.item, 2);
            pManager.AddNumberParameter("Scale", "S", "Scale", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("isoValue", "ISO", "isoValue", GH_ParamAccess.item, 5.0);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddCurveParameter("Boundary", "B", "The boundingbox Boundary of input geometries.", GH_ParamAccess.list);
            pManager.AddGenericParameter("Voxel", "V", "Voxel", GH_ParamAccess.item);
            pManager.AddNumberParameter("Time", "T", "Time", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            #region 输入数据
            List<IGH_GeometricGoo> geos = new List<IGH_GeometricGoo>();
            List<IGH_GeometricGoo> new_geos = new List<IGH_GeometricGoo>();
            List<double> Weights = new List<double>();
            List<Point3d> samplePoints = new List<Point3d>();
            double scale = 1.0;
            double boundaryRatio = 2.0;
            double isovalue = 5.0;
            List<double> time = new List<double>();

            DA.GetDataList("Geometries", geos);
            DA.GetData("BoundaryRatio", ref boundaryRatio);
            DA.GetData("Scale", ref scale);
            DA.GetData("isoValue", ref isovalue);
            #endregion

            #region 初始化MC数据
            //建立基box
            Box box1 = BasicFunctions.CreateUnionBBoxFromGeometry(geos, boundaryRatio);

            //求三个方向上单元的数量
            Interval xD = box1.X;
            Interval yD = box1.Y;
            Interval zD = box1.Z;

            int xCount = (int)Math.Abs(Math.Round((xD.T1 - xD.T0), MidpointRounding.AwayFromZero));
            int yCount = (int)Math.Abs(Math.Round((yD.T1 - yD.T0), MidpointRounding.AwayFromZero));
            int zCount = (int)Math.Abs(Math.Round((zD.T1 - zD.T0), MidpointRounding.AwayFromZero));

            Point3d[] a = box1.GetCorners();
            List<double> b = new List<double>();
            for (int i = 0; i < 8; i++)
            {
                double t = a[i].X + a[i].Y + a[i].Z;
                b.Add(t);
            }
            Point3d baseP = a[b.IndexOf(b.Min())];

            samplePoints = BasicFunctions.ConvertGeosToPoints(geos);

            Alea.int3 gridS = new Alea.int3();
            gridS.x = xCount;
            gridS.y = yCount;
            gridS.z = zCount;

            Alea.float3 voxelS = new Alea.float3();
            voxelS.x = (float)scale;
            voxelS.y = (float)scale;
            voxelS.z = (float)scale;

            var MCgpu = new MarchingCubes_GPU(baseP, box1, gridS, voxelS, (float)scale, (float)isovalue, samplePoints.ToArray());
            #endregion

            #region 分类体素、扫描体素
            Stopwatch sw = new Stopwatch();
            sw.Start();
            MCgpu.runClassifyVoxel();
            sw.Stop();
            double ta = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            //MCgpu.runExtractActiveVoxels();
            sw.Stop();
            double tb = sw.Elapsed.TotalMilliseconds;
            #endregion

            this.Message = MCgpu.numVoxels.ToString();
            #region 计算运行时间、输出数据
            time.Add(ta);
            time.Add(tb);
            List<Line> boundaries = BasicFunctions.GetBoundingBoxBoundaries(MCgpu.sourceBox);
            DA.SetDataList("Boundary", boundaries);
            DA.SetData("Voxel", MCgpu);
            DA.SetDataList("Time", time);
            #endregion
        }

        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{0B62C19C-90C5-4C8F-91E4-D0D77AF21596}"); }
        }
    }
}