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
    public class ALG_MC : GH_Component
    {
        public ALG_MC()
          : base("MarchingCubes(ALG)", "ALG_MC", "Create mesh pipes from lines.", "ALG", "MarchingCubes") { }
        public override GH_Exposure Exposure => GH_Exposure.primary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGeometryParameter("Geometries", "G", "Geometries", GH_ParamAccess.list);
            pManager.AddNumberParameter("BoundaryRatio", "B", "BoundaryRatio", GH_ParamAccess.item,2);
            pManager.AddNumberParameter("Scale", "S", "Scale", GH_ParamAccess.item, 1);
            pManager.AddNumberParameter("isoValue", "ISO", "isoValue", GH_ParamAccess.item,5.0);
            pManager.AddBooleanParameter("GPU", "GPU", "GPU", GH_ParamAccess.item,true);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Mesh", GH_ParamAccess.item);
            pManager.AddNumberParameter("Time", "", "", GH_ParamAccess.list);
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
            bool gpu = false;
            List<double> time = new List<double>();

            DA.GetDataList("Geometries", geos);
            DA.GetData("BoundaryRatio", ref boundaryRatio);
            DA.GetData("Scale", ref scale);
            DA.GetData("isoValue", ref isovalue);
            #endregion

            #region 初始化MC数据
            //建立基box
            Box box1 = BasicFunctions.CreateUnionBBoxFromGeometry(geos, boundaryRatio);
            Plane plane = new Plane(Point3d.Origin, Vector3d.XAxis, Vector3d.YAxis);

            //求三个方向上单元的数量
            Interval xD = box1.X;
            Interval yD = box1.Y;
            Interval zD = box1.Z;

            int xCount = (int)Math.Abs(Math.Round((xD.T1 - xD.T0), MidpointRounding.AwayFromZero));
            int yCount = (int)Math.Abs(Math.Round((yD.T1 - yD.T0), MidpointRounding.AwayFromZero));
            int zCount = (int)Math.Abs(Math.Round((zD.T1 - zD.T0), MidpointRounding.AwayFromZero));

            Interval intervalX = new Interval(0, xD.Length);
            Interval intervalY = new Interval(0, yD.Length);
            Interval intervalZ = new Interval(0, zD.Length);

            Point3d[] a = box1.GetCorners();
            List<double> b = new List<double>();
            for (int i = 0; i < 8; i++)
            {
                double t = a[i].X + a[i].Y + a[i].Z;
                b.Add(t);
            }
            Point3d baseP = a[b.IndexOf(b.Min())];

            //建立映射目标box
            Box box2 = new Box(plane, intervalX, intervalY, intervalZ);

            //开始映射
            for (int i = 0; i < geos.Count; i++)
            {
                new_geos.Add(BasicFunctions.BoxTrans(box1, box2, geos[i]));
            }

            //转换几何数据为点数据
            samplePoints = BasicFunctions.ConvertGeosToPoints(new_geos);

            Alea.int3 gridS = new Alea.int3();
            gridS.x = xCount;
            gridS.y = yCount;
            gridS.z = zCount;

            Alea.float3 voxelS = new Alea.float3();
            voxelS.x = (float)scale;
            voxelS.y = (float)scale;
            voxelS.z = (float)scale;

            
            var MCgpu = new MarchingCubes_GPU(baseP, box1, box2, gridS, voxelS, (float)scale, (float)isovalue, samplePoints.ToArray());
            #endregion

            #region 分类体素、扫描体素
            Stopwatch sw = new Stopwatch();
            sw.Start();
            MCgpu.runClassifyVoxel();
            MCgpu.runExtractActiveVoxels();
            sw.Stop();
            double ta = sw.Elapsed.TotalMilliseconds;
            #endregion

            #region 提取Isosurface点集
            sw.Restart();
            List<Point3d> resultPts = new List<Point3d>();
            resultPts = MCgpu.runExtractIsoSurfaceGPU();
            //resultPts = MCgpu.runExtractIsoSurfaceCPU();
            sw.Stop();
            double tb = sw.Elapsed.TotalMilliseconds;
            #endregion

            #region 提取网格、检查网格
            sw.Restart();
            Mesh mesh = BasicFunctions.ExtractMesh(resultPts);
            sw.Stop();
            double tc = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            //GH_Mesh ghm = new GH_Mesh(mesh);
            //IGH_GeometricGoo geoResult = BasicFunctions.BoxTrans(MCgpu.targetBox, MCgpu.sourceBox, ghm);
            //GH_Convert.ToMesh(geoResult, ref mesh, GH_Conversion.Both);

            mesh.Vertices.CombineIdentical(true, true);
            mesh.Vertices.CullUnused();
            mesh.Weld(3.1415926535897931);
            mesh.FaceNormals.ComputeFaceNormals();
            mesh.Normals.ComputeNormals();
            sw.Stop();
            double td = sw.Elapsed.TotalMilliseconds;
            #endregion

            #region 计算运行时间、输出数据
            time.Add(ta);
            time.Add(tb);
            time.Add(tc);
            time.Add(td);
            

            DA.SetDataList("Time", time);
            DA.SetData("Mesh", mesh);
            #endregion
        }
        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{F4D2FDE0-365A-46B5-9AC3-3A4939D5E61A}"); }
        }
    }
}
