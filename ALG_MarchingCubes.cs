using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Rhino.Geometry;
using Grasshopper.Kernel.Types;
using System.Drawing;

namespace ALG_MarchingCubes_GPU
{
    public class ALG_MC : GH_Component
    {
        public ALG_MC()
          : base("MarchingCubes(ALG)", "ALG_MC", "Create mesh pipes from lines.", "ALG", "MarchingCubes") { }
        public override GH_Exposure Exposure => GH_Exposure.primary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGeometryParameter("Geometries", "G", "Geometries", GH_ParamAccess.list);
            pManager.AddNumberParameter("Scale", "S", "Scale", GH_ParamAccess.item,1.2);
            pManager.AddNumberParameter("isoValue", "ISO", "isoValue", GH_ParamAccess.item,5.0);
            pManager.AddIntegerParameter("Count", "Count", "Count", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Mesh", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<IGH_GeometricGoo> geos = new List<IGH_GeometricGoo>();
            List<IGH_GeometricGoo> new_geos = new List<IGH_GeometricGoo>();
            List<double> Weights = new List<double>();
            List<Point3d> samplePoints = new List<Point3d>();
            double scale = 1.0;
            double size = 1.0;
            double isovalue = 5.0;
            int count = 0;

            DA.GetDataList("Geometries",  geos);
            DA.GetData("Scale", ref scale);
            DA.GetData("isoValue", ref isovalue);
            DA.GetData("Count", ref count);

            //建立基box
            Box box1 = BasicFunctions.CreateUnionBBoxFromGeometry(geos, scale);
            Plane plane = new Plane(Point3d.Origin, Vector3d.XAxis, Vector3d.YAxis);
            Interval interval = new Interval(0, 1);

            //建立映射目标box
            Box box2 = new Box(plane, interval, interval, interval);

            //开始映射
            for (int i = 0; i < geos.Count; i++)
            {
                new_geos.Add(BasicFunctions.BoxMapping(box1, box2, geos[i]));
            }

            ////求三个方向上单元的数量
            //Interval xD = box1.X;
            //Interval yD = box1.Y;
            //Interval zD = box1.Z;

            //int xCount = (int)Math.Abs(Math.Round((xD.T1 - xD.T0)/ size, MidpointRounding.AwayFromZero));
            //int yCount = (int)Math.Abs(Math.Round((yD.T1 - yD.T0) / size, MidpointRounding.AwayFromZero));
            //int zCount = (int)Math.Abs(Math.Round((zD.T1 - zD.T0) / size, MidpointRounding.AwayFromZero));

            //转换几何数据为点数据
            samplePoints = BasicFunctions.ConvertGeosToPoints(new_geos);

            //初始化网格数据
            List<Point3d> meshVs = new List<Point3d>();

            //开始计算MC
            for (int X = 0; X < count; X++)
            {
                for (int Y = 0; Y < count; Y++)
                {
                    for (int Z = 0; Z < count; Z++)
                    {
                        List<Point3d> pts = new List<Point3d>();
                        pts = MarchingCubes_CPU.MarchCube(isovalue, X * (1.0 / count), Y * (1.0 / count), Z * (1.0 / count), (1.0 / count), samplePoints, Weights);
                        if (pts != null)
                        {
                            foreach (var item in pts)
                            {
                                meshVs.Add(item);
                            }
                        }
                    }
                }
            }

            Mesh mesh = BasicFunctions.ExtractMesh(meshVs);
            GH_Mesh ghm = new GH_Mesh(mesh);
            IGH_GeometricGoo geoResult = BasicFunctions.BoxMapping(box2, box1, ghm);
            GH_Convert.ToMesh(geoResult, ref mesh, GH_Conversion.Both);

            DA.SetData(0, mesh);
        }
        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{F4D2FDE0-365A-46B5-9AC3-3A4939D5E61A}"); }
        }
    }
}
