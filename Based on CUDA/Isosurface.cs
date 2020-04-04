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
using ALG_MarchingCubes.Based_on_CUDA;

namespace ALG_MarchingCubes
{
    public class ALG_Isosurface : GH_Component
    {
        public ALG_Isosurface()
          : base("Isosurface", "Isosurface", "Compute voxels from points.", "ALG", "MarchingCubes") { }
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
            pManager.AddMeshParameter("Mesh", "P", "The boundingbox Boundary of input geometries.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Time", "T", "Time", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            #region input parameters
            List<IGH_GeometricGoo> geos = new List<IGH_GeometricGoo>();
            double scale = 1.0;
            double boundaryRatio = 2.0;
            double isovalue = 5.0;
            List<double> time = new List<double>();

            DA.GetDataList("Geometries", geos);
            DA.GetData("BoundaryRatio", ref boundaryRatio);
            DA.GetData("Scale", ref scale);
            DA.GetData("isoValue", ref isovalue);
            #endregion

            #region initialization
            Stopwatch sw = new Stopwatch();
            Box box1 = BasicFunctions.CreateUnionBBoxFromGeometry(geos, boundaryRatio);

            Interval xD = box1.X;
            Interval yD = box1.Y;
            Interval zD = box1.Z;

            int xCount = (int)Math.Abs(Math.Round(((xD.T1 - xD.T0) / scale), MidpointRounding.AwayFromZero));
            int yCount = (int)Math.Abs(Math.Round(((yD.T1 - yD.T0) / scale), MidpointRounding.AwayFromZero));
            int zCount = (int)Math.Abs(Math.Round(((zD.T1 - zD.T0) / scale), MidpointRounding.AwayFromZero));

            Point3d[] a = box1.GetCorners();
            List<double> b = new List<double>();
            for (int i = 0; i < 8; i++)
            {
                double t = a[i].X + a[i].Y + a[i].Z;
                b.Add(t);
            }
            Point3d baseP = a[b.IndexOf(b.Min())];

            List<Point3d> samplePoints = BasicFunctions.ConvertGeosToPoints(geos);
            Point3d voxelS = new Point3d(scale, scale, scale);

            var isoSurface = new ExtractIsosurface(baseP, xCount, yCount, zCount, voxelS, (float)scale, (float)isovalue, samplePoints);
            #endregion

            sw.Restart();
            int num_activeVoxels = 0, num_Voxels = xCount*yCount*zCount;
            List<Point3d> resultPts = isoSurface.runIsosurface(ref num_activeVoxels);

            this.Message = num_Voxels.ToString();
            sw.Stop();
            double tb = sw.Elapsed.TotalMilliseconds;

            #region extract the mesh from result vertices
            sw.Restart();
            Mesh mesh = BasicFunctions.ExtractMesh(resultPts);
            sw.Stop();
            double ta = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
                mesh.Faces.CullDegenerateFaces();
                mesh.FaceNormals.ComputeFaceNormals();
                mesh.Normals.ComputeNormals();
            sw.Stop();
            double tc = sw.Elapsed.TotalMilliseconds;
            #endregion

            time.Add(ta);
            time.Add(tb);
            time.Add(tc);

            DA.SetData("Mesh", mesh);
            DA.SetDataList("Time", time);
        }

        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{0F2A4148-B516-453E-A589-4FD9BFD13D42}"); }
        }
    }
}