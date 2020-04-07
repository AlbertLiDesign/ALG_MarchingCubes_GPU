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
          : base("Isosurface Extraction", "Isosurface", "Extract isosurface from points using marching cubes algorithm on GPU.", "Mesh", "Triangulation") { }
        public override GH_Exposure Exposure => GH_Exposure.primary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddPointParameter("Point", "P", "Sample points.", GH_ParamAccess.list);
            pManager.AddNumberParameter("Boundary", "B", "The scale of the boundingbox's boundary.", GH_ParamAccess.item, 1.1);
            pManager.AddNumberParameter("VoxelSize", "S", "Voxel Size", GH_ParamAccess.item, 1.0);
            pManager.AddNumberParameter("Isovalue", "Iso", "Isovalue.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Extract isosurface.", GH_ParamAccess.item);
            pManager.AddNumberParameter("Time", "T", "time", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            #region input parameters
            List<Point3d> samplePts = new List<Point3d>();
            double scale = 1.0;
            double boundaryRatio = 2.0;
            double isovalue = 5.0;
            List<double> time = new List<double>();
            Stopwatch sw = new Stopwatch();

            DA.GetDataList("Point", samplePts);
            DA.GetData("Boundary", ref boundaryRatio);
            DA.GetData("VoxelSize", ref scale);
            DA.GetData("Isovalue", ref isovalue);
            #endregion

            #region initialization
            Box box1 = BasicFunctions.CreateUnionBBoxFromGeometry(samplePts, boundaryRatio);

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

            Point3d voxelS = new Point3d(scale, scale, scale);

            var isoSurface = new ExtractIsosurface(baseP, xCount, yCount, zCount, voxelS, (float)scale, (float)isovalue, samplePts);
            #endregion

            sw.Start();
            int num_activeVoxels = 0, num_Voxels = xCount*yCount*zCount;
            List<Point3d> resultPts = new List<Point3d>();
            bool successful = isoSurface.runIsosurface(ref resultPts,  ref num_activeVoxels);

            if (successful == false)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "No eligible isosurface can be extracted, please change isovalue.");
                return;
            }
            this.Message = num_Voxels.ToString();
            sw.Stop();
            double tb = sw.Elapsed.TotalMilliseconds;
            // extract the mesh from result vertices

            sw.Restart();
            Mesh mesh = BasicFunctions.ExtractMesh(resultPts);
            //mesh.Faces.CullDegenerateFaces();
            mesh.FaceNormals.ComputeFaceNormals();
            mesh.Normals.ComputeNormals();
            
            sw.Stop();
            double tc = sw.Elapsed.TotalMilliseconds;

            time.Add(tb);
            time.Add(tc);

            DA.SetData("Mesh", mesh);
            DA.SetDataList("Time", time);
        }

        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{04728D21-346C-4D33-B0DF-0BC34E99CC82}"); }
        }
    }
}