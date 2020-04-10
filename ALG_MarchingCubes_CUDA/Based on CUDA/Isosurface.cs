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

namespace ALG.MarchingCubes
{
    public class ALG_Isosurface : GH_Component
    {
        public ALG_Isosurface()
          : base("MarchingCubes", "MarchingCubes", "Extract isosurface from points using marching cubes algorithm on GPU.", "Mesh", "Triangulation") { }
        public override GH_Exposure Exposure => GH_Exposure.primary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddBoxParameter("Box", "B", "Define the boundary of marching cubes.", GH_ParamAccess.item);
            pManager.AddIntegerParameter("X", "X", "The number of voxels in X asis", GH_ParamAccess.item, 10);
            pManager.AddIntegerParameter("Y", "Y", "The number of voxels in Y asis", GH_ParamAccess.item, 10);
            pManager.AddIntegerParameter("Z", "Z", "The number of voxels in Z asis", GH_ParamAccess.item, 10);
            pManager.AddNumberParameter("Isovalue", "Iso", "Isovalue.", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Extract isosurface.", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            #region input parameters
            Box box1 = new Box();
            int xCount = 0, yCount = 0, zCount = 0;
            double isovalue = 5.0;

            DA.GetData("Box", ref box1);
            DA.GetData("X", ref xCount);
            DA.GetData("Y", ref yCount);
            DA.GetData("Z", ref zCount);
            DA.GetData("Isovalue", ref isovalue);
            #endregion

            #region initialization
            Interval xD = box1.X;
            Interval yD = box1.Y;
            Interval zD = box1.Z;

            Point3d[] a = box1.GetCorners();
            List<double> b = new List<double>();
            for (int i = 0; i < 8; i++)
            {
                double t = a[i].X + a[i].Y + a[i].Z;
                b.Add(t);
            }
            Point3d baseP = a[b.IndexOf(b.Min())];

            float xSize = (float)(xD.Length / xCount);
            float ySize = (float)(yD.Length / yCount);
            float zSize = (float)(zD.Length / zCount);

            var isoSurface = new ExtractIsosurface(baseP, xCount, yCount, zCount, (float)isovalue);
            #endregion

            int num_activeVoxels = 0, num_Voxels = xCount*yCount*zCount;
            List<Point3d> resultPts = new List<Point3d>();
            bool successful = isoSurface.runIsosurface(xSize,ySize, zSize, ref resultPts,  ref num_activeVoxels);

            if (successful == false)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "No eligible isosurface can be extracted, please change isovalue.");
                return;
            }
            this.Message = num_Voxels.ToString();
            // extract the mesh from result vertices

            Mesh mesh = BasicFunctions.ExtractMesh(resultPts);
            //mesh.Faces.CullDegenerateFaces();
            mesh.FaceNormals.ComputeFaceNormals();
            mesh.Normals.ComputeNormals();

            DA.SetData("Mesh", mesh);
        }

        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{04728D21-346C-4D33-B0DF-0BC34E99CC82}"); }
        }
    }
}