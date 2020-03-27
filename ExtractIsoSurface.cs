using System;
using System.Collections.Generic;
using Grasshopper.Kernel;
using Rhino.Geometry;
using Grasshopper.Kernel.Types;
using System.Drawing;
using System.Diagnostics;
using Grasshopper;
using Grasshopper.Kernel.Data;

namespace ALG_MarchingCubes
{
    public class ALG_ExtractIsosurface : GH_Component
    {
        public ALG_ExtractIsosurface()
          : base("ExtractIsosurface", "ExtractIsosurface", "Extract Isosurface", "ALG", "MarchingCubes") { }
        public override GH_Exposure Exposure => GH_Exposure.primary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Voxels", "V", "Voxels", GH_ParamAccess.item);
            pManager.AddBooleanParameter("Weld", "W", "Weld mesh", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "", "", GH_ParamAccess.list);
            pManager.AddNumberParameter("Time", "T", "Time", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            MarchingCubes_GPU MCgpu = new MarchingCubes_GPU();
            bool weld = false;
            DA.GetData("Voxels", ref MCgpu);
            DA.GetData("Weld", ref weld);
            Stopwatch sw = new Stopwatch();

            List<double> time = new List<double>();
            #region extract all vertices of the Isosurface
            sw.Start();
            List<Point3d> resultPts = new List<Point3d>();
            resultPts = MCgpu.runExtractIsoSurfaceGPU();
            sw.Stop();
            double ta = sw.Elapsed.TotalMilliseconds;
            #endregion

            #region extract the mesh from result vertices
            sw.Restart();
            Mesh mesh = BasicFunctions.ExtractMesh(resultPts);
            sw.Stop();
            double tb = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            if (weld)
            {
                mesh.Faces.CullDegenerateFaces();
                mesh.Vertices.CombineIdentical(true, true);
                mesh.Vertices.CullUnused();
                mesh.Weld(3.1415926535897931);
                mesh.FaceNormals.ComputeFaceNormals();
                mesh.Normals.ComputeNormals();
            }
            else
            {
                mesh.Faces.CullDegenerateFaces();
                mesh.FaceNormals.ComputeFaceNormals();
                mesh.Normals.ComputeNormals();
            }
            sw.Stop();
            double tc = sw.Elapsed.TotalMilliseconds;
            #endregion
            this.Message = MCgpu.num_voxelActive.ToString();

            time.Add(ta);
            time.Add(tb);
            time.Add(tc);

            DA.SetData("Mesh", mesh);
            DA.SetDataList("Time", time);
        }
        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{466A3084-31CD-4E26-9ECA-52C772ABE7DD}"); }
        }
    }
}