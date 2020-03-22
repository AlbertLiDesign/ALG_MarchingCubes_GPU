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
            pManager.AddGenericParameter("VoxelValues", "V", "VoxelValues", GH_ParamAccess.item);
            pManager.AddBooleanParameter("GPU", "GPU", "VoxelValues", GH_ParamAccess.item,false);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "", "", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            MarchingCubes_GPU MCgpu = new MarchingCubes_GPU();
            bool gpu_on = false;
            DA.GetData("VoxelValues", ref MCgpu);
            DA.GetData("GPU", ref gpu_on);

            List<Point3d> resultPts = new List<Point3d>();
            if (gpu_on == true)
            {
                resultPts= MCgpu.runExtractIsoSurfaceGPU();
            }
            else
            {
                resultPts= MCgpu.runExtractIsoSurfaceCPU();
            }

            List<Point3d> pts = MCgpu.ConvertDouble3ToPoint3d(MCgpu.model_voxelActive);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            Mesh mesh = BasicFunctions.ExtractMesh(pts);
            sw.Stop();
            double ta = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            GH_Mesh ghm = new GH_Mesh(mesh);
            IGH_GeometricGoo geoResult = BasicFunctions.BoxTrans(MCgpu.TargetBox, MCgpu.SourceBox, ghm);
            GH_Convert.ToMesh(geoResult, ref mesh, GH_Conversion.Both);
            sw.Stop();
            double tb = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            mesh.Vertices.CombineIdentical(true, true);
            mesh.Vertices.CullUnused();
            mesh.Weld(3.1415926535897931);
            sw.Stop();
            double tc = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            mesh.FaceNormals.ComputeFaceNormals();
            mesh.Normals.ComputeNormals();
            sw.Stop();
            double td = sw.Elapsed.TotalMilliseconds;

            DA.SetData("Mesh", mesh);
        }
        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{466A3084-31CD-4E26-9ECA-52C772ABE7DD}"); }
        }
    }
}
