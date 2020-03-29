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
    public class ALG_TestComponent : GH_Component
    {
        public ALG_TestComponent()
          : base("TestComponent", "TestComponent", "TestComponent", "ALG", "MarchingCubes") { }
        public override GH_Exposure Exposure => GH_Exposure.primary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Voxels", "V", "Voxels", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddPointParameter("testPoint", "", "", GH_ParamAccess.list);
            pManager.AddNumberParameter("verts_scanIdx", "", "", GH_ParamAccess.list);
            pManager.AddNumberParameter("cubeValues", "", "", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            MarchingCubes_GPU MCgpu = new MarchingCubes_GPU();
            DA.GetData("Voxels", ref MCgpu);

            //MCgpu.ConvertFloat3ToPoint3d(MCgpu.testPoint)
            DA.SetDataList("testPoint",null );
            DA.SetDataList("verts_scanIdx", MCgpu.verts_scanIdx);
            DA.SetDataList("cubeValues", MCgpu.cubeValues);
        }
        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{8522153F-E14A-4C93-8273-63392475BFF5}"); }
        }
    }
}