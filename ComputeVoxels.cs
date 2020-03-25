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
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
            #region input parameters
            List<IGH_GeometricGoo> geos = new List<IGH_GeometricGoo>();
            double scale = 1.0;
            double boundaryRatio = 2.0;
            double isovalue = 5.0;

            DA.GetDataList("Geometries", geos);
            DA.GetData("BoundaryRatio", ref boundaryRatio);
            DA.GetData("Scale", ref scale);
            DA.GetData("isoValue", ref isovalue);
            #endregion

            #region initialization
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
            Alea.int3 gridS = new Alea.int3(xCount,yCount,zCount);
            Alea.float3 voxelS = new Alea.float3((float)scale, (float)scale, (float)scale);

            var MCgpu = new MarchingCubes_GPU(baseP, box1, gridS, voxelS, (float)scale, (float)isovalue, samplePoints.ToArray());
            #endregion

            #region classify voxel and reduce data
            MCgpu.runClassifyVoxel();
            MCgpu.runExtractActiveVoxels();
            #endregion

            this.Message = MCgpu.numVoxels.ToString();
            #region output voxel data
            List<Line> boundaries = BasicFunctions.GetBoundingBoxBoundaries(MCgpu.sourceBox);
            DA.SetDataList("Boundary", boundaries);
            DA.SetData("Voxel", MCgpu);
            #endregion
        }

        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{0B62C19C-90C5-4C8F-91E4-D0D77AF21596}"); }
        }
    }
}