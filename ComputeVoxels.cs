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
            pManager.AddGenericParameter("VoxelValues", "V", "Voxels", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {
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

            //建立映射目标box
            Box box2 = new Box(plane, intervalX, intervalY, intervalZ);

            //开始映射
            for (int i = 0; i < geos.Count; i++)
            {
                new_geos.Add(BasicFunctions.BoxTrans(box1, box2, geos[i]));
            }

            //转换几何数据为点数据
            samplePoints = BasicFunctions.ConvertGeosToPoints(new_geos);

            MarchingCubes_GPU MCgpu = new MarchingCubes_GPU();

            MCgpu.SourceBox = box1;
            MCgpu.TargetBox = box2;

            Alea.int3 gridS = new Alea.int3();
            gridS.x = xCount;
            gridS.y = yCount;
            gridS.z = zCount;
            MCgpu.gridSize = gridS;

            Alea.CudaToolkit.double3 voxelS = new Alea.CudaToolkit.double3();
            voxelS.x = 1 * scale;
            voxelS.y = 1 * scale;
            voxelS.z = 1 * scale;
            MCgpu.voxelSize = voxelS;

            MCgpu.numVoxels = MCgpu.gridSize.x * MCgpu.gridSize.y * MCgpu.gridSize.z;
            MCgpu.scale = scale;
            MCgpu.isoValue = isovalue;
            MCgpu.samplePoints = samplePoints.ToArray();

            MCgpu.runClassifyVoxel();
            MCgpu.runExtractActiveVoxels();

            DA.SetData("VoxelValues", MCgpu);
        }
        protected override Bitmap Icon => null;
        public override Guid ComponentGuid
        {
            get { return new Guid("{0B62C19C-90C5-4C8F-91E4-D0D77AF21596}"); }
        }
    }
}
