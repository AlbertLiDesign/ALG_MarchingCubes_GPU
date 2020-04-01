using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;
using Rhino.Geometry;
using System.Diagnostics;
using Plankton;
using PlanktonGh;
using System.Threading.Tasks;
using System.Linq;

namespace ALG_MarchingCubes
{
    public class MeshChecker : GH_Component
    {
        public Rhino.Geometry.Collections.MeshVertexList MV;
        public Rhino.Geometry.Collections.MeshTopologyVertexList MTV;
        public Rhino.Geometry.Collections.MeshTopologyEdgeList MTE;
        public Rhino.Geometry.Collections.MeshFaceList MF;
        public MeshChecker()
          : base("MeshChecker", "MC", "Check the quality of your mesh. ", "ALG", "MeshChecker") { }
        public override GH_Exposure Exposure => GH_Exposure.secondary;
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddMeshParameter("Mesh", "M", "Input a mesh.", GH_ParamAccess.list);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddCurveParameter("Non-Manifold Edges", "NME", "Output non-manifold edges.", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Non-Manifold Vertices Indices", "NMVI", "Output indices of non-manifold vertices.", GH_ParamAccess.list);
        }

        public override Guid ComponentGuid
        {
            get { return new Guid("{818AB035-1E66-4E04-AE1D-B5FB51130068}"); }
        }
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<Mesh> ms = new List<Mesh>();
            DA.GetDataList("Mesh", ms);
            List<int> NMVI = new List<int>();
            List<Line> NME = new List<Line>();
            List<int> NMEI = new List<int>();

            Mesh M = new Mesh();
            string report = null;
            List<string> reports = new List<string>();

            foreach (var item in ms)
            {
                M.Append(item);
            }

            if (M == null || !M.IsValid)
            {
                report = "This is a invalid mesh.";
                Message = report;
                return;
            }

            MV = M.Vertices;
            MTV = M.TopologyVertices;
            MTE = M.TopologyEdges;

            if (M.DisjointMeshCount > 1)
            {
                reports.Add(string.Format("Disjoined Pieces: {0}", M.DisjointMeshCount));
            }

            NMVI = NonManifoldVertices(M);

            NME = new List<Line>();//非流形边
            NMEI = NonManifoldEdges(M, ref NME);//非流形边索引

            if (NMVI.Count + NMEI.Count == 0)
            {
                bool hasBoundaries = hasBoundary(M);
                if (hasBoundaries == true)//外露边检查
                {
                    Polyline[] pls = M.GetNakedEdges();
                    int num2 = 0;
                    foreach (var item in pls)
                    {
                        num2 += item.SegmentCount;
                    }
                    reports.Add(string.Format("Naked Edges: {0}", num2));
                }
                else//优质网格
                {
                    report = "This is a good mesh.";
                    Message = report;
                    return;
                }
            }

            if (NMEI.Count > 0)//非流形边报错
            {
                reports.Add(string.Format("Non-Manifold Edges: {0}", NMEI.Count));
            }
            if (NMVI.Count > 0)//非流形顶点报错
            {
                reports.Add(string.Format("Non-Manifold Vertices: {0}", NMVI.Count));
            }

            report = string.Join("\n", reports.ToArray());
            Message = report;

            DA.SetDataList("Non-Manifold Edges", NME);
            DA.SetDataList("Non-Manifold Vertices Indices", NMVI);
        }

        protected override Bitmap Icon => null;

        public static List<double> CheckPlanarFaces(Mesh mesh)
        {
            List<double> values = new List<double>();

            for (int i = 0; i < mesh.Faces.Count; i++)
            {
                Point3d a = mesh.Vertices[mesh.Faces[i].A];
                Point3d b = mesh.Vertices[mesh.Faces[i].B];
                Point3d c = mesh.Vertices[mesh.Faces[i].C];
                Point3d d = mesh.Vertices[mesh.Faces[i].D];

                Line ac = new Line(a, c);
                Line bd = new Line(b, d);

                Rhino.Geometry.Intersect.Intersection.LineLine(ac, bd, out double t1, out double t2);
                double distance = ac.PointAt(t1).DistanceTo(bd.PointAt(t2));

                values.Add(distance);
            }

            return values;
        }
        public static bool isNonManifold(Mesh mesh)
        {
            bool isOriented = false;
            bool hasBoundary = false;
            bool isNonManifold = mesh.IsManifold(true, out isOriented, out hasBoundary);
            return !isNonManifold;
        }
        public static bool hasBoundary(Mesh mesh)
        {
            PlanktonMesh PM = mesh.ToPlanktonMesh();
            return !PM.IsClosed();
        }

        public static List<int> NonManifoldEdges(Mesh mesh, ref List<Line> NME)
        {
            var MTE = mesh.TopologyEdges;
            var MTV = mesh.TopologyVertices;
            List<int> NMEI = new List<int>();
            for (int te = 0; te < MTE.Count; te++)
            {
                int[] connFaces = MTE.GetConnectedFaces(te);
                if (connFaces.Length > 2)
                {
                    NMEI.Add(te);
                    NME.Add(mesh.TopologyEdges.EdgeLine(te));
                }
            }
            return NMEI;
        }
        //检查非流形点
        public static List<int> NonManifoldVertices(Mesh mesh)
        {
            List<int> NMVI = new List<int>();
            var MTE = mesh.TopologyEdges;
            var MTV = mesh.TopologyVertices;
            bool[] sampledV = new bool[MTV.Count];

            int k = 0;
            Parallel.For(k, MTV.Count, tv =>
            {
                List<int> fList = MTV.ConnectedFaces(tv).ToList();
                if (fList.Count == 1) return;
                bool[] sampled = new bool[fList.Count];
                int clusters = 0;

                for (int i = 0; i < fList.Count; i++)
                {
                    if (!sampled[i])
                    {
                        List<int> clusterIndices = new List<int>();
                        clusterIndices.Add(i);
                        sampled[i] = true;
                        do
                        {
                            for (int j = 0; j < fList.Count; j++)
                            {
                                if (j != clusterIndices[0] && !sampled[j])
                                {
                                    List<int> faceIEI = MTE.GetEdgesForFace(fList[clusterIndices[0]]).ToList();
                                    List<int> faceJEI = MTE.GetEdgesForFace(fList[j]).ToList();
                                    List<int> common = faceIEI.Intersect(faceJEI).ToList();
                                    if (common.Any())
                                    {
                                        clusterIndices.Add(j);
                                        sampled[j] = true;
                                    }
                                }
                            }
                            clusterIndices.RemoveAt(0);
                        } while (clusterIndices.Count > 0);
                        clusters++;
                    }
                }

                if (clusters > 1)
                {
                    NMVI.Add(tv);
                }
            });

            return NMVI;
        }
    }
}
