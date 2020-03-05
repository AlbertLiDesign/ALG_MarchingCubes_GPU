using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Rhino.Geometry;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Grasshopper.Kernel.Types.Transforms;

namespace ALG_MarchingCubes_GPU
{
    public class BasicFunctions
    {
        //构建用于映射的Box
        public static Box CreateUnionBBoxFromGeometry(List<IGH_GeometricGoo> geos, double scale)
        {
            Plane worldXY = Plane.WorldXY;
            Transform xform = Transform.ChangeBasis(Plane.WorldXY, worldXY);
            BoundingBox empty = BoundingBox.Empty;
            int num = geos.Count - 1;
            for (int i = 0; i <= num; i++)
            {
                if (geos[i] != null)
                {
                    BoundingBox boundingBox = geos[i].GetBoundingBox(xform);
                    empty.Union(boundingBox);
                }
            }

            Transform xform2 = Transform.Scale(empty.Center, scale);
            empty.Transform(xform2);
            Box box = new Box(empty);
            return box;
        }
        //Box映射
        public static IGH_GeometricGoo BoxMapping(Box box1, Box box2, IGH_GeometricGoo geo)
        {

            if (!box1.IsValid || !box2.IsValid)
            {
                return null;
            }
            Plane plane1 = box1.Plane;
            Plane plane2 = box2.Plane;
            plane1.Origin = box1.Center;
            plane2.Origin = box2.Center;
            double xscale = box2.X.Length / box1.X.Length;
            double yscale = box2.Y.Length / box1.Y.Length;
            double zscale = box2.Z.Length / box1.Z.Length;
            ITransform item = new Scale(plane1, xscale, yscale, zscale);
            ITransform item2 = new Orientation(plane1, plane2);
            GH_Transform gh_Transform = new GH_Transform();
            gh_Transform.CompoundTransforms.Add(item);
            gh_Transform.CompoundTransforms.Add(item2);
            gh_Transform.ClearCaches();

            IGH_GeometricGoo geo2 = geo.DuplicateGeometry();
            geo2 = geo2.Transform(gh_Transform.Value);
            return geo2;
        }

        public static List<IGH_GeometricGoo> ConvertPointsToGeo(List<Point3d> pts)
        {
            List<IGH_GeometricGoo> geos = new List<IGH_GeometricGoo>();
            for (int i = 0; i < pts.Count; i++)
            {
                GH_Point point = new GH_Point(pts[i]);
                geos.Add(point);
            }
            return geos;
        }
        public static List<Point3d> ConvertGeosToPoints(List<IGH_GeometricGoo> geos)
        {
            List<Point3d> pts = new List<Point3d>();
            for (int i = 0; i < geos.Count; i++)
            {
                if (geos[i].IsValid)
                {
                    Point3d point = new Point3d();
                    GH_Convert.ToPoint3d(geos[i], ref point, GH_Conversion.Both);
                    pts.Add(point);
                }
            }
            return pts;
        }
        public static Mesh ExtractMesh(List<Point3d> pts)
        {
            Mesh mesh = new Mesh();
            int FCount = pts.Count / 3;
            for (int i = 0; i < FCount; i++)
            {
                Point3d a = pts[i * 3];
                Point3d b = pts[i * 3 + 1];
                Point3d c = pts[i * 3 + 2];

                Mesh subMesh = new Mesh();
                subMesh.Vertices.Add(a);
                subMesh.Vertices.Add(b);
                subMesh.Vertices.Add(c);
                subMesh.Faces.AddFace(0, 1, 2);

                mesh.Append(subMesh);
            }
            return mesh;
        }
    }
}
