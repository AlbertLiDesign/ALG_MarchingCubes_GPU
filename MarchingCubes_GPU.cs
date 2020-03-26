using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Rhino.Geometry;
using Alea;
using Alea.CSharp;
using Alea.Parallel;
using float3 = Alea.float3;
using int3 = Alea.int3;
using float4 = Alea.float4;

namespace ALG_MarchingCubes
{
    public class MarchingCubes_GPU
    {
        private static readonly GlobalArraySymbol<int> verticesTable = Gpu.DefineConstantArraySymbol<int>(256);
        private static readonly GlobalArraySymbol<int> edgeTable = Gpu.DefineConstantArraySymbol<int>(Tables.EdgeTable.Length);
        private static readonly GlobalArraySymbol<int> triangleTable = Gpu.DefineConstantArraySymbol<int>(256*16);

        private static readonly GlobalVariableSymbol<float> constIsovalue = Gpu.DefineConstantVariableSymbol<float>();
        private static readonly GlobalVariableSymbol<float> constScale = Gpu.DefineConstantVariableSymbol<float>();
        private static readonly GlobalVariableSymbol<float3> constBasePoint = Gpu.DefineConstantVariableSymbol<float3>();
        private static readonly GlobalVariableSymbol<float3> constVoxelSize = Gpu.DefineConstantVariableSymbol<float3>();
        private static readonly GlobalVariableSymbol<int3> constGridSize = Gpu.DefineConstantVariableSymbol<int3>();

        private static readonly GlobalArraySymbol<float> constVertices = Gpu.DefineConstantArraySymbol<float>(24);
        private static readonly GlobalArraySymbol<float> constEdgeDirection = Gpu.DefineConstantArraySymbol<float>(36);
        private static readonly GlobalArraySymbol<int> constEdgeConnection = Gpu.DefineConstantArraySymbol<int>(24);

        // boudingbox of input parameters
        public Box sourceBox;
        public Point3d basePoint;

        // constants
        public int numVoxels = 0;
        public float3 voxelSize;
        public float isoValue;

        public  int3 gridSize;
        public int[,] gridIndex3d;

        // sample points
        private float3[] samplePts;
        public Point3d[] samplePoints;

        // the index of voxel in the grid
        public int3[] gridIdx;
        // the index of active voxel in the grid
        public int3[] index3d_voxelActive;
        // the vertex positions of all voxels 
        public float3[] result_voxelV;

        // the number of all vertices
        public int sumVerts;
        // the number of active voxels
        public int num_voxelActive;
        // the positions of all active voxel vertices
        public float3[] model_voxelActive;

        // voxel scale
        public float scale;

        public float[] cubeValues;
        public int[] voxelVerts;
        public int[] verts_voxelActive;
        public int[] verts_scanIdx;
        public int[] voxelOccupied;

        public MarchingCubes_GPU() { }
        public MarchingCubes_GPU(Point3d basePoint, Box sourceBox, int3 gridSize, float3 voxelSize,
            float scale, float isoValue,Point3d[] samplePoints)
        {
            this.basePoint = basePoint;
            this.sourceBox = sourceBox;
            this.gridSize = gridSize;
            this.voxelSize = voxelSize;
            this.numVoxels = this.gridSize.x * this.gridSize.y * this.gridSize.z;
            this.scale = scale;
            this.isoValue = isoValue;
            this.samplePoints = samplePoints;
        }

        private static float[] Vertices = new float[24]
         {
             0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f
         };
        private int[] EdgeConnection = new int[24]
        {
             0,1, 1,2, 2,3, 3,0,
             4,5, 5,6, 6,7, 7,4,
             0,4, 1,5, 2,6, 3,7
         };
        private float[] EdgeDirection = new float[36]
         {
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f
         };

        #region basic functions
        private float3[] ConvertPointsToFloat3(Point3d[] pts)
        {
            float3[] d = new float3[pts.Length];
            for (int i = 0; i < pts.Length; i++)
            {
                d[i].x = (float)pts[i].X;
                d[i].y = (float)pts[i].Y;
                d[i].z = (float)pts[i].Z;
            }
            return d;
        }
        public int[,] ConvertInt3ToIntArray(int3[] a)
        {
            int[,] b = new int[a.Length, 3];
            for (int i = 0; i < a.Length; i++)
            {
                b[i, 0] = a[i].x;
                b[i, 1] = a[i].y;
                b[i, 2] = a[i].z;
            }
            return b;
        }
        public List<Point3d> ConvertFloat3ToPoint3d(float3[] array)
        {
            List<Point3d> pts = new List<Point3d>();
            for (int i = 0; i < array.Length; i++)
            {
                pts.Add(new Point3d(array[i].x, array[i].y, array[i].z));
            }
            return pts;
        }
        public List<Point3d> ConvertFloat4ToPoint3d(float4[] array)
        {
            List<Point3d> pts = new List<Point3d>();
            for (int i = 0; i < array.Length; i++)
            {
                pts.Add(new Point3d(array[i].x, array[i].y, array[i].z));
            }
            return pts;
        }
        public int Compact(float a, float b) { if (a < b) { return 1; } else { return 0; } }

        //compute distance
        //v = ((3x)^4 - 5(3x)^2 - 5(3y)^2 + (3z)^4 - 5(z)^2 + 11.8) * 0.2 + 0.5
        private float tangle(float3[] samplePts, float x, float y, float z)
        {
            float result = 0.0f;
            float Dx, Dy, Dz;

            for (int j = 0; j < samplePts.Length; j++)
            {
                Dx = x - samplePts[j].x;
                Dy = y - samplePts[j].y;
                Dz = z - samplePts[j].z;

                result += 1 / (Dx * Dx + Dy * Dy + Dz * Dz);
            }
            return result;
        }

        //compute 3d index of each voxel on the grid according to 1d index
        private int3 calcGridPos(int i, int3 gridSize)
        {
            int3 gridPos;

            gridPos.z = i / (gridSize.x * gridSize.y);
            gridPos.y = i % (gridSize.x * gridSize.y) / gridSize.x;
            gridPos.x = i % (gridSize.x * gridSize.y) % gridSize.x;

            return gridPos;
        }
        public float3 CreateFloat3(float x, float y, float z)
        {
            float3 p = new float3();
            p.x = x;
            p.y = y;
            p.z = z;
            return p;
        }

        public float ComputeValue(float3[] samplePts, float3 testP)
        {
            float result = 0.0f;
            float Dx, Dy, Dz;

            for (int j = 0; j < samplePts.Length; j++)
            {
                Dx = testP.x - samplePts[j].x;
                Dy = testP.y - samplePts[j].y;
                Dz = testP.z - samplePts[j].z;

                result +=1 / (Dx * Dx + Dy * Dy + Dz * Dz);
            }
            return result;
        }
        public float GetOffset(float Value1, float Value2, float ValueDesired)
        {
            if ((Value2 - Value1) == 0.0f)
                return 0.5f;

            return (ValueDesired - Value1) / (Value2 - Value1);
        }
        #endregion
        // classify all voxels according to their activity
        public void ClassifyVoxel(int3[] d_gridIdx, float3[] d_voxelV, int[] d_voxelVerts, int[] d_voxelOccupied, float3[] d_samplePts)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x; //block在grid中的位置
            int i = blockId * blockDim.x + threadIdx.x; //线程索引

            // compute 3d index in the grid
            int3 gridPos = calcGridPos(i, constGridSize.Value);
            d_gridIdx[i] = gridPos;
            float3 p = new float3();

            p.x = constBasePoint.Value.x + gridPos.x * constVoxelSize.Value.x;
            p.y = constBasePoint.Value.y + gridPos.y * constVoxelSize.Value.y;
            p.z = constBasePoint.Value.z + gridPos.z * constVoxelSize.Value.z;

            // compute all vertices
            d_voxelV[i * 8] = p;
            d_voxelV[i * 8 + 1] = CreateFloat3(constVoxelSize.Value.x + p.x, 0 + p.y, 0 + p.z);
            d_voxelV[i * 8 + 2] = CreateFloat3(constVoxelSize.Value.x + p.x, constVoxelSize.Value.y + p.y, 0 + p.z);
            d_voxelV[i * 8 + 3] = CreateFloat3(0 + p.x, constVoxelSize.Value.y + p.y, 0 + p.z);
            d_voxelV[i * 8 + 4] = CreateFloat3(0 + p.x, 0 + p.y, constVoxelSize.Value.z + p.z);
            d_voxelV[i * 8 + 5] = CreateFloat3(constVoxelSize.Value.x + p.x, 0 + p.y, constVoxelSize.Value.z + p.z);
            d_voxelV[i * 8 + 6] = CreateFloat3(constVoxelSize.Value.x + p.x, constVoxelSize.Value.y + p.y, constVoxelSize.Value.z + p.z);
            d_voxelV[i * 8 + 7] = CreateFloat3(0 + p.x, constVoxelSize.Value.y + p.y, constVoxelSize.Value.z + p.z);

            // compute cube value of each vertex
            float d0 = ComputeValue(d_samplePts, d_voxelV[i * 8]);
            float d1 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 1]);
            float d2 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 2]);
            float d3 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 3]);
            float d4 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 4]);
            float d5 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 5]);
            float d6 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 6]);
            float d7 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 7]);

            // check their status
            int cubeindex;
            cubeindex = Compact(d0, constIsovalue.Value);
            cubeindex += Compact(d1, constIsovalue.Value) * 2;
            cubeindex += Compact(d2, constIsovalue.Value) * 4;
            cubeindex += Compact(d3, constIsovalue.Value) * 8;
            cubeindex += Compact(d4, constIsovalue.Value) * 16;
            cubeindex += Compact(d5, constIsovalue.Value) * 32;
            cubeindex += Compact(d6, constIsovalue.Value) * 64;
            cubeindex += Compact(d7, constIsovalue.Value) * 128;

            //find out the number of vertices in each voxel
            int numVerts = verticesTable[cubeindex];

            d_voxelVerts[i] = numVerts;
            if (numVerts > 0)
            {
                d_voxelOccupied[i] = 1;
            }
        }
        public void runClassifyVoxel()
        { 
            var gpu = Gpu.Default;

            int threads = 256;
            Alea.dim3 gridDim = new Alea.dim3((numVoxels + threads - 1) / threads, 1, 1); //block的数量，维度
            Alea.dim3 blockDim = new Alea.dim3(threads, 1, 1); //thread的数量，维度

            if (gridDim.x > 65535)
            {
                gridDim.y = gridDim.x / 32768;
                gridDim.x = 32768;
            }
            LaunchParam lp = new LaunchParam(gridDim, blockDim);

            // copy arguments
            // output arguments
            int[] d_voxelVerts = gpu.Allocate<int>(numVoxels);
            int[] d_voxelOccupied = gpu.Allocate<int>(numVoxels);
            float3[] d_voxelV = gpu.Allocate<float3>(numVoxels*8);
            int3[] d_gridIdx = gpu.Allocate<int3>(numVoxels);

            // input arguments
            samplePts = ConvertPointsToFloat3(samplePoints);
            float3[] d_samplePts = gpu.Allocate<float3>(samplePts);
            
            // copy const values
            float3 baseP = new float3((float)basePoint.X, (float)basePoint.Y, (float)basePoint.Z);
            gpu.Copy(isoValue, constIsovalue);
            gpu.Copy(baseP, constBasePoint);
            gpu.Copy(voxelSize, constVoxelSize);
            gpu.Copy(gridSize, constGridSize);
            gpu.Copy(Tables.VertsTable, verticesTable);

            // launch kernel
            gpu.Launch(ClassifyVoxel, lp, d_gridIdx, d_voxelV, d_voxelVerts, d_voxelOccupied, d_samplePts);

            result_voxelV = Gpu.CopyToHost(d_voxelV);
            voxelOccupied = Gpu.CopyToHost(d_voxelOccupied);
            voxelVerts = Gpu.CopyToHost(d_voxelVerts);
            gridIdx = Gpu.CopyToHost(d_gridIdx);

            // free memory
            Gpu.Free(d_voxelVerts);
            Gpu.Free(d_voxelOccupied);
            Gpu.Free(d_voxelV);
            Gpu.Free(d_gridIdx);
            Gpu.Free(d_samplePts);
        }
        // reduce empty voxel and extract active voxels
        public void runExtractActiveVoxels()
        {
            var gpu = Gpu.Default;
            // compute the number of active voxels
            List<int> index_voxelActiveList = new List<int>();
            List<int3> index3d_voxelActiveList = new List<int3>();

            for (int i = 0; i < voxelOccupied.Length; i++)
            {
                if (voxelOccupied[i] > 0)
                {
                    index_voxelActiveList.Add(i);
                    index3d_voxelActiveList.Add(gridIdx[i]);
                }
            }

            // the index of active voxel
            int[] index_voxelActive = index_voxelActiveList.ToArray();
            num_voxelActive = index_voxelActive.Length;
            // the 3d index of active voxel in the grid
            index3d_voxelActive = index3d_voxelActiveList.ToArray();
            gridIndex3d = ConvertInt3ToIntArray(index3d_voxelActive);
            // all vertices of each voxel
            model_voxelActive = new float3[8 * num_voxelActive];
            // the number of vertices in each active voxel
            verts_voxelActive = new int[num_voxelActive];
            // the number of all vertices
            sumVerts = 0;

            Parallel.For(0, num_voxelActive, i =>
             {
                 model_voxelActive[8 * i] = result_voxelV[8 * index_voxelActive[i]];
                 model_voxelActive[8 * i + 1] = result_voxelV[8 * index_voxelActive[i] + 1];
                 model_voxelActive[8 * i + 2] = result_voxelV[8 * index_voxelActive[i] + 2];
                 model_voxelActive[8 * i + 3] = result_voxelV[8 * index_voxelActive[i] + 3];
                 model_voxelActive[8 * i + 4] = result_voxelV[8 * index_voxelActive[i] + 4];
                 model_voxelActive[8 * i + 5] = result_voxelV[8 * index_voxelActive[i] + 5];
                 model_voxelActive[8 * i + 6] = result_voxelV[8 * index_voxelActive[i] + 6];
                 model_voxelActive[8 * i + 7] = result_voxelV[8 * index_voxelActive[i] + 7];
                 verts_voxelActive[i] = voxelVerts[index_voxelActive[i]];
             });

            // execute exclusive scan for finding out the indices of result vertices
            var op = new Func<int, int, int>((a, b) => { return a + b; });
            Alea.Session session = new Alea.Session(gpu);
            int[] d_verts_voxelActive = Gpu.Default.Allocate<int>(verts_voxelActive);
            int[] d_voxelVertsScan = Gpu.Default.Allocate<int>(num_voxelActive);

            GpuExtension.Scan<int>(session, d_voxelVertsScan, d_verts_voxelActive, 0, op, 0);

            var result_Scan = Gpu.CopyToHost(d_voxelVertsScan);

            verts_scanIdx = new int[num_voxelActive];

            for (int i = 1; i < num_voxelActive; i++)
            {
                verts_scanIdx[i] = result_Scan[i - 1];
            }

            try
            {
                verts_scanIdx[0] = 0;
            }
            catch (Exception)
            {
                throw new Exception("No eligible isosurface can be extracted, please change isovalue.");
            }
                
            sumVerts = verts_scanIdx.ElementAt(verts_scanIdx.Length-1) + verts_voxelActive.ElementAt(verts_voxelActive.Length-1);

            Gpu.Free(d_verts_voxelActive);
            Gpu.Free(d_voxelVertsScan);
        }
        // extract isosurface points using GPU
        public List<Point3d> runExtractIsoSurfaceGPU()
        {
            var gpu = Gpu.Default;

            // output arguments
            float3[] pts = new float3[12 * num_voxelActive];
            float3[] d_pts = Gpu.Default.Allocate<float3>(pts);
            float3[] Apts = Gpu.Default.Allocate<float3>(sumVerts);
            float[] d_cubeValues = Gpu.Default.Allocate<float>(8 * num_voxelActive);

            // input arguments
            int3[] d_index3d_voxelActive = Gpu.Default.Allocate<int3>(index3d_voxelActive);
            float3[] d_model_voxelActive = Gpu.Default.Allocate<float3>(model_voxelActive);
            float3[] d_samplePts = Gpu.Default.Allocate<float3>(samplePts);
            int[] d_verts_scanIdx = Gpu.Default.Allocate<int>(verts_scanIdx);

            // const values
            gpu.Copy(Vertices, constVertices);
            gpu.Copy(EdgeDirection, constEdgeDirection);
            gpu.Copy(EdgeConnection, constEdgeConnection);
            gpu.Copy(Tables.EdgeTable, edgeTable);
            gpu.Copy(Tables.TriangleTable_GPU,triangleTable);

            float3 baseP = new float3((float)basePoint.X, (float)basePoint.Y, (float)basePoint.Z);
            gpu.Copy(baseP, constBasePoint);
            gpu.Copy(isoValue, constIsovalue);
            gpu.Copy(scale, constScale);
            

            gpu.For(0, num_voxelActive, i =>
             {
                 //Compute cubeValues of 8 vertices
                 d_cubeValues[i * 8] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8]);
                 d_cubeValues[i * 8 + 1] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 1]);
                 d_cubeValues[i * 8 + 2] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 2]);
                 d_cubeValues[i * 8 + 3] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 3]);
                 d_cubeValues[i * 8 + 4] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 4]);
                 d_cubeValues[i * 8 + 5] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 5]);
                 d_cubeValues[i * 8 + 6] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 6]);
                 d_cubeValues[i * 8 + 7] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 7]);

                 //Check each vertex state
                 int flag = Compact(d_cubeValues[i * 8], constIsovalue.Value);
                 flag += Compact(d_cubeValues[i * 8 + 1], constIsovalue.Value) * 2;
                 flag += Compact(d_cubeValues[i * 8 + 2], constIsovalue.Value) * 4;
                 flag += Compact(d_cubeValues[i * 8 + 3], constIsovalue.Value) * 8;
                 flag += Compact(d_cubeValues[i * 8 + 4], constIsovalue.Value) * 16;
                 flag += Compact(d_cubeValues[i * 8 + 5], constIsovalue.Value) * 32;
                 flag += Compact(d_cubeValues[i * 8 + 6], constIsovalue.Value) * 64;
                 flag += Compact(d_cubeValues[i * 8 + 7], constIsovalue.Value) * 128;

                 //find out which edge intersects the isosurface
                 int EdgeFlag = edgeTable[flag];

                 //check whether this voxel is crossed by the isosurface
                 for (int j = 0; j < 12; j++)
                 {
                     //check whether an edge have a point
                     if ((EdgeFlag & (1 << j)) != 0) 
                    {
                         //compute t values from two end points on each edge
                         float Offset = GetOffset(d_cubeValues[i * 8 + constEdgeConnection[j * 2 + 0]], d_cubeValues[i * 8 + constEdgeConnection[j * 2 + 1]], constIsovalue.Value);
                         float3 pt = new float3();
                         //get positions
                         pt.x = constBasePoint.Value.x + (d_index3d_voxelActive[i].x + constVertices[constEdgeConnection[j * 2 + 0] * 3 + 0] + Offset * constEdgeDirection[j * 3 + 0]) * constScale.Value;
                         pt.y = constBasePoint.Value.y + (d_index3d_voxelActive[i].y + constVertices[constEdgeConnection[j * 2 + 0] * 3 + 1] + Offset * constEdgeDirection[j * 3 + 1]) * constScale.Value;
                         pt.z = constBasePoint.Value.z + (d_index3d_voxelActive[i].z + constVertices[constEdgeConnection[j * 2 + 0] * 3 + 2] + Offset * constEdgeDirection[j * 3 + 2]) * constScale.Value;
                         d_pts[12*i+j] = pt;
                     }
                 }
                 int num = 0;
                 //Find out points from each triangle
                 for (int Triangle = 0; Triangle < 5; Triangle++)
                 {
                     if (triangleTable[flag*16+ 3 * Triangle] < 0)
                         break;

                     for (int Corner = 0; Corner < 3; Corner++)
                     { 
                         int Vertex = triangleTable[flag*16+ 3 * Triangle + Corner];
                         float3 pd = CreateFloat3(d_pts[12*i+Vertex].x,d_pts[12 * i + Vertex].y, d_pts[12 * i + Vertex].z);
                         Apts[d_verts_scanIdx[i] + num] = pd;
                         num++;
                         
                     }
                 }
             });
            var result_Scan = Gpu.CopyToHost(Apts);
            return ConvertFloat3ToPoint3d(result_Scan);
        }
    }
}
