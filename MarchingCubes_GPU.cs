using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Rhino.Geometry;
using Alea;
using Alea.CSharp;
using Alea.Parallel;
using Alea.cuBLAS;
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
        public int numVoxels;
        public float3 voxelSize;
        public float isoValue;

        public  int3 gridSize;

        // sample points
        private float3[] samplePts;
        public Point3d[] samplePoints;

        // the index of voxel in the grid
        public int3[] gridIdx;
        // the index of active voxel
        public int[] index_voxelActive;
        // the vertex positions of all voxels 
        public float3[] result_voxelV;

        // the number of all vertices
        public int sumVerts;
        // the number of active voxels
        public int num_voxelActive;

        // voxel scale
        public float scale;

        public float[] cubeValues;
        public int[] voxelVerts;
        public int[] verts_voxelActive;
        public int[] verts_scanIdx;
        public int[] voxelOccupied;

        // result
        public float3[] resultVerts;

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
        public float ComputeValue(float3[] samplePts, float3 testP, int sampleLength)
        {
            float result = 0.0f;
            float Dx, Dy, Dz;

            for (int j = 0; j < sampleLength; j++)
            {
                Dx = testP.x - samplePts[j].x;
                Dy = testP.y - samplePts[j].y;
                Dz = testP.z - samplePts[j].z;

                result += 1 / (Dx * Dx + Dy * Dy + Dz * Dz);
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
        public void runClassifyVoxel()
        {
            var gpu = Gpu.Default;

            samplePts = ConvertPointsToFloat3(samplePoints);

            // allocate memorys
            var d_voxelVerts = gpu.Allocate<int>(numVoxels);
            float3[] d_samplePts = gpu.Allocate<float3>(samplePts);

            //Copy const values
            float3 baseP = new float3((float)basePoint.X, (float)basePoint.Y, (float)basePoint.Z);
            gpu.Copy(isoValue, constIsovalue);
            gpu.Copy(baseP, constBasePoint);
            gpu.Copy(voxelSize, constVoxelSize);
            gpu.Copy(gridSize, constGridSize);
            gpu.Copy(Tables.VertsTable, verticesTable);

            gpu.For(0, numVoxels, i =>
            {
                //计算grid中的位置
                int3 gridPos = calcGridPos(i, constGridSize.Value);
                float3 p = new float3();

                p.x = constBasePoint.Value.x + gridPos.x * constVoxelSize.Value.x;
                p.y = constBasePoint.Value.y + gridPos.y * constVoxelSize.Value.y;
                p.z = constBasePoint.Value.z + gridPos.z * constVoxelSize.Value.z;

                //输出所有顶点
                float3 a0 = p;
                float3 a1 = CreateFloat3(constVoxelSize.Value.x + p.x, 0 + p.y, 0 + p.z);
                float3 a2 = CreateFloat3(constVoxelSize.Value.x + p.x, constVoxelSize.Value.y + p.y, 0 + p.z);
                float3 a3 = CreateFloat3(0 + p.x, constVoxelSize.Value.y + p.y, 0 + p.z);
                float3 a4 = CreateFloat3(0 + p.x, 0 + p.y, constVoxelSize.Value.z + p.z);
                float3 a5 = CreateFloat3(constVoxelSize.Value.x + p.x, 0 + p.y, constVoxelSize.Value.z + p.z);
                float3 a6 = CreateFloat3(constVoxelSize.Value.x + p.x, constVoxelSize.Value.y + p.y, constVoxelSize.Value.z + p.z);
                float3 a7 = CreateFloat3(0 + p.x, constVoxelSize.Value.y + p.y, constVoxelSize.Value.z + p.z);

                //计算cube中的8个点对应的value
                float d0 = ComputeValue(d_samplePts, a0, d_samplePts.Length);
                float d1 = ComputeValue(d_samplePts, a1, d_samplePts.Length);
                float d2 = ComputeValue(d_samplePts, a2, d_samplePts.Length);
                float d3 = ComputeValue(d_samplePts, a3, d_samplePts.Length);
                float d4 = ComputeValue(d_samplePts, a4, d_samplePts.Length);
                float d5 = ComputeValue(d_samplePts, a5, d_samplePts.Length);
                float d6 = ComputeValue(d_samplePts, a6, d_samplePts.Length);
                float d7 = ComputeValue(d_samplePts, a7, d_samplePts.Length);

                //判定它们的状态
                int cubeindex;
                cubeindex = Compact(d0, constIsovalue.Value);
                cubeindex += Compact(d1, constIsovalue.Value) * 2;
                cubeindex += Compact(d2, constIsovalue.Value) * 4;
                cubeindex += Compact(d3, constIsovalue.Value) * 8;
                cubeindex += Compact(d4, constIsovalue.Value) * 16;
                cubeindex += Compact(d5, constIsovalue.Value) * 32;
                cubeindex += Compact(d6, constIsovalue.Value) * 64;
                cubeindex += Compact(d7, constIsovalue.Value) * 128;

                //根据表来查出该体素的顶点数
                 int vertexNum= verticesTable[cubeindex];

                d_voxelVerts[i] = vertexNum;
            });
            voxelVerts = Gpu.CopyToHost(d_voxelVerts);

            Gpu.Free(d_samplePts);
        }
        // reduce empty voxel and extract active voxels
        public void runExtractActiveVoxels()
        {
            var gpu = Gpu.Default;
            // compute the number of active voxels
            List<int> index_voxelActiveList = new List<int>();

            for (int i = 0; i < voxelVerts.Length; i++)
            {
                if (voxelVerts[i] > 0)
                {
                    index_voxelActiveList.Add(i);
                }
            }

            // the index of active voxel
            index_voxelActive = index_voxelActiveList.ToArray();
            num_voxelActive = index_voxelActive.Length;
            // the number of vertices in each active voxel
            verts_voxelActive = new int[num_voxelActive];
            // the number of all vertices
            sumVerts = 0;

            Parallel.For(0, num_voxelActive, i =>
             {
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
            float3[] d_resultV = Gpu.Default.Allocate<float3>(sumVerts);
            float[] d_cubeValues = Gpu.Default.Allocate<float>(8 * num_voxelActive);

            // input arguments
            float3[] d_samplePts = Gpu.Default.Allocate<float3>(samplePts);
            int[] d_verts_scanIdx = Gpu.Default.Allocate<int>(verts_scanIdx);
            int[] d_index_voxelActive = Gpu.Default.Allocate<int>(index_voxelActive);

            // const values
            gpu.Copy(Vertices, constVertices);
            gpu.Copy(EdgeDirection, constEdgeDirection);
            gpu.Copy(EdgeConnection, constEdgeConnection);
            gpu.Copy(Tables.EdgeTable, edgeTable);
            gpu.Copy(Tables.TriangleTable_GPU, triangleTable);

            float3 baseP = new float3((float)basePoint.X, (float)basePoint.Y, (float)basePoint.Z);
            gpu.Copy(baseP, constBasePoint);
            gpu.Copy(isoValue, constIsovalue);
            gpu.Copy(scale, constScale);


            gpu.For(0, num_voxelActive, i =>
            {
                //计算grid中的位置
                int3 gridPos = calcGridPos(d_index_voxelActive[i], constGridSize.Value);
                float3 p = new float3();

                p.x = constBasePoint.Value.x + gridPos.x * constVoxelSize.Value.x;
                p.y = constBasePoint.Value.y + gridPos.y * constVoxelSize.Value.y;
                p.z = constBasePoint.Value.z + gridPos.z * constVoxelSize.Value.z;

                //输出所有顶点
                float3 a0 = p;
                float3 a1 = CreateFloat3(constVoxelSize.Value.x + p.x, 0 + p.y, 0 + p.z);
                float3 a2 = CreateFloat3(constVoxelSize.Value.x + p.x, constVoxelSize.Value.y + p.y, 0 + p.z);
                float3 a3 = CreateFloat3(0 + p.x, constVoxelSize.Value.y + p.y, 0 + p.z);
                float3 a4 = CreateFloat3(0 + p.x, 0 + p.y, constVoxelSize.Value.z + p.z);
                float3 a5 = CreateFloat3(constVoxelSize.Value.x + p.x, 0 + p.y, constVoxelSize.Value.z + p.z);
                float3 a6 = CreateFloat3(constVoxelSize.Value.x + p.x, constVoxelSize.Value.y + p.y, constVoxelSize.Value.z + p.z);
                float3 a7 = CreateFloat3(0 + p.x, constVoxelSize.Value.y + p.y, constVoxelSize.Value.z + p.z);

                //Compute cubeValues of 8 vertices
                d_cubeValues[i * 8] = ComputeValue(d_samplePts, a0, d_samplePts.Length);
                d_cubeValues[i * 8 + 1] = ComputeValue(d_samplePts, a1, d_samplePts.Length);
                d_cubeValues[i * 8 + 2] = ComputeValue(d_samplePts, a2, d_samplePts.Length);
                d_cubeValues[i * 8 + 3] = ComputeValue(d_samplePts, a3, d_samplePts.Length);
                d_cubeValues[i * 8 + 4] = ComputeValue(d_samplePts, a4, d_samplePts.Length);
                d_cubeValues[i * 8 + 5] = ComputeValue(d_samplePts, a5, d_samplePts.Length);
                d_cubeValues[i * 8 + 6] = ComputeValue(d_samplePts, a6, d_samplePts.Length);
                d_cubeValues[i * 8 + 7] = ComputeValue(d_samplePts, a7, d_samplePts.Length);

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
                        pt.x = constBasePoint.Value.x + (gridPos.x + constVertices[constEdgeConnection[j * 2 + 0] * 3 + 0] + Offset * constEdgeDirection[j * 3 + 0]) * constScale.Value;
                        pt.y = constBasePoint.Value.y + (gridPos.y + constVertices[constEdgeConnection[j * 2 + 0] * 3 + 1] + Offset * constEdgeDirection[j * 3 + 1]) * constScale.Value;
                        pt.z = constBasePoint.Value.z + (gridPos.z + constVertices[constEdgeConnection[j * 2 + 0] * 3 + 2] + Offset * constEdgeDirection[j * 3 + 2]) * constScale.Value;
                        d_pts[12 * i + j] = pt;
                    }
                }
                int num = 0;
                //Find out points from each triangle
                for (int Triangle = 0; Triangle < 5; Triangle++)
                {
                    if (triangleTable[flag * 16 + 3 * Triangle] < 0)
                        break;

                    for (int Corner = 0; Corner < 3; Corner++)
                    {
                        int Vertex = triangleTable[flag * 16 + 3 * Triangle + Corner];
                        float3 pd = CreateFloat3(d_pts[12 * i + Vertex].x, d_pts[12 * i + Vertex].y, d_pts[12 * i + Vertex].z);
                        d_resultV[d_verts_scanIdx[i] + num] = pd;
                        num++;

                    }
                }
            });
            resultVerts = Gpu.CopyToHost(d_resultV);
            return ConvertFloat3ToPoint3d(resultVerts);
        }

        #region pointer based MCGPU
        public float ComputeValue(deviceptr<float3> samplePts, float3 testP, int sampleLength)
        {
            float result = 0.0f;
            float Dx, Dy, Dz;

            for (int j = 0; j < sampleLength; j++)
            {
                Dx = testP.x - samplePts[j].x;
                Dy = testP.y - samplePts[j].y;
                Dz = testP.z - samplePts[j].z;

                result += 1 / (Dx * Dx + Dy * Dy + Dz * Dz);
            }
            return result;
        }
        public void ClassifyVoxel(deviceptr<int3> d_gridIdx, deviceptr<float3> d_voxelV, deviceptr<int> d_voxelVerts,
    deviceptr<int> d_voxelOccupied, deviceptr<float3> d_samplePts, int sampleLength)
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
            float d0 = ComputeValue(d_samplePts, d_voxelV[i * 8], sampleLength);
            float d1 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 1], sampleLength);
            float d2 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 2], sampleLength);
            float d3 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 3], sampleLength);
            float d4 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 4], sampleLength);
            float d5 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 5], sampleLength);
            float d6 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 6], sampleLength);
            float d7 = ComputeValue(d_samplePts, d_voxelV[i * 8 + 7], sampleLength);

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
        public void runClassifyVoxelPtr()
        {
            var gpu = Gpu.Default;

            int threads = 512;
            Alea.dim3 gridDim = new Alea.dim3((numVoxels + threads - 1) / threads, 1, 1); //block的数量，维度
            Alea.dim3 blockDim = new Alea.dim3(threads, 1, 1); //thread的数量，维度

            if (gridDim.x > 65535)
            {
                gridDim.y = gridDim.x / 32768;
                gridDim.x = 32768;
            }
            LaunchParam lp = new LaunchParam(gridDim, blockDim);

            // copy arguments
            // copy const values
            float3 baseP = new float3((float)basePoint.X, (float)basePoint.Y, (float)basePoint.Z);
            gpu.Copy(isoValue, constIsovalue);
            gpu.Copy(baseP, constBasePoint);
            gpu.Copy(voxelSize, constVoxelSize);
            gpu.Copy(gridSize, constGridSize);
            gpu.Copy(Tables.VertsTable, verticesTable);

            // input arguments
            samplePts = ConvertPointsToFloat3(samplePoints);

            // output arguments
            using (var d_voxelVerts = gpu.AllocateDevice<int>(numVoxels))
            using (var d_voxelOccupied = gpu.AllocateDevice<int>(numVoxels))
            using (var d_voxelV = gpu.AllocateDevice<float3>(numVoxels * 8))
            using (var d_gridIdx = gpu.AllocateDevice<int3>(numVoxels))

            using (var d_samplePts = gpu.AllocateDevice<float3>(samplePts))
            {
                // launch kernel
                gpu.Launch(ClassifyVoxel, lp, d_gridIdx.Ptr, d_voxelV.Ptr, d_voxelVerts.Ptr, d_voxelOccupied.Ptr, d_samplePts.Ptr, d_samplePts.Length);

                result_voxelV = Gpu.CopyToHost(d_voxelV);
                voxelOccupied = Gpu.CopyToHost(d_voxelOccupied);
                voxelVerts = Gpu.CopyToHost(d_voxelVerts);
                gridIdx = Gpu.CopyToHost(d_gridIdx);
            }
        }
        public void ExtractIsoSurface(deviceptr<float> d_cubeValues, deviceptr<float3> d_pts, deviceptr<float3> Apts,
    deviceptr<int3> d_index3d_voxelActive, deviceptr<float3> d_model_voxelActive, deviceptr<float3> d_samplePts, deviceptr<int> d_verts_scanIdx, int sampleLength)
        {
            int blockId = blockIdx.y * gridDim.x + blockIdx.x;
            int i = blockId * blockDim.x + threadIdx.x;

            //Compute cubeValues of 8 vertices
            d_cubeValues[i * 8] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8], sampleLength);
            d_cubeValues[i * 8 + 1] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 1], sampleLength);
            d_cubeValues[i * 8 + 2] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 2], sampleLength);
            d_cubeValues[i * 8 + 3] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 3], sampleLength);
            d_cubeValues[i * 8 + 4] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 4], sampleLength);
            d_cubeValues[i * 8 + 5] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 5], sampleLength);
            d_cubeValues[i * 8 + 6] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 6], sampleLength);
            d_cubeValues[i * 8 + 7] = ComputeValue(d_samplePts, d_model_voxelActive[i * 8 + 7], sampleLength);

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
                    d_pts[12 * i + j] = pt;
                }
            }
            int num = 0;
            //Find out points from each triangle
            for (int Triangle = 0; Triangle < 5; Triangle++)
            {
                if (triangleTable[flag * 16 + 3 * Triangle] < 0)
                    break;

                for (int Corner = 0; Corner < 3; Corner++)
                {
                    int Vertex = triangleTable[flag * 16 + 3 * Triangle + Corner];
                    float3 pd = CreateFloat3(d_pts[12 * i + Vertex].x, d_pts[12 * i + Vertex].y, d_pts[12 * i + Vertex].z);
                    Apts[d_verts_scanIdx[i] + num] = pd;
                    num++;

                }
            }
        }
        #endregion
    }
}
