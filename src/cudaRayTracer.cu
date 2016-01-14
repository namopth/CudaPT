#include "cudahelper.h"
#include "raytracer.h"
#include "mathhelper.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <curand_kernel.h>

#define BLOCK_SIZE 16
#define BVH_DEPTH_MAX 128

texture<float4, 1, cudaReadModeElementType> g_bvhMinMaxBounds;
texture<uint1, 1, cudaReadModeElementType> g_bvhOffsetTriStartN;
texture<float4, 1, cudaReadModeElementType> g_triIntersectionData;
texture<float4, 1, cudaReadModeElementType>* g_sceneTexturesData;

RTTexture* g_devTextures = nullptr;
std::vector<float*> g_devTextureRaws; // to CUFREE on CPU Side
RTTriangle* g_devTriangles = nullptr;
RTMaterial* g_devMaterials = nullptr;
float* g_devResultData = nullptr;
float* g_devAccResultData = nullptr;
float4* g_devBVHMinMaxBounds = nullptr;
uint1* g_devBVHOffsetTriStartN = nullptr;
float4* g_devTriIntersectionData = nullptr;

bool g_bIsCudaInit = false;
size_t g_resultDataSize = 0;

struct CURay
{
	float3 orig;
	float3 dir;
	__hd__ CURay(float3 _orig, float3 _dir) : orig(_orig), dir(_dir) {}
	__hd__ float IntersectAABB(const float3& _min, const float3& _max) const
	{
		float3 modDir = dir;
		modDir.x = escapeZero(modDir.x, M_EPSILON);
		modDir.y = escapeZero(modDir.y, M_EPSILON);
		modDir.z = escapeZero(modDir.z, M_EPSILON);
		modDir = vecRcp(modDir);
		float3 tmin = vecMul((_min - orig), modDir);
		float3 tmax = vecMul((_max - orig), modDir);
		float3 real_min = vecMin(tmin, tmax);
		float3 real_max = vecMax(tmin, tmax);
		float minmax = fminf(fminf(real_max.x, real_max.y), real_max.z);
		float maxmin = fmaxf(fmaxf(real_min.x, real_min.y), real_min.z);
		if (minmax >= maxmin)
			return (maxmin > M_EPSILON) ? maxmin : 0;
		return M_INF;
	}

	template<bool cullback = true>
	__hd__ float IntersectTri(const float3& _p0, const float3& _e0, const float3& _e1, float& w, float& u, float& v, float epsilon = M_EPSILON) const
	{
		if (cullback && vecDot(vecCross(_e0, _e1), dir) > 0.f)
			return M_INF;
		float3 de2 = vecCross(dir, _e1);
		float divisor = vecDot(de2, _e0);
		if (fabs(divisor) < M_EPSILON)
			return M_INF;
		divisor = rcpf(divisor);
		float3 t = (orig + epsilon * dir) - _p0;
		float3 te1 = vecCross(t, _e0);
		float rT = vecDot(te1, _e1) * divisor;
		if (rT < 0.f)
			return M_INF;
		u = vecDot(de2, t) * divisor;
		if (u < 0.f || u > 1.f)
			return M_INF;
		v = vecDot(te1, dir) * divisor;
		if (v < 0.f || (u + v) > 1.f)
			return M_INF;
		w = 1 - u - v;
		return rT;
	}
};

struct TracePrimitiveResult
{
	float dist;
	int32 triId;
	float w;
	float u;
	float v;
};

template<bool cullback = true>
__device__ bool TracePrimitive(const CURay &ray, TracePrimitiveResult& result, const float maxDist = M_INF, const float rayEpsilon = M_EPSILON)
{
	float minIntersect = maxDist;
	uint32 tracedTriId = 0;
	float w, u, v;
	uint32 traceCmd[BVH_DEPTH_MAX];
	traceCmd[0] = 0;
	int32 traceCmdPointer = 0;
	while (traceCmdPointer >= 0)
	{
		uint32 curInd = traceCmd[traceCmdPointer--];
		float4 boundMin = tex1Dfetch(g_bvhMinMaxBounds, curInd * 2);
		float4 boundMax = tex1Dfetch(g_bvhMinMaxBounds, curInd * 2 + 1);
		float min = ray.IntersectAABB(make_float3(boundMin.x, boundMin.y, boundMin.z),
			make_float3(boundMax.x, boundMax.y, boundMax.z));
		if (min >= 0 && min < minIntersect)
		{
			uint1 offOrTs = tex1Dfetch(g_bvhOffsetTriStartN, curInd * 2);
			uint1 tN = tex1Dfetch(g_bvhOffsetTriStartN, curInd * 2 + 1);
			if (tN.x == 0)
			{
				if (traceCmdPointer < BVH_DEPTH_MAX - 2)
				{
					traceCmd[++traceCmdPointer] = curInd + 1;
					traceCmd[++traceCmdPointer] = curInd + offOrTs.x;
				}
			}
			else
			{
				for (uint32 i = offOrTs.x; i < offOrTs.x + tN.x; i++)
				{
					float _w, _u, _v;
					float4 p0 = tex1Dfetch(g_triIntersectionData, i * 3);
					float4 e0 = tex1Dfetch(g_triIntersectionData, i * 3 + 1);
					float4 e1 = tex1Dfetch(g_triIntersectionData, i * 3 + 2);
					float triIntersect = ray.IntersectTri<cullback>(make_float3(p0.x, p0.y, p0.z),
						make_float3(e0.x, e0.y, e0.z), make_float3(e1.x, e1.y, e1.z),
						_w, _u, _v, rayEpsilon);
					if (triIntersect >= 0 && triIntersect < minIntersect)
					{
						minIntersect = triIntersect;
						tracedTriId = i;
						w = _w; u = _u; v = _v;
					}
				}
			}
		}
	}

	if (minIntersect < maxDist)
	{
		result.dist = minIntersect;
		result.triId = tracedTriId;
		result.w = w;
		result.u = u;
		result.v = v;
		return true;
	}

	return false;
}

__global__ void pt0_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
	float width, float height, RTTriangle* triangles, RTMaterial* materials, RTTexture* textures
	, float* result, float* accResult)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint ind = (y * width + x) * 3;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	float u = (2.f * ((float)x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
	float v = (2.f * ((float)y + 0.5f) / height - 1.f) * tan(fov * 0.5f);
	float3 dir = normalize(camRight * u + camUp * v + camDir);
	CURay ray(camPos, dir);

	TracePrimitiveResult traceResult;
	if (TracePrimitive(ray, traceResult))
	{
		uint32 matId = triangles[traceResult.triId].matInd;
		result[ind] = materials[matId].diffuse._x;
		result[ind + 1] = materials[matId].diffuse._y;
		result[ind + 2] = materials[matId].diffuse._z;
	}
	else
	{
		result[ind] = dir.x;
		result[ind + 1] = dir.y;
		result[ind + 2] = dir.z;
	}
}

float4 V32F4(const NPMathHelper::Vec3& vec3)
{
	return make_float4(vec3._x, vec3._y, vec3._z, 0.f);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
void BindCudaTexture(texture<T, dim, readMode> *tex, void* data, size_t size)
{
	tex->normalized = false;
	tex->filterMode = cudaFilterModePoint;
	tex->addressMode[0] = cudaAddressModeWrap;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	HANDLE_ERROR(cudaBindTexture(0, *tex, data, channelDesc, size));
}

bool cudaPT0Render(float3 camPos, float3 camDir, float3 camUp, float fov, RTScene* scene
	, float width, float height, float* result)
{
	// Check and allocate everything
	if (!scene || !scene->GetCompactBVH()->IsValid())
		return false;
	if (!g_bIsCudaInit || scene->GetIsCudaDirty())
	{
		if (g_bIsCudaInit)
		{
			HANDLE_ERROR(cudaUnbindTexture(g_bvhMinMaxBounds));
			HANDLE_ERROR(cudaUnbindTexture(g_bvhOffsetTriStartN));
			HANDLE_ERROR(cudaUnbindTexture(g_triIntersectionData));
		}

		uint triSize = scene->m_pTriangles.size();
		RTTriangle* tempTriangles = new RTTriangle[triSize];
		float4* tempTriIntersectionData = new float4[triSize * 3];
		{
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
				{
					tempTriIntersectionData[i * 3] = V32F4((*scene->GetTriIntersectData())[i * 3]);
					tempTriIntersectionData[i * 3 + 1] = V32F4(((*scene->GetTriIntersectData())[i * 3 + 1] - (*scene->GetTriIntersectData())[i * 3]));
					tempTriIntersectionData[i * 3 + 2] = V32F4(((*scene->GetTriIntersectData())[i * 3 + 2] - (*scene->GetTriIntersectData())[i * 3]));
					tempTriangles[i] = scene->m_pTriangles[i];
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, triSize), f);
		}

		RTMaterial* tempMaterials = new RTMaterial[scene->m_pMaterials.size()];
		{
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
				{
					tempMaterials[i] = scene->m_pMaterials[i];
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, scene->m_pMaterials.size()), f);
		}

		uint bvhNodeN = scene->GetCompactBVH()->nodeN;
		float4* tempBVHMinMaxBounds = new float4[bvhNodeN * 2];
		uint1* tempBVHOffsetTriStartN = new uint1[bvhNodeN * 2];
		{
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
				{
					tempBVHMinMaxBounds[i * 2] = V32F4(scene->GetCompactBVH()->bounds[i].minPoint);
					tempBVHMinMaxBounds[i * 2 + 1] = V32F4(scene->GetCompactBVH()->bounds[i].maxPoint);
					tempBVHOffsetTriStartN[i * 2].x = scene->GetCompactBVH()->offOrTSTN[i * 2];
					tempBVHOffsetTriStartN[i * 2 + 1].x = scene->GetCompactBVH()->offOrTSTN[i * 2 + 1];
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, bvhNodeN), f);
		}
		CUFREE(g_devBVHMinMaxBounds);
		CUFREE(g_devBVHOffsetTriStartN);
		CUFREE(g_devTriIntersectionData);
		CUFREE(g_devTriangles);
		CUFREE(g_devMaterials);

		// Create Dev Data
		cudaMalloc((void**)&g_devMaterials, sizeof(RTMaterial) * scene->m_pMaterials.size());
		cudaMalloc((void**)&g_devTriangles, sizeof(RTTriangle) * triSize);
		cudaMalloc((void**)&g_devTriIntersectionData, sizeof(float4) * triSize * 3);
		cudaMalloc((void**)&g_devBVHOffsetTriStartN, sizeof(uint1) * bvhNodeN * 2);
		cudaMalloc((void**)&g_devBVHMinMaxBounds, sizeof(float4) * bvhNodeN * 2);

		// MemCpy Dev Data
		cudaMemcpy(g_devMaterials, tempMaterials, sizeof(RTMaterial) * scene->m_pMaterials.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(g_devTriangles, tempTriangles, sizeof(RTTriangle) * triSize, cudaMemcpyHostToDevice);
		cudaMemcpy(g_devTriIntersectionData, tempTriIntersectionData, sizeof(float4) * triSize * 3, cudaMemcpyHostToDevice);
		cudaMemcpy(g_devBVHOffsetTriStartN, tempBVHOffsetTriStartN, sizeof(uint1) * bvhNodeN * 2, cudaMemcpyHostToDevice);
		cudaMemcpy(g_devBVHMinMaxBounds, tempBVHMinMaxBounds, sizeof(float4) * bvhNodeN * 2, cudaMemcpyHostToDevice);

		// Del Temp Data
		DEL_ARRAY(tempBVHOffsetTriStartN);
		DEL_ARRAY(tempBVHMinMaxBounds);
		DEL_ARRAY(tempTriIntersectionData);
		DEL_ARRAY(tempTriangles);
		DEL_ARRAY(tempMaterials);

		// Bind Dev Data To Texture
		BindCudaTexture(&g_bvhMinMaxBounds, g_devBVHMinMaxBounds, sizeof(float4) * bvhNodeN * 2);
		BindCudaTexture(&g_bvhOffsetTriStartN, g_devBVHOffsetTriStartN, sizeof(uint1) * bvhNodeN * 2);
		BindCudaTexture(&g_triIntersectionData, g_devTriIntersectionData, sizeof(float4) * triSize * 3);

		g_bIsCudaInit = true;
		scene->SetIsCudaDirty();
	}

	if (!g_bIsCudaInit)
		return false;

	if (!g_devResultData || !g_devAccResultData || g_resultDataSize != (sizeof(float) * 3 * width * height))
	{
		g_resultDataSize = sizeof(float) * 3 * width * height;
		CUFREE(g_devResultData);
		cudaMalloc((void**)&g_devResultData, g_resultDataSize);
		CUFREE(g_devAccResultData);
		cudaMalloc((void**)&g_devAccResultData, g_resultDataSize);
	}

	float3 camRight = normalize(vecCross(camDir, camUp));
	camUp = normalize(vecCross(camRight, camDir));

	// Kernel go here
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	pt0_kernel << < grid, block >> > (camPos, camDir, camUp, camRight, fov, width, height, g_devTriangles, g_devMaterials, g_devTextures, g_devResultData, g_devAccResultData);

	// Copy result to host
	cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
	return true;
}

void freeAllCudaMem()
{
	HANDLE_ERROR(cudaUnbindTexture(g_bvhMinMaxBounds));
	HANDLE_ERROR(cudaUnbindTexture(g_bvhOffsetTriStartN));
	HANDLE_ERROR(cudaUnbindTexture(g_triIntersectionData));
	CUFREE(g_devTriangles);
	CUFREE(g_devMaterials);
	CUFREE(g_devAccResultData);
	CUFREE(g_devResultData);
	CUFREE(g_devBVHMinMaxBounds);
	CUFREE(g_devBVHOffsetTriStartN);
	CUFREE(g_devTriIntersectionData);
}