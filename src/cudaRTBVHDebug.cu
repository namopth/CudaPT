#include "cudaRTCommon.h"

#define BLOCK_SIZE 16

namespace cudaRTBVHDebug
{
	const char* g_enumDebugModeName[] = { "Normal", "TraceCost", "BVHDepth" };
	NPAttrHelper::Attrib g_enumDebugMode("BVHDebugMode", g_enumDebugModeName, 3, 0);
	NPAttrHelper::Attrib g_uiSpecDepth("SpecifiedBVHDepth", 0);

	CUDA_RT_COMMON_ATTRIBS_N(2)
	CUDA_RT_COMMON_ATTRIBS_BGN
	CUDA_RT_COMMON_ATTRIB_DECLARE(0, Debug Mode, g_enumDebugMode)
	CUDA_RT_COMMON_ATTRIB_DECLARE(1, BVH Depth, g_uiSpecDepth)
	CUDA_RT_COMMON_ATTRIBS_END

	float* g_devResultData = nullptr;
	size_t g_resultDataSize = 0;

	struct ShootRayResult
	{
		float4 light;
	};

	__device__ ShootRayResult ptDebug_normalRay(const CURay& ray, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures)
	{
		ShootRayResult rayResult;

		{
			rayResult.light.x = 0.f;
			rayResult.light.y = 0.f;
			rayResult.light.z = 0.f;
			rayResult.light.w = 0.f;
		}

		uint traceDepth;
		bool leaf;
		if (TraceDepth(ray, traceDepth, leaf))
		{
			float depthValue = traceDepth / 10.f;
			if (leaf)
				rayResult.light.y += depthValue;
			else
				rayResult.light.x += depthValue;
		}

		return rayResult;
	}

	__device__ ShootRayResult ptDebug_costRay(const CURay& ray, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures)
	{
		ShootRayResult rayResult;

		{
			rayResult.light.x = 0.f;
			rayResult.light.y = 0.f;
			rayResult.light.z = 0.f;
			rayResult.light.w = 0.f;
		}

		uint traceCost;
		bool leaf;
		if (TraceCost(ray, traceCost, leaf))
		{
			float costValue = traceCost / 10.f;
			if (leaf)
				rayResult.light.y += costValue;
			else
				rayResult.light.x += costValue;
		}

		return rayResult;
	}

	__device__ ShootRayResult ptDebug_specDepthRay(const uint specDepth, const CURay& ray
		, RTVertex* vertices, RTTriangle* triangles , RTMaterial* materials, CURTTexture* textures)
	{
		ShootRayResult rayResult;

		{
			rayResult.light.x = 0.f;
			rayResult.light.y = 0.f;
			rayResult.light.z = 0.f;
			rayResult.light.w = 0.f;
		}

		int traceDepth;
		uint parentId;
		bool leaf;
		if (TraceDepthParent(ray, traceDepth, parentId, specDepth))
		{
			curandState randstate;
			curand_init(parentId, 0, 0, &randstate);
			float col = curand_uniform(&randstate);
			if (traceDepth == specDepth)
			{
				rayResult.light.y += col;
			}
			else
			{
				rayResult.light.x = 0.3f;
			}
		}

		return rayResult;
	}

	__global__ void ptDebug_kernel(const uint specDepth, const uint debugMode, float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
		float width, float height, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures
		, float* result)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;
		uint ind = (y * width + x) * 3;
		int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


		float u = (2.f * ((float)x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
		float v = (2.f * ((float)y + 0.5f) / height - 1.f) * tan(fov * 0.5f);
		float3 dir = normalize(camRight * u + camUp * v + camDir);
		CURay ray(camPos, dir);

		ShootRayResult rayResult;
		if (debugMode == 0)
		{
			rayResult = ptDebug_normalRay(ray, vertices, triangles, materials, textures);
		}
		else if (debugMode == 1)
		{
			rayResult = ptDebug_costRay(ray, vertices, triangles, materials, textures);
		}
		else
		{
			rayResult = ptDebug_specDepthRay(specDepth, ray, vertices, triangles, materials, textures);
		}

		result[ind] = rayResult.light.x;
		result[ind + 1] = rayResult.light.y;
		result[ind + 2] = rayResult.light.z;
	}

	void CleanMem()
	{
		freeAllBVHCudaMem();
		CUFREE(g_devResultData);
	}

	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene* scene
		, float width, float height, float* result)
	{
		// Check and allocate everything
		if (!scene || !scene->GetCompactBVH()->IsValid())
			return false;

		NPMathHelper::Vec3 camRight = camDir.cross(camUp).normalize();
		camUp = camRight.cross(camDir).normalize();
		
		if (!g_bIsCudaInit || scene->GetIsCudaDirty())
		{
			CleanMem();
			initAllSceneCudaMem(scene);
		}

		if (!g_bIsCudaInit)
			return false;

		if (!g_devResultData || g_resultDataSize != (sizeof(float) * 3 * width * height))
		{
			g_resultDataSize = sizeof(float) * 3 * width * height;
			CUFREE(g_devResultData);
			cudaMalloc((void**)&g_devResultData, g_resultDataSize);
		}

		float3 f3CamPos = V32F3(camPos);
		float3 f3CamUp = V32F3(camUp);
		float3 f3CamDir = V32F3(camDir);
		float3 f3CamRight = V32F3(camRight);

		// Kernel go here
		dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
		dim3 grid(width / block.x, height / block.y, 1);
		ptDebug_kernel << < grid, block >> > (*g_uiSpecDepth.GetUint(), *g_enumDebugMode.GetUint(), f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height, g_devVertices, g_devTriangles, g_devMaterials, g_devTextures
			, g_devResultData);

		// Copy result to host
		cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
		return true;
	}

};