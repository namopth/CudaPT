#include "cudaRTCommon.h"

#define BLOCK_SIZE 16

namespace cudaRTDebug
{

	float* g_devResultData = nullptr;
	size_t g_resultDataSize = 0;

	struct ShootRayResult
	{
		float4 light;
	};

	__device__ ShootRayResult ptDebug_normalRay(const CURay& ray, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures)
	{
		ShootRayResult rayResult;

		TracePrimitiveResult traceResult;
		if (TracePrimitive(ray, traceResult))
		{
			RTTriangle* tri = &triangles[traceResult.triId];
			RTMaterial* mat = &materials[tri->matInd];
			RTVertex* v0 = &vertices[tri->vertInd0];
			RTVertex* v1 = &vertices[tri->vertInd1];
			RTVertex* v2 = &vertices[tri->vertInd2];
			float2 uv0 = make_float2(v0->tex._x, v0->tex._y);
			float2 uv1 = make_float2(v1->tex._x, v1->tex._y);
			float2 uv2 = make_float2(v2->tex._x, v2->tex._y);
			float2 uv = uv0 * traceResult.w + uv1 * traceResult.u + uv2 * traceResult.v;
			float3 n0 = V32F3(v0->norm);
			float3 n1 = V32F3(v1->norm);
			float3 n2 = V32F3(v2->norm);
			float3 norm = n0 * traceResult.w + n1 * traceResult.u + n2 * traceResult.v;

			float4 diff;
			float3 ambient;
			float3 specular;
			float3 emissive;
			GetMaterialColors(mat, uv, textures, diff, ambient, specular, emissive);

			float3 w = norm;
			float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = vecCross(w, u);

			rayResult.light = diff;
		}
		else
		{
			rayResult.light.x = 1.f;
			rayResult.light.y = 1.f;
			rayResult.light.z = 1.f;
			rayResult.light.w = 1.f;
		}

		return rayResult;
	}

	uint32 WangHash(uint32 a) {
		a = (a ^ 61) ^ (a >> 16);
		a = a + (a << 3);
		a = a ^ (a >> 4);
		a = a * 0x27d4eb2d;
		a = a ^ (a >> 15);
		return a;
	}

	__global__ void ptDebug_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
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

		ShootRayResult rayResult = ptDebug_normalRay(ray, vertices, triangles, materials, textures);

		result[ind] = rayResult.light.x;
		result[ind + 1] = rayResult.light.y;
		result[ind + 2] = rayResult.light.z;
	}

	void cudaDebugClean()
	{
		freeAllBVHCudaMem();
		CUFREE(g_devResultData);
	}

	bool cudaDebugRender(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene* scene
		, float width, float height, float* result)
	{
		// Check and allocate everything
		if (!scene || !scene->GetCompactBVH()->IsValid())
			return false;

		NPMathHelper::Vec3 camRight = camDir.cross(camUp).normalize();
		camUp = camRight.cross(camDir).normalize();

		if (!g_bIsCudaInit || scene->GetIsCudaDirty())
		{
			initAllBVHCudaMem(scene);
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
		ptDebug_kernel << < grid, block >> > (f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height, g_devVertices, g_devTriangles, g_devMaterials, g_devTextures
			, g_devResultData);

		// Copy result to host
		cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
		return true;
	}

};