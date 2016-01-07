#include "cudahelper.h"
#include "raytracer.h"
#include "mathhelper.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#define BLOCK_SIZE 16

texture<float4, 1, cudaReadModeElementType> g_bvhMinMaxBounds;
texture<uint1, 1, cudaReadModeElementType> g_bvhOffsetTriStartN;
texture<float4, 1, cudaReadModeElementType> g_triIntersectionData;
RTTriangle* g_devTriangles = nullptr;
RTMaterial* g_devMaterials = nullptr;
float* g_devResultData = nullptr;

float4* g_tempBVHMinMaxBounds = nullptr;
uint1* g_tempBVHOffsetTriStartN = nullptr;
float4* g_tempTriIntersectionData = nullptr;
bool g_bIsCudaInit = false;
size_t g_resultDataSize = 0;

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
	cudaBindTexture(0, *tex, data, channelDesc, size);
}

void cudaRender(float3 camPos, float3 camDir, float3 camUp, float fov, RTScene* scene
	, float width, float height, float* result)
{
	// Check and allocate everything
	if (!scene || !scene->GetCompactBVH()->IsValid())
		return;
	if (!g_bIsCudaInit || scene->GetIsCudaDirty())
	{
		if (g_bIsCudaInit)
		{
			HANDLE_ERROR(cudaUnbindTexture(g_bvhMinMaxBounds));
			HANDLE_ERROR(cudaUnbindTexture(g_bvhOffsetTriStartN));
			HANDLE_ERROR(cudaUnbindTexture(g_triIntersectionData));
		}
		DEL_ARRAY(g_tempBVHMinMaxBounds);
		DEL_ARRAY(g_tempBVHOffsetTriStartN);
		DEL_ARRAY(g_tempTriIntersectionData);

		uint triSize = scene->m_pTriangles.size();
		RTTriangle* tempTriangles = new RTTriangle[triSize];
		g_tempBVHMinMaxBounds = new float4[triSize * 2];
		g_tempBVHOffsetTriStartN = new uint1[triSize * 2];
		g_tempTriIntersectionData = new float4[triSize * 3];
		{
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
				{
					g_tempBVHMinMaxBounds[i * 2] = V32F4(scene->GetCompactBVH()->bounds[i].minPoint);
					g_tempBVHMinMaxBounds[i * 2 + 1] = V32F4(scene->GetCompactBVH()->bounds[i].maxPoint);
					g_tempBVHOffsetTriStartN[i * 2].x = scene->GetCompactBVH()->offOrTSTN[i * 2];
					g_tempBVHOffsetTriStartN[i * 2 + 1].x = scene->GetCompactBVH()->offOrTSTN[i * 2 + 1];
					g_tempTriIntersectionData[i * 3] = V32F4((*scene->GetTriIntersectData())[i * 3]);
					g_tempTriIntersectionData[i * 3 + 1] = V32F4((*scene->GetTriIntersectData())[i * 3 + 1]);
					g_tempTriIntersectionData[i * 3 + 2] = V32F4((*scene->GetTriIntersectData())[i * 3 + 2]);
					tempTriangles[i] = scene->m_pTriangles[i];
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, triSize), f);
		}

		BindCudaTexture(&g_bvhMinMaxBounds, g_tempBVHMinMaxBounds, sizeof(float4) * triSize * 2);
		BindCudaTexture(&g_bvhOffsetTriStartN, g_tempBVHOffsetTriStartN, sizeof(uint1) * triSize * 2);
		BindCudaTexture(&g_triIntersectionData, g_tempTriIntersectionData, sizeof(float4) * triSize * 3);

		CUFREE(g_devTriangles);
		CUFREE(g_devMaterials);
		cudaMalloc((void**)&g_devTriangles, sizeof(RTTriangle) * triSize);
		cudaMemcpy(g_devTriangles, tempTriangles, sizeof(RTTriangle) * triSize, cudaMemcpyHostToDevice);
		DEL_ARRAY(tempTriangles);


		g_bIsCudaInit = true;
		scene->SetIsCudaDirty();
	}

	if (!g_bIsCudaInit)
		return;

	if (!g_devResultData || g_resultDataSize != (sizeof(float) * 3 * width * height))
	{
		g_resultDataSize = sizeof(float) * 3 * width * height;
		CUFREE(g_devResultData);
		cudaMalloc((void**)&g_devResultData, g_resultDataSize);
	}

	float3 camRight = normalize(vecCross(camDir, camUp));
	camUp = normalize(vecCross(camRight, camDir));

	// Kernel go here

	// Copy result to host
	cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
}

void freeAllCudaMem()
{
	HANDLE_ERROR(cudaUnbindTexture(g_bvhMinMaxBounds));
	HANDLE_ERROR(cudaUnbindTexture(g_bvhOffsetTriStartN));
	HANDLE_ERROR(cudaUnbindTexture(g_triIntersectionData));
	CUFREE(g_devTriangles);
	CUFREE(g_devMaterials);
	CUFREE(g_devResultData);
}