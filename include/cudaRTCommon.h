#ifndef CUDARAYTRACERCOMMON_H
#define CUDARAYTRACERCOMMON_H
#include "cudahelper.h"
#include "raytracer.h"
#include "mathhelper.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <curand_kernel.h>

#define BVH_DEPTH_MAX 128

extern texture<float4, 1, cudaReadModeElementType> g_bvhMinMaxBounds;
extern texture<uint1, 1, cudaReadModeElementType> g_bvhOffsetTriStartN;
extern texture<float4, 1, cudaReadModeElementType> g_triIntersectionData;

struct CURTTexture
{
	cudaTextureObject_t texObj;
	cudaArray* cuArray;
	uint width;
	uint height;
	__hd__ CURTTexture();
};
extern CURTTexture* g_devTextures;
extern std::vector<CURTTexture> g_cuRTTextures; // to CUFREE on CPU Side

extern RTVertex* g_devVertices;
extern RTTriangle* g_devTriangles;
extern RTMaterial* g_devMaterials;
extern float4* g_devBVHMinMaxBounds;
extern uint1* g_devBVHOffsetTriStartN;
extern float4* g_devTriIntersectionData;

extern bool g_bIsCudaInit;

struct CURay
{
	float3 orig;
	float3 dir;
	__hd__ CURay(float3 _orig, float3 _dir);
	__hd__ float IntersectAABB(const float3& _min, const float3& _max) const;

	__hd__ float IntersectTri(const float3& _p0, const float3& _e0, const float3& _e1, float& w, float& u, float& v, float epsilon = M_EPSILON, bool cullback = true) const;
};

struct TracePrimitiveResult
{
	float dist;
	int32 triId;
	float w;
	float u;
	float v;
};

__device__ bool TracePrimitive(const CURay &ray, TracePrimitiveResult& result, const float maxDist = M_INF, const float rayEpsilon = M_EPSILON, bool cullback = true);
__device__ bool TraceDepth(const CURay &ray, uint& result, bool& isLeaf, const float maxDist = M_INF, const float rayEpsilon = M_EPSILON, bool cullback = true);

__hd__ float4 V32F4(const NPMathHelper::Vec3& vec3);
__hd__ float3 V32F3(const NPMathHelper::Vec3& vec3);

__device__ void GetMaterialColors(const RTMaterial* mat, const float2 uv, const CURTTexture* textures, float4 &diff, float3 &ambient, float3 &specular, float3 &emissive);

template<class T, int dim, enum cudaTextureReadMode readMode>
void BindCudaTexture(texture<T, dim, readMode> *tex, void* data, size_t size, uint32 filterMode = cudaFilterModePoint);

CURTTexture CreateCURTTexture(const RTTexture &cpuTex);

void freeAllBVHCudaMem();

void initAllBVHCudaMem(RTScene* scene);
#endif