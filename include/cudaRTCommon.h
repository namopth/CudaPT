#ifndef CUDARAYTRACERCOMMON_H
#define CUDARAYTRACERCOMMON_H
#include "cudahelper.h"
#include "raytracer.h"
#include "mathhelper.h"
#include "attrhelper.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <curand_kernel.h>

#define CUDA_RT_COMMON_ATTRIBS_BGN  NPAttrHelper::Attrib* GetAttribute(unsigned __int32 ind, std::string &name) \
				{ \
		switch (ind) \
								{
#define CUDA_RT_COMMON_ATTRIB_DECLARE(__N__, __NAME__, __VAR__) case __N__: name = #__NAME__; return &__VAR__;
#define CUDA_RT_COMMON_ATTRIBS_END } \
		return nullptr; \
				}
#define CUDA_RT_COMMON_ATTRIBS_N(__N__) \
	unsigned __int32 GetAttribsN() \
				{ \
		return __N__;\
				}

#define BVH_DEPTH_MAX 32
#define BVH_TRACE_MAX 512

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
	bool isValid;
	float3 orig;
	float3 dir;
	__hd__ CURay(float3 _orig, float3 _dir);
	__hd__ CURay();
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
__device__ bool TraceDepthParent(const CURay &ray, int& result, uint& parentId, const uint specDepth, const float maxDist = M_INF, const float rayEpsilon = M_EPSILON, bool cullback = true);
__device__ bool TraceDepth(const CURay &ray, uint& result, bool& isLeaf, const float maxDist = M_INF, const float rayEpsilon = M_EPSILON, bool cullback = true);
__device__ bool TraceCost(const CURay &ray, uint& result, bool& isLeaf, const float maxDist = M_INF, const float rayEpsilon = M_EPSILON, bool cullback = true);

__hd__ float4 V32F4(const NPMathHelper::Vec3& vec3);
__hd__ float3 V32F3(const NPMathHelper::Vec3& vec3);

__device__ bool ProbabilityRand(curandState* state, const float prob);

#pragma region SHADING_FUNC
__device__ float3 Diffuse_Lambert(float3 DiffuseColor);
__device__ float Vis_SmithJointApprox(float Roughness, float NoV, float NoL);
__device__ float D_GGX(float Roughness, float NoH);
__device__ float3 F_Schlick(float3 SpecularColor, float VoH);
__device__ float3 ImportanceSampleGGX(float2 Xi, float Roughness, float3 N);
__device__ float3 Diffuse(float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH);
__device__ float Distribution(float Roughness, float NoH);
__device__ float GeometricVisibility(float Roughness, float NoV, float NoL, float VoH);
__device__ float3 Fresnel(float3 SpecularColor, float VoH);
#pragma endregion SHADING_FUNC

__device__ void GetMaterialColors(const RTMaterial* mat, const float2 uv, const CURTTexture* textures,
	float3 &diff, float3 &normal, float3 &emissive, float &trans, float &specular, float &metallic, float &roughness
	, float &anisotropic, float &sheen, float &sheenTint, float &clearcoat, float &clearcoatGloss);

template<class T, int dim, enum cudaTextureReadMode readMode>
void BindCudaTexture(texture<T, dim, readMode> *tex, void* data, size_t size, uint32 filterMode = cudaFilterModePoint);

CURTTexture CreateCURTTexture(const RTTexture &cpuTex);

void freeAllBVHCudaMem();

void initAllSceneCudaMem(RTScene* scene);
void updateAllSceneMaterialsCudaMem(RTScene* scene);
#endif