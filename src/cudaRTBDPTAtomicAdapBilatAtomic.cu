#include "cudaRTCommon.h"

#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "conffilehelper.h"

#include "cudaBilateralFilter.h"

#define FANN_NO_DLL
#include "floatfann.h"

#define BLOCK_SIZE 16
#define NORMALRAY_BOUND_MAX 5
#define PATHSTREAM_SIZE 1E4*64

#define LIGHTRAY_BOUND_MAX 5
#define LIGHTVERTEX_N 640
#define PATHQUEUE_MUL 1

#define PERFBREAKDOWN

namespace cudaRTBDPTAtomicAdapBilatAtomic
{
	const char* g_enumAdapModeName[] = { "PDF", "Const" };
	NPAttrHelper::Attrib g_enumAdapMode("Adaptive Mode", g_enumAdapModeName, 2, 0);
	NPAttrHelper::Attrib g_uiDesiredSamplingN("DesiredSamplingN", 5);
	NPAttrHelper::Attrib g_fMinTraceProb("MinTraceProb", 0.f);
	const char* g_enumDebugModeName[] = { "None", "Traced", "Prob", "Prob With Limit", "Filtered Result" };
	NPAttrHelper::Attrib g_enumDebugMode("Debug Mode", g_enumDebugModeName, 5, 0);

	NPAttrHelper::Attrib g_fFilterColorEuD("Filter Color E Delta", 50.f);
	NPAttrHelper::Attrib g_fFilterPosEuD("Filter Pos E Delta", 1.f);
	NPAttrHelper::Attrib g_fFilterNormEuD("Filter Norm E Delta", 0.25f);
	NPAttrHelper::Attrib g_fFilterDiffEuD("Filter Diff E Delta", 0.1f);
	NPAttrHelper::Attrib g_uiFilterRadius("Filter Radius", 5);
	NPAttrHelper::Attrib g_bFilterDiffuse("Filter Diff Flag", false);

#ifdef PERFBREAKDOWN
	NPAttrHelper::Attrib g_fAvgProcessTimeA("Avg Proc Time A", 0.f);
	NPAttrHelper::Attrib g_fAvgProcessTimeB("Avg Proc Time B", 0.f);
	NPAttrHelper::Attrib g_fAvgProcessTimeC("Avg Proc Time C", 0.f);
	NPAttrHelper::Attrib g_fAvgProcessTimeD("Avg Proc Time D", 0.f);
	NPAttrHelper::Attrib g_fAvgProcessTimeE("Avg Proc Time E", 0.f);
#endif

#ifdef PERFBREAKDOWN
	RT_ATTRIBS_N(15)
#else
	RT_ATTRIBS_N(10)
#endif
		RT_ATTRIBS_BGN
		RT_ATTRIB_DECLARE(0, Adaptive Mode, g_enumAdapMode)
		RT_ATTRIB_DECLARE(1, Desired Max Sampling, g_uiDesiredSamplingN)
		RT_ATTRIB_DECLARE(2, Min Trace Probability, g_fMinTraceProb)
		RT_ATTRIB_DECLARE(3, Debug Mode, g_enumDebugMode)
		RT_ATTRIB_DECLARE(4, Filter Color EuD, g_fFilterColorEuD)
		RT_ATTRIB_DECLARE(5, Filter Pos EuD, g_fFilterPosEuD)
		RT_ATTRIB_DECLARE(6, Filter Norm EuD, g_fFilterNormEuD)
		RT_ATTRIB_DECLARE(7, Filter Diff EuD, g_fFilterDiffEuD)
		RT_ATTRIB_DECLARE(8, Filter Radius, g_uiFilterRadius)
		RT_ATTRIB_DECLARE(9, Filter Diffuse Flag, g_bFilterDiffuse)
#ifdef PERFBREAKDOWN
		RT_ATTRIB_DECLARE(10, Avg Filter Time, g_fAvgProcessTimeA)
		RT_ATTRIB_DECLARE(11, Avg MSE Time, g_fAvgProcessTimeB)
		RT_ATTRIB_DECLARE(12, Avg Gen Time, g_fAvgProcessTimeC)
		RT_ATTRIB_DECLARE(13, Avg Trace Time, g_fAvgProcessTimeD)
		RT_ATTRIB_DECLARE(14, Avg Accum Time, g_fAvgProcessTimeE)
#endif
		RT_ATTRIBS_END

	struct LightVertex
	{
		float3 pos;
		float3 norm;
		float3 irrad;
		float3 irradDir;

		float3 diff;
		float3 emissive;
		float specular;
		float metallic;
		float roughness;

		float pathPotential;

		__hd__ LightVertex()
		{
			pos = norm = irrad = irradDir = make_float3(0.f, 0.f, 0.f);
			pathPotential = 1.f;
		}
	};

	LightVertex* g_devLightVertices = nullptr;
	uint g_uLightVerticesSize = 0;
	uint* g_devLightTri = nullptr;
	uint g_lightTriN = 0;

	void freeLightPathMem()
	{
		g_uLightVerticesSize = 0;
		g_lightTriN = 0;
		CUFREE(g_devLightVertices);
		CUFREE(g_devLightTri);
	}

	void allocateLightPathMem()
	{
		HANDLE_ERROR(cudaMalloc((void**)&g_devLightVertices, sizeof(LightVertex) * LIGHTVERTEX_N));
		HANDLE_ERROR(cudaMemset((void*)g_devLightVertices, 0, sizeof(LightVertex) * LIGHTVERTEX_N));
	}

	void updateLightTriCudaMem(RTScene* scene)
	{
		g_lightTriN = 0;
		CUFREE(g_devLightTri);
		std::vector<uint> lightTri;
		for (uint i = 0; i < scene->m_pTriangles.size(); i++)
		{
			if (NPMathHelper::Vec3::length(scene->m_pMaterials[scene->m_pTriangles[i].matInd].emissive) > 0.f)
				lightTri.push_back(i);
		}
		uint* tempLightTri = new uint[lightTri.size()];
		for (uint i = 0; i < lightTri.size(); i++)
		{
			tempLightTri[i] = lightTri[i];
		}
		g_lightTriN = lightTri.size();
		HANDLE_ERROR(cudaMalloc((void**)&g_devLightTri, sizeof(uint) * g_lightTriN));
		HANDLE_ERROR(cudaMemcpy(g_devLightTri, tempLightTri, sizeof(uint) * g_lightTriN, cudaMemcpyHostToDevice));

		DEL_ARRAY(tempLightTri);
	}

	enum RAYTYPE
	{
		RAYTYPE_EYE = 0,
		RAYTYPE_DIFF = 1,
		RAYTYPE_SPEC = 2,
		RAYTYPE_LIGHT = 3
	};

	struct PTPathVertex
	{
		uint isTerminated;
		uint2 pathPixel;
		float3 pathOutDir;
		float3 pathVertexPos;
		float3 pathOutMulTerm;
		RAYTYPE pathType;
		float3 pathSample;
		float3 pathAccumSample;
		uint pathSampleN;
		uint pathSampleDepth;
		curandState randState;

		// for connecting light path
		float3 pathInMulTerm;
		float3 pathInDir;
		float3 origNorm;
		float3 origDiff;
		float origMetallic;
		float origRoughness;
		float origSpecular;
		float origTrans;

		// for adaptive sampling
		float pathPotential;
		float pathAccumPotential;

		// for filtering
		float3 pathDirectPos;
		float3 pathDirectNorm;
		float3 pathDirectDiffuse;

		__device__ PTPathVertex()
			: isTerminated(true)
			, pathPixel(make_uint2(0, 0))
			, pathOutDir(make_float3(0.f, 1.f, 0.f))
			, pathVertexPos(make_float3(0.f, 0.f, 0.f))
			, pathOutMulTerm(make_float3(1.f, 1.f, 1.f))
			, pathType(RAYTYPE_EYE)
			, pathSample(make_float3(0.f, 0.f, 0.f))
			, pathAccumSample(make_float3(0.f, 0.f, 0.f))
			, pathSampleN(0)
			, pathSampleDepth(0)
			, randState()
			, pathInMulTerm(make_float3(0.f, 0.f, 0.f))
			, pathInDir(make_float3(0.f, 0.f, 0.f))
			, origNorm(make_float3(0.f, 1.f, 0.f))
			, origDiff(make_float3(0.f, 0.f, 0.f))
			, origMetallic(0.f)
			, origRoughness(0.f)
			, origSpecular(0.f)
			, origTrans(0.f)
			, pathPotential(1.f)
			, pathAccumPotential(0.f)
			, pathDirectPos(make_float3(0.f, 0.f, 0.f))
			, pathDirectNorm(make_float3(0.f, 0.f, 0.f))
			, pathDirectDiffuse(make_float3(0.f, 0.f, 0.f))
		{}

		__device__ PTPathVertex(uint _isTerminated, uint2 _pathPixel, float3 _pathOutDir, float3 _pathVertexPos, RAYTYPE _pathType, curandState _randState)
			: isTerminated(_isTerminated)
			, pathPixel(_pathPixel)
			, pathOutDir(_pathOutDir)
			, pathVertexPos(_pathVertexPos)
			, pathOutMulTerm(make_float3(1.f, 1.f, 1.f))
			, pathType(_pathType)
			, pathSample(make_float3(0.f, 0.f, 0.f))
			, pathAccumSample(make_float3(0.f, 0.f, 0.f))
			, pathSampleN(0)
			, pathSampleDepth(0)
			, randState(_randState)
			, pathInMulTerm(make_float3(0.f, 0.f, 0.f))
			, pathInDir(make_float3(0.f, 0.f, 0.f))
			, origNorm(make_float3(0.f, 1.f, 0.f))
			, origDiff(make_float3(0.f, 0.f, 0.f))
			, origMetallic(0.f)
			, origRoughness(0.f)
			, origSpecular(0.f)
			, origTrans(0.f)
			, pathPotential(1.f)
			, pathAccumPotential(0.f)
			, pathDirectPos(make_float3(0.f, 0.f, 0.f))
			, pathDirectNorm(make_float3(0.f, 0.f, 0.f))
			, pathDirectDiffuse(make_float3(0.f, 0.f, 0.f))
		{}
	};

	uint* g_devAtomicN = nullptr;
	uint* g_devTempPathQueue = nullptr;
	PTPathVertex* g_devPathQueue = nullptr;
	uint g_uPathQueueCur = 0;
	uint g_uPathQueueSize = 0;
	PTPathVertex** g_devPathStream = nullptr;
	PTPathVertex** g_devTempPathStream = nullptr;
	PTPathVertex** g_devEyeLightConPathStream = nullptr;
	uint g_uPathStreamSize = PATHSTREAM_SIZE;

	void freeStreamMem()
	{
		g_uPathQueueCur = g_uPathQueueSize = 0;
		CUFREE(g_devTempPathQueue);
		CUFREE(g_devPathQueue);
		CUFREE(g_devPathStream);
		CUFREE(g_devTempPathStream);
		CUFREE(g_devEyeLightConPathStream);
		CUFREE(g_devAtomicN);
	}

	void allocateStreamMem(uint queueSize = 480000)
	{
		g_uPathQueueSize = queueSize;
		HANDLE_ERROR(cudaMalloc((void**)&g_devPathQueue, sizeof(PTPathVertex) * g_uPathQueueSize * PATHQUEUE_MUL));
		HANDLE_ERROR(cudaMemset((void*)g_devPathQueue, 0, sizeof(PTPathVertex) * g_uPathQueueSize * PATHQUEUE_MUL));

		HANDLE_ERROR(cudaMalloc((void**)&g_devTempPathQueue, sizeof(uint) * g_uPathQueueSize * PATHQUEUE_MUL));
		HANDLE_ERROR(cudaMemset((void*)g_devTempPathQueue, 0, sizeof(uint) * g_uPathQueueSize * PATHQUEUE_MUL));

		HANDLE_ERROR(cudaMalloc((void**)&g_devPathStream, sizeof(PTPathVertex*) * g_uPathStreamSize));
		HANDLE_ERROR(cudaMemset((void*)g_devPathStream, 0, sizeof(PTPathVertex*) * g_uPathStreamSize));

		HANDLE_ERROR(cudaMalloc((void**)&g_devTempPathStream, sizeof(PTPathVertex*) * g_uPathStreamSize));
		HANDLE_ERROR(cudaMemset((void*)g_devTempPathStream, 0, sizeof(PTPathVertex*) * g_uPathStreamSize));

		HANDLE_ERROR(cudaMalloc((void**)&g_devEyeLightConPathStream, sizeof(PTPathVertex*) * g_uPathStreamSize));
		HANDLE_ERROR(cudaMemset((void*)g_devEyeLightConPathStream, 0, sizeof(PTPathVertex*) * g_uPathStreamSize));

		HANDLE_ERROR(cudaMalloc((void**)&g_devAtomicN, sizeof(uint)));
		HANDLE_ERROR(cudaMemset((void*)g_devAtomicN, 0, sizeof(uint)));
	}

	float* g_devResultData = nullptr;
	float* g_devAccResultData = nullptr;
	float* g_devResultVarData = nullptr;
	uint* g_devSampleResultN = nullptr;
	float* g_devTempResultData = nullptr;
	uint* g_devTempResultN = nullptr;

	float* g_devTempPositionData = nullptr;
	float* g_devTempNormalData = nullptr;
	float* g_devTempDiffuseData = nullptr;

	float* g_devPositionData = nullptr;
	float* g_devNormalData = nullptr;
	float* g_devDiffuseData = nullptr;

	float* g_devFilteredResult = nullptr;
	float* g_devFilterGaussianConst = nullptr;

	NPMathHelper::Mat4x4 g_matLastCamMat;
	NPMathHelper::Mat4x4 g_matCurCamMat;
	uint32 g_uCurFrameN = 0;
	size_t g_resultDataSize = 0;

	uint32 WangHash(uint32 a) {
		a = (a ^ 61) ^ (a >> 16);
		a = a + (a << 3);
		a = a ^ (a >> 4);
		a = a * 0x27d4eb2d;
		a = a ^ (a >> 15);
		return a;
	}

	__global__ void pt_traceLight_kernel(RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures, PTPathVertex** pathStream, uint activePathStreamSize, LightVertex* lightVertices, uint curLightVerticesSize)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= activePathStreamSize || pathStream[x]->isTerminated) return;
		PTPathVertex* procVertex = pathStream[x];
		CURay ray = CURay(procVertex->pathVertexPos, procVertex->pathOutDir);
		TracePrimitiveResult traceResult;
		if (TracePrimitive(ray, traceResult, M_INF, M_FLT_BIAS_EPSILON, false))
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
			float3 triPos = V32F3(v0->pos) * traceResult.w + V32F3(v1->pos) * traceResult.u + V32F3(v2->pos) * traceResult.v;

			float3 diff;
			float3 emissive;
			float trans;
			float specular;
			float metallic;
			float roughness;
			float anisotropic;
			float sheen;
			float sheenTint;
			float clearcoat;
			float clearcoatGloss;
			GetMaterialColors(mat, uv, textures, diff, norm, emissive, trans, specular, metallic, roughness
				, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss);
			float3 nl = vecDot(norm, ray.dir) < 0.f ? norm : -1 * norm;

			lightVertices[curLightVerticesSize + x].irrad = procVertex->pathSample;
			lightVertices[curLightVerticesSize + x].irradDir = -1 * ray.dir;
			lightVertices[curLightVerticesSize + x].norm = nl;
			lightVertices[curLightVerticesSize + x].pos = triPos;
			lightVertices[curLightVerticesSize + x].diff = diff;
			lightVertices[curLightVerticesSize + x].emissive = emissive;
			lightVertices[curLightVerticesSize + x].specular = specular;
			lightVertices[curLightVerticesSize + x].metallic = metallic;
			lightVertices[curLightVerticesSize + x].roughness = roughness;
			lightVertices[curLightVerticesSize + x].pathPotential = procVertex->pathPotential;
			{
				// Get some random microfacet
				float3 hDir = ImportanceSampleGGX(make_float2(curand_uniform(&procVertex->randState), curand_uniform(&procVertex->randState)), roughness, nl);

				// Calculate flesnel
				float voH = vecDot(-1 * ray.dir, hDir);
				float3 f0 = vecLerp(0.08 * make_float3(specular, specular, specular), diff, metallic);
				float3 brdf_f = Fresnel(f0, voH);

				// PDF
				float NoH = vecDot(nl, hDir);
				float VoH = vecDot(-1 * ray.dir, hDir);
				float pdf = D_GGX(roughness, NoH) * NoH / (4.f * VoH);

				// Reflected or Refracted
				float reflProb = lerp(length(brdf_f), 1.0f, metallic);
				float refrProb = trans;
				float3 reflDir;
				float3 refrDir;

				CURay nextRay = ray;
				float3 lightMulTerm;
				RAYTYPE nextRayType = procVertex->pathType;

				if (refrProb > 0)
				{
					bool into = vecDot(nl, norm) > 0.f;
					float nt = specular * 0.8f + 1.f;
					float nc = 1.0f;
					float nnt = into ? nc / nt : nt / nc;
					float ddn = vecDot(hDir, ray.dir);
					float cos2t = 1.f - nnt * nnt *(1.f - ddn * ddn);
					//if (cos2t < 0.f)
					//{
					//	reflProb = 1.0f;//refrProb = 0.f;
					//}
					//else
					{
						refrDir = normalize(ray.dir * nnt - hDir * (ddn*nnt + sqrtf(cos2t)));
					}
				}

				if (reflProb > 0)
				{
					reflDir = normalize(ray.dir - hDir * 2 * vecDot(hDir, ray.dir));
					if (vecDot(reflDir, nl) < 0.f)
						reflProb = 0.f;
				}

				// Reflected
				if (ProbabilityRand(&procVertex->randState, reflProb))
				{
					nextRay = CURay(ray.orig + (traceResult.dist - M_FLT_BIAS_EPSILON) * ray.dir, reflDir);
					// ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

					// Microfacet specular = D*G*F / (4*NoL*NoV)
					// pdf = D * NoH / (4 * VoH)
					// (G * F * VoH) / (NoV * NoH)
					float NoV = vecDot(nl, -1 * ray.dir);
					float NoL = vecDot(nl, reflDir);
					float G = GeometricVisibility(roughness, NoV, NoL, VoH);
					//shadeResult = vecMul((brdf_f * G * VoH) / (NoV * NoH * reflProb) , nextRayResult.light) + emissive;
					lightMulTerm = (brdf_f * G * VoH) / (NoV * NoH * reflProb);
					nextRayType = RAYTYPE_SPEC;
					pdf *= reflProb;
				}

				// Diffused or Transmited
				else
				{
					// Transmited
					if (ProbabilityRand(&procVertex->randState, refrProb))
					{
						nextRay = CURay(ray.orig + (traceResult.dist + M_FLT_BIAS_EPSILON) * ray.dir + refrDir * M_FLT_BIAS_EPSILON, refrDir);
						//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
						float cosine = vecDot(-1 * nl, refrDir);
						//shadeResult = (cosine * vecMul(diff, nextRayResult.light)) / (refrProb * (1 - reflProb)) + emissive;
						lightMulTerm = cosine * diff / (refrProb * (1 - reflProb));
						nextRayType = RAYTYPE_SPEC;
						pdf *= (refrProb * (1.f - reflProb));
					}
					// Diffused
					else
					{
						float3 w = nl;
						float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
						float3 v = vecCross(w, u);
						u = vecCross(v, w);

						float r1 = 2.f * M_PI * curand_uniform(&procVertex->randState);
						float r2cos = sqrtf(curand_uniform(&procVertex->randState));
						float r2sin = 1.f - r2cos*r2cos;
						float3 diffDir = normalize(w * r2cos + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

						nextRay = CURay(ray.orig + traceResult.dist * ray.dir + diffDir * M_FLT_BIAS_EPSILON, diffDir);
						//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

						float VoH = vecDot(-1 * ray.dir, hDir);
						float NoV = vecDot(nl, -1 * ray.dir);
						float NoL = vecDot(nl, diffDir);
						//shadeResult = (M_PI * vecMul(Diffuse(diff, roughness, NoV, NoL, VoH), nextRayResult.light)) / ((1 - refrProb) * (1 - reflProb)) + emissive;
						lightMulTerm = M_PI * Diffuse(diff, roughness, NoV, NoL, VoH) / ((1 - refrProb) * (1 - reflProb));
						nextRayType = RAYTYPE_DIFF;
						pdf *= ((1.f - refrProb) * (1.f - reflProb)) * vecDot(diffDir, nl);
					}
				}

				if (nextRayType != RAYTYPE_DIFF)
					lightVertices[curLightVerticesSize + x].irrad = make_float3(0.f, 0.f, 0.f);

				if (vecDot(nextRay.dir, nl) < 0.f)
					lightVertices[curLightVerticesSize + x].norm = -1 * lightVertices[curLightVerticesSize + x].norm;

				procVertex->pathSample = emissive + vecMul(procVertex->pathSample, lightMulTerm);
				procVertex->pathPotential *= pdf;

				float pixelContrib = length(procVertex->pathOutMulTerm) * length(lightMulTerm);

				if (/*(procVertex->pathType == RAYTYPE_DIFF && nextRayType == RAYTYPE_SPEC) ||*/ length(emissive) > 0.f)
					pixelContrib = 0.f;

				if (curand_uniform(&procVertex->randState) > pixelContrib || procVertex->pathSampleDepth + 1 >= NORMALRAY_BOUND_MAX)
				{
					procVertex->isTerminated = true;
				}
				else
				{
					procVertex->pathOutMulTerm = vecMul(procVertex->pathOutMulTerm, lightMulTerm);
					procVertex->pathOutDir = nextRay.dir;
					procVertex->pathVertexPos = nextRay.orig;
					procVertex->pathType = nextRayType;
					procVertex->pathSampleDepth++;
				}
			}
		}
		else
		{
			lightVertices[curLightVerticesSize + x] = lightVertices[procVertex->pathPixel.x];
			procVertex->isTerminated = true;
		}
	}

	__global__ void pt_traceSample_kernel(RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures, PTPathVertex** pathStream, uint activePathStreamSize)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= activePathStreamSize || pathStream[x]->isTerminated) return;
		PTPathVertex* procVertex = pathStream[x];
		CURay ray = CURay(procVertex->pathVertexPos, procVertex->pathOutDir);
		TracePrimitiveResult traceResult;
		if (TracePrimitive(ray, traceResult, M_INF, M_FLT_BIAS_EPSILON, false))
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
			float3 triPos = V32F3(v0->pos) * traceResult.w + V32F3(v1->pos) * traceResult.u + V32F3(v2->pos) * traceResult.v;

			float3 diff;
			float3 emissive;
			float trans;
			float specular;
			float metallic;
			float roughness;
			float anisotropic;
			float sheen;
			float sheenTint;
			float clearcoat;
			float clearcoatGloss;
			GetMaterialColors(mat, uv, textures, diff, norm, emissive, trans, specular, metallic, roughness
				, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss);
			float3 nl = vecDot(norm, ray.dir) < 0.f ? norm : -1 * norm;
			{
				// Get some random microfacet
				float3 hDir = ImportanceSampleGGX(make_float2(curand_uniform(&procVertex->randState), curand_uniform(&procVertex->randState)), roughness, nl);

				// Calculate flesnel
				float voH = vecDot(-1 * ray.dir, hDir);
				float3 f0 = vecLerp(0.08 * make_float3(specular, specular, specular), diff, metallic);
				float3 brdf_f = Fresnel(f0, voH);

				// PDF
				float NoH = vecDot(nl, hDir);
				float VoH = vecDot(-1 * ray.dir, hDir);
				float pdf = D_GGX(roughness, NoH) * NoH / (4.f * VoH + M_EPSILON);

				// Reflected or Refracted
				float reflProb = lerp(length(brdf_f), 1.0f, metallic);
				float refrProb = trans;
				float3 reflDir;
				float3 refrDir;

				CURay nextRay = ray;
				float3 lightMulTerm;
				RAYTYPE nextRayType = procVertex->pathType;

				if (refrProb > 0)
				{
					bool into = vecDot(nl, norm) > 0.f;
					float nt = specular * 0.8f + 1.f;
					float nc = 1.0f;
					float nnt = into ? nc / nt : nt / nc;
					float ddn = vecDot(hDir, ray.dir);
					float cos2t = 1.f - nnt * nnt *(1.f - ddn * ddn);
					//if (cos2t < 0.f)
					//{
					//	reflProb = 1.0f;//refrProb = 0.f;
					//}
					//else
					{
						refrDir = normalize(ray.dir * nnt - hDir * (ddn*nnt + sqrtf(cos2t)));
					}
				}

				if (reflProb > 0)
				{
					reflDir = normalize(ray.dir - hDir * 2 * vecDot(hDir, ray.dir));
					if (vecDot(reflDir, nl) < 0.f)
						reflProb = 0.f;
				}

				// Reflected
				if (ProbabilityRand(&procVertex->randState, reflProb))
				{
					nextRay = CURay(ray.orig + (traceResult.dist - M_FLT_BIAS_EPSILON) * ray.dir + reflDir * M_FLT_BIAS_EPSILON, reflDir);
					// ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

					// Microfacet specular = D*G*F / (4*NoL*NoV)
					// pdf = D * NoH / (4 * VoH)
					// (G * F * VoH) / (NoV * NoH)
					float NoV = vecDot(nl, -1 * ray.dir);
					float NoL = vecDot(nl, reflDir);
					float G = GeometricVisibility(roughness, NoV, NoL, VoH);
					//shadeResult = vecMul((brdf_f * G * VoH) / (NoV * NoH * reflProb) , nextRayResult.light) + emissive;
					lightMulTerm = (brdf_f * G * VoH) / (NoV * NoH * reflProb);
					nextRayType = RAYTYPE_SPEC;
					pdf *= reflProb;
				}

				// Diffused or Transmited
				else
				{
					// Transmited
					if (ProbabilityRand(&procVertex->randState, refrProb))
					{
						nextRay = CURay(ray.orig + (traceResult.dist + M_FLT_BIAS_EPSILON) * ray.dir + refrDir * M_FLT_BIAS_EPSILON, refrDir);
						//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
						float cosine = vecDot(-1 * nl, refrDir);
						//shadeResult = (cosine * vecMul(diff, nextRayResult.light)) / (refrProb * (1 - reflProb)) + emissive;
						lightMulTerm = cosine * diff / (refrProb * (1 - reflProb));
						nextRayType = RAYTYPE_SPEC;
						pdf *= (refrProb * (1.f - reflProb));
					}
					// Diffused
					else
					{
						float3 w = nl;
						float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
						float3 v = vecCross(w, u);
						u = vecCross(v, w);

						float r1 = 2.f * M_PI * curand_uniform(&procVertex->randState);
						float r2cos = sqrtf(curand_uniform(&procVertex->randState));
						float r2sin = 1.f - r2cos*r2cos;
						float3 diffDir = normalize(w * r2cos + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

						nextRay = CURay(ray.orig + traceResult.dist * ray.dir + diffDir * M_FLT_BIAS_EPSILON, diffDir);
						//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

						float VoH = vecDot(-1 * ray.dir, hDir);
						float NoV = vecDot(nl, -1 * ray.dir);
						float NoL = vecDot(nl, diffDir);
						//shadeResult = (M_PI * vecMul(Diffuse(diff, roughness, NoV, NoL, VoH), nextRayResult.light)) / ((1 - refrProb) * (1 - reflProb)) + emissive;
						lightMulTerm = M_PI * Diffuse(diff, roughness, NoV, NoL, VoH) / ((1 - refrProb) * (1 - reflProb));
						nextRayType = RAYTYPE_DIFF;
						pdf *= ((1.f - refrProb) * (1.f - reflProb)) * vecDot(diffDir, nl);
					}
				}

				procVertex->pathSample = procVertex->pathSample + vecMul(emissive, procVertex->pathOutMulTerm);

				procVertex->origDiff = diff;
				procVertex->pathInDir = -1 * ray.dir;
				procVertex->origNorm = nl;
				procVertex->origRoughness = roughness;
				procVertex->origMetallic = metallic;
				procVertex->origSpecular = specular;
				procVertex->origTrans = trans;
				procVertex->pathInMulTerm = procVertex->pathOutMulTerm;
				procVertex->pathPotential *= pdf;

				if (procVertex->pathSampleDepth == 0)
				{
					procVertex->pathDirectPos = triPos;
					procVertex->pathDirectNorm = nl;
					procVertex->pathDirectDiffuse = diff;
				}

				float pixelContrib = length(procVertex->pathOutMulTerm) * length(lightMulTerm);

				if (/*(procVertex->pathType == RAYTYPE_DIFF && nextRayType == RAYTYPE_SPEC) ||*/ length(emissive) > 0.f)
					pixelContrib = 0.f;

				if (curand_uniform(&procVertex->randState) > pixelContrib || procVertex->pathSampleDepth + 1 >= NORMALRAY_BOUND_MAX)
				{
					procVertex->pathAccumSample = procVertex->pathAccumSample + procVertex->pathSample;
					procVertex->pathAccumPotential = procVertex->pathAccumPotential + procVertex->pathPotential;
					procVertex->pathSampleN++;
					procVertex->isTerminated = true;
				}
				else
				{
					procVertex->pathOutMulTerm = vecMul(procVertex->pathOutMulTerm, lightMulTerm);
					procVertex->pathOutDir = nextRay.dir;
					procVertex->pathSampleDepth++;
				}
				procVertex->pathVertexPos = nextRay.orig;
				procVertex->pathType = nextRayType;

			}
		}
		else
		{
			procVertex->pathAccumSample = procVertex->pathAccumSample + procVertex->pathSample;
			procVertex->pathAccumPotential = procVertex->pathAccumPotential + procVertex->pathPotential;
			procVertex->pathSampleN++;
			procVertex->isTerminated = true;
		}
	}

	__global__ void pt_genLightPathQueue_kernel(uint32 frameN, uint32 hashedFrameN, uint* lightTri, uint lightTriN, RTVertex* vertices,
		RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures, PTPathVertex* pathQueue, uint pathQueueCap, LightVertex* lightVertices, uint curLightVerticesSize)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x > pathQueueCap) return;

		curandState randstate;
		curand_init(hashedFrameN + x, 0, 0, &randstate);

		uint lightSourceId = curand_uniform(&randstate) * lightTriN;
		float lightW = curand_uniform(&randstate);
		float lightU = curand_uniform(&randstate);
		if (lightW + lightU > 1.0f)
		{
			lightW = 1.f - lightW;
			lightU = 1.f - lightU;
		}
		float lightV = 1.f - lightW - lightU;

		uint triId = lightTri[lightSourceId];
		RTTriangle* tri = &triangles[triId];
		RTMaterial* mat = &materials[tri->matInd];
		RTVertex* v0 = &vertices[tri->vertInd0];
		RTVertex* v1 = &vertices[tri->vertInd1];
		RTVertex* v2 = &vertices[tri->vertInd2];
		float2 uv0 = make_float2(v0->tex._x, v0->tex._y);
		float2 uv1 = make_float2(v1->tex._x, v1->tex._y);
		float2 uv2 = make_float2(v2->tex._x, v2->tex._y);
		float2 uv = uv0 * lightW + uv1 * lightU + uv2 * lightV;
		float3 n0 = V32F3(v0->norm);
		float3 n1 = V32F3(v1->norm);
		float3 n2 = V32F3(v2->norm);
		float3 triNorm = n0 * lightW + n1 * lightU + n2 * lightV;
		float3 triPos = V32F3(v0->pos) * lightW + V32F3(v1->pos) * lightU + V32F3(v2->pos) * lightV;

		float3 diff;
		float3 emissive;
		float trans;
		float specular;
		float metallic;
		float roughness;
		float anisotropic;
		float sheen;
		float sheenTint;
		float clearcoat;
		float clearcoatGloss;
		GetMaterialColors(mat, uv, textures, diff, triNorm, emissive, trans, specular, metallic, roughness
			, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss);

		float3 w = triNorm;
		float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = vecCross(w, u);
		u = vecCross(v, w);

		float r1 = 2.f * M_PI * curand_uniform(&randstate);
		float r2cos = sqrtf(curand_uniform(&randstate));
		float r2sin = 1.f - r2cos*r2cos;
		float3 diffDir = normalize(w * r2cos + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

		pathQueue[x] = PTPathVertex(false, make_uint2(curLightVerticesSize + x, 0), diffDir
			, triPos + M_FLT_BIAS_EPSILON * triNorm, RAYTYPE_LIGHT, randstate);
		pathQueue[x].pathSample = emissive;

		lightVertices[curLightVerticesSize + x].irrad = emissive;
		lightVertices[curLightVerticesSize + x].irradDir = make_float3(0.f, 0.f, 0.f);
		lightVertices[curLightVerticesSize + x].norm = triNorm;
		lightVertices[curLightVerticesSize + x].pos = triPos;
		lightVertices[curLightVerticesSize + x].diff = diff;
		lightVertices[curLightVerticesSize + x].emissive = emissive;
		lightVertices[curLightVerticesSize + x].specular = specular;
		lightVertices[curLightVerticesSize + x].metallic = metallic;
		lightVertices[curLightVerticesSize + x].roughness = roughness;
	}

	__global__ void pt_genPathQueue_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
		float width, float height, uint32 frameN, uint32 hashedFrameN, PTPathVertex* pathQueue)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height) return;

		uint ind = (y * width + x);

		float u = (2.f * ((float)x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
		float v = (2.f * ((float)y + 0.5f) / height - 1.f) * tan(fov * 0.5f);

		curandState randstate;
		curand_init(hashedFrameN + ind, 0, 0, &randstate);
		float au = u + (curand_uniform(&randstate) - 0.5f) / height * tan(fov * 0.5f);
		float av = v + (curand_uniform(&randstate) - 0.5f) / height * tan(fov * 0.5f);

		float3 dir = normalize(camRight * au + camUp * av + camDir);

		pathQueue[ind] = PTPathVertex(false, make_uint2(x, y), dir, camPos, RAYTYPE_EYE, randstate);
	}

	__global__ void pt_fillTempAdapPathQueue_kernel(uint* pathQueue, uint fillSize)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;

		if (ind >= fillSize) return;

		pathQueue[ind] = ind;
	}

	__global__ void pt_genTempAdapPathQueue_kernel(float width, float height, uint32 hashedFrameN, uint32 seedoffset
		, float* genChance, uint* pathQueue, float minProb = 0.f, float mulRand = 1.f)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height) return;

		uint ind = (y * width + x);

		curandState randstate;
		curand_init(hashedFrameN + ind + seedoffset, 0, 0, &randstate);

		pathQueue[ind] = x + y * width;

		//float modChance = 1.f - expf(-genChance[ind]);
		if (curand_uniform(&randstate)*mulRand > fmaxf(genChance[ind], minProb))
		{
			pathQueue[ind] = 0 - 1;
		}
	}

	__global__ void pt_genAdapPathQueue_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
		float width, float height, uint32 frameN, uint32 hashedFrameN, uint* atomicN, float* mseData, float sumMSE, PTPathVertex* pathQueue, uint genSize)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (ind >= width * height) return;

		uint y = ind / width;
		uint x = ind - y * width;

		float u = (2.f * ((float)x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
		float v = (2.f * ((float)y + 0.5f) / height - 1.f) * tan(fov * 0.5f);

		
		float mseRatio = mseData[ind] / sumMSE;
		uint sampleN = genSize * mseRatio;

		for (uint i = 0; i < sampleN; i++)
		{
			curandState randstate;
			curand_init(hashedFrameN + ind + i * genSize, 0, 0, &randstate);
			float au = u + (curand_uniform(&randstate) - 0.5f) / height * tan(fov * 0.5f);
			float av = v + (curand_uniform(&randstate) - 0.5f) / height * tan(fov * 0.5f);
			float3 dir = normalize(camRight * au + camUp * av + camDir);
			pathQueue[atomicAggInc((int *)atomicN)] = PTPathVertex(false, make_uint2(x, y), dir, camPos, RAYTYPE_EYE, randstate);
		}
	}

	__device__ float3 GetShadingResult(const float3& lightOutDir, const float3& lightInDir, const float3& lightInIrrad, const float3& norm,
		const float3& diff, const float metallic, const float roughness, const float specular, const float2 diffspec)
	{
		if (vecDot(norm, lightInDir) <= 0.f)
			return make_float3(0.f, 0.f, 0.f);

		float3 h = normalize(lightOutDir + lightInDir);

		float voH = vecDot(lightOutDir, h);
		float noV = vecDot(norm, lightOutDir);
		float noH = vecDot(norm, h);
		float noL = vecDot(norm, lightInDir);
		float3 f0 = vecLerp(0.08f * specular * make_float3(1.f, 1.f, 1.f), diff, metallic);
		float3 brdf_f = Fresnel(f0, voH);
		//float g = GeometricVisibility(roughness, noV, noL, voH);
		float d = D_GGX(roughness, noH);
		float v = Vis_SmithJointApprox(roughness, noV, noL);
		// Microfacet specular = D*G*F / (4*NoL*NoV)
		float3 specIrrad = d*v*brdf_f;// vecMul(d*g*brdf_f / (4.f * noV), lightInIrrad);
		float3 diffIrrad = vecMul((make_float3(1.f, 1.f, 1.f) - brdf_f), Diffuse(diff, roughness, noV, noL, voH));//vecMul((make_float3(1.f, 1.f, 1.f) - brdf_f), diff / M_PI);
		return vecMul(lightInIrrad*noL, diffspec.y*specIrrad + diffspec.x*diffIrrad);
	}

	__device__ void  GetLightFromRandLightVertices(float3 pos, float3 norm, LightVertex* lightVertices, uint lightVerticesSize, curandState* randstate, float3& irrad, float3& irradDir, float& pathPotential)
	{
		//LightVertex dummy;
		//dummy.diff = make_float3(1.f, 1.f, 1.f);
		//dummy.irrad = make_float3(1.f, 0.f, 0.f);
		//dummy.pos = make_float3(0.f, 0.f, 0.f);
		//dummy.norm = dummy.irradDir = normalize(pos - dummy.pos);
		//dummy.roughness = 0.5f;
		//dummy.specular = 0.5f;
		//dummy.metallic = 0.f;

		irrad = make_float3(0.f, 0.f, 0.f);
		uint lightVert = curand_uniform(randstate) * lightVerticesSize;
		LightVertex* lightVertex = &lightVertices[lightVert];
		float3 toLightVertexDir = normalize(lightVertex->pos - pos);
		float toLightVertexDist = length(lightVertex->pos - pos);

		CURay toLightVertex(pos, toLightVertexDir);
		TracePrimitiveResult traceResult;
		if (length(lightVertex->irrad) > 0.f && vecDot(norm, toLightVertexDir) > 0.f &&
			!TracePrimitive(toLightVertex, traceResult, toLightVertexDist - M_FLT_BIAS_EPSILON, M_FLT_BIAS_EPSILON, false))
		{
			if (toLightVertexDist > M_FLT_EPSILON)
			{
				irrad = GetShadingResult(-1 * toLightVertexDir, lightVertex->irradDir, lightVertex->irrad, lightVertex->norm
					, lightVertex->diff, lightVertex->metallic, lightVertex->roughness, lightVertex->specular, make_float2(1.f, 1.f)) + lightVertex->emissive;
				irradDir = toLightVertexDir;
			}
			else
			{
				irrad = lightVertex->irrad;
				irradDir = -1.f * lightVertex->irradDir;
			}
		}
	}


	__global__ void pt_connectEyeLightPath_kernel(PTPathVertex** eyeStream, uint eyeStreamSize, LightVertex* lightVertices, uint lightVerticesSize)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (ind >= eyeStreamSize) return;

		PTPathVertex* eyePath = eyeStream[ind];
		float3 lightFromLightVertex = make_float3(0.f, 0.f, 0.f);
		float3 toLightVertexDir = make_float3(0.f, 0.f, 0.f);
		float lightPathPotential = 1.f;
		GetLightFromRandLightVertices(eyePath->pathVertexPos + eyePath->origNorm * M_FLT_BIAS_EPSILON, eyePath->origNorm
			, lightVertices, lightVerticesSize, &eyePath->randState, lightFromLightVertex, toLightVertexDir, lightPathPotential);
		float3 lightContribFromLightVertex = vecMax(make_float3(0.f, 0.f, 0.f)
			, GetShadingResult(eyePath->pathInDir, toLightVertexDir, lightFromLightVertex, eyePath->origNorm
			, eyePath->origDiff, eyePath->origMetallic, eyePath->origRoughness, eyePath->origSpecular
			, make_float2(1.f - eyePath->origTrans, 1.f)));

		if (length(lightContribFromLightVertex) > 0.f)
		{
			eyePath->pathAccumSample = eyePath->pathAccumSample + vecMul(lightContribFromLightVertex, eyePath->pathInMulTerm);
			eyePath->pathSampleN += 4;
			eyePath->pathPotential *= lightPathPotential;
		}
	}

	__global__ void pt_assignPathStream_kernel(PTPathVertex** pathStream, uint pathStreamSize, PTPathVertex* pathQueue, uint pathQueueCur, uint pathQueueSize, uint assignableSlot)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (ind < assignableSlot)
		{
			int pathStreamInd = pathStreamSize + ind;
			int pathQueueInd = pathQueueCur + ind;
			PTPathVertex* assignSample = nullptr;
			if (pathQueueInd < pathQueueSize)
			{
				assignSample = &pathQueue[pathQueueInd];
			}
			pathStream[pathStreamInd] = assignSample;
		}
	}

	__global__ void pt_applyPixelProbToResult_kernel(uint width, uint height, float* result, float* varResult, float minProb = 0.f)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		uint ind = (y * width + x);
		result[ind * 3] = result[ind * 3 + 1] = result[ind * 3 + 2] = fmaxf(minProb, varResult[ind]);
	}

	__global__ void pt_debugTracedPathQueueResult_kernel(PTPathVertex* pathQueue, uint pathQueueSize, uint width, uint height
		, float* tempResult, uint* tempResultN)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;

		if (x >= pathQueueSize) return;

		// add calculating sample to the result
		if (!pathQueue[x].isTerminated)
		{
			pathQueue[x].pathAccumSample = pathQueue[x].pathAccumSample + pathQueue[x].pathSample;
			pathQueue[x].pathAccumPotential = pathQueue[x].pathAccumPotential + pathQueue[x].pathPotential;
			pathQueue[x].pathSampleN++;
		}

		if (pathQueue[x].pathSampleN > 0)
		{
			uint ind = pathQueue[x].pathPixel.y * width + pathQueue[x].pathPixel.x;

			uint tempNextSampleResultN = pathQueue[x].pathSampleN;

			float3 sampleResult = make_float3(1.f, 1.f, 1.f);
			float potentialResult = 1.f - pathQueue[x].pathAccumPotential;

			atomicAdd(tempResult + ind * 3, 1.f);
			atomicAdd(tempResult + ind * 3 + 1, 1.f);
			atomicAdd(tempResult + ind * 3 + 2, 1.f);
			atomicAdd(tempResultN + ind, 1);
		}
	}

	__global__ void pt_applyPathQueueResult_kernel(PTPathVertex* pathQueue, uint pathQueueSize, uint width, uint height, uint frameN, float* result, float* accResult
		, float* varResult, float* posResult, float* normResult, float* diffResult, uint* sampleResultN)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;

		if (x >= pathQueueSize) return;

		// add calculating sample to the result
		if (!pathQueue[x].isTerminated)
		{
			pathQueue[x].pathAccumSample = pathQueue[x].pathAccumSample + pathQueue[x].pathSample;
			pathQueue[x].pathAccumPotential = pathQueue[x].pathAccumPotential + pathQueue[x].pathPotential;
			pathQueue[x].pathSampleN++;
		}

		if (pathQueue[x].pathSampleN > 0)
		{
			uint ind = pathQueue[x].pathPixel.y * width + pathQueue[x].pathPixel.x;
			if (!frameN)
			{
				sampleResultN[ind] = 0;
			}
			uint tempNextSampleResultN = sampleResultN[ind] + pathQueue[x].pathSampleN;
			if (tempNextSampleResultN > sampleResultN[ind])
			{
				float3 sampleResult = pathQueue[x].pathAccumSample;
				if (!isinf(sampleResult.x) && !isinf(sampleResult.y) && !isinf(sampleResult.z))
				{
					float potentialResult = 1.f - pathQueue[x].pathAccumPotential;
					float resultInf = 1.f / (float)(tempNextSampleResultN);
					float oldInf = sampleResultN[ind] * resultInf;

					result[ind * 3] = max(resultInf * sampleResult.x + oldInf * result[ind * 3], 0.f);
					result[ind * 3 + 1] = max(resultInf * sampleResult.y + oldInf * result[ind * 3 + 1], 0.f);
					result[ind * 3 + 2] = max(resultInf * sampleResult.z + oldInf * result[ind * 3 + 2], 0.f);
					varResult[ind] = max(resultInf * potentialResult + oldInf * varResult[ind], 0.f);
					sampleResultN[ind] = tempNextSampleResultN;

					sampleResult = pathQueue[x].pathDirectPos * pathQueue[x].pathSampleN;
					posResult[ind * 3] = resultInf * sampleResult.x + oldInf *posResult[ind * 3];
					posResult[ind * 3 + 1] = resultInf * sampleResult.y + oldInf *posResult[ind * 3 + 1];
					posResult[ind * 3 + 2] = resultInf * sampleResult.z + oldInf *posResult[ind * 3 + 2];

					sampleResult = pathQueue[x].pathDirectNorm * pathQueue[x].pathSampleN;
					normResult[ind * 3] = resultInf * sampleResult.x + oldInf *normResult[ind * 3];
					normResult[ind * 3 + 1] = resultInf * sampleResult.y + oldInf *normResult[ind * 3 + 1];
					normResult[ind * 3 + 2] = resultInf * sampleResult.z + oldInf *normResult[ind * 3 + 2];

					sampleResult = pathQueue[x].pathDirectDiffuse * pathQueue[x].pathSampleN;
					diffResult[ind * 3] = resultInf * sampleResult.x + oldInf *diffResult[ind * 3];
					diffResult[ind * 3 + 1] = resultInf * sampleResult.y + oldInf *diffResult[ind * 3 + 1];
					diffResult[ind * 3 + 2] = resultInf * sampleResult.z + oldInf *diffResult[ind * 3 + 2];
				}
			}
		}
	}

	__global__ void pt_atomicApplyPathQueueResult_kernel(PTPathVertex* pathQueue, uint pathQueueSize, uint width, uint height
		, float* tempResult, uint* tempResultN, float* tempPos, float* tempNorm, float* tempDiff)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;

		if (x >= pathQueueSize) return;

		// add calculating sample to the result
		if (!pathQueue[x].isTerminated)
		{
			pathQueue[x].pathAccumSample = pathQueue[x].pathAccumSample + pathQueue[x].pathSample;
			pathQueue[x].pathAccumPotential = pathQueue[x].pathAccumPotential + pathQueue[x].pathPotential;
			pathQueue[x].pathSampleN++;
		}

		if (pathQueue[x].pathSampleN > 0)
		{
			uint ind = pathQueue[x].pathPixel.y * width + pathQueue[x].pathPixel.x;
			
			float3 sampleResult = pathQueue[x].pathAccumSample;
			if (!isinf(sampleResult.x) && !isinf(sampleResult.y) && !isinf(sampleResult.z))
			{
				atomicAdd(tempResult + ind * 3, sampleResult.x);
				atomicAdd(tempResult + ind * 3 + 1, sampleResult.y);
				atomicAdd(tempResult + ind * 3 + 2, sampleResult.z);

				atomicAdd(tempResultN + ind, pathQueue[x].pathSampleN);

				sampleResult = pathQueue[x].pathDirectPos * pathQueue[x].pathSampleN;
				atomicAdd(tempPos + ind * 3, sampleResult.x);
				atomicAdd(tempPos + ind * 3 + 1, sampleResult.y);
				atomicAdd(tempPos + ind * 3 + 2, sampleResult.z);

				sampleResult = pathQueue[x].pathDirectNorm * pathQueue[x].pathSampleN;
				atomicAdd(tempNorm + ind * 3, sampleResult.x);
				atomicAdd(tempNorm + ind * 3 + 1, sampleResult.y);
				atomicAdd(tempNorm + ind * 3 + 2, sampleResult.z);

				sampleResult = pathQueue[x].pathDirectDiffuse * pathQueue[x].pathSampleN;
				atomicAdd(tempDiff + ind * 3, sampleResult.x);
				atomicAdd(tempDiff + ind * 3 + 1, sampleResult.y);
				atomicAdd(tempDiff + ind * 3 + 2, sampleResult.z);
			}
		}
	}


	__global__ void pt_accumTempResultToResult_kernel(uint width, uint height, uint frameN, float* tempResult, uint* tempResultN
		, float* tempPos, float* tempNorm, float* tempDiff, float* result, uint* resultN
		, float* pos, float* norm, float* diff)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;

		if (x >= width * height) return;

		if (frameN == 0)
		{
			resultN[x] = 0;

			result[x * 3] = 0.f;
			result[x * 3 + 1] = 0.f;
			result[x * 3 + 2] = 0.f;

			pos[x * 3] = 0.f;
			pos[x * 3 + 1] = 0.f;
			pos[x * 3 + 2] = 0.f;

			norm[x * 3] = 0.f;
			norm[x * 3 + 1] = 0.f;
			norm[x * 3 + 2] = 0.f;

			diff[x * 3] = 0.f;
			diff[x * 3 + 1] = 0.f;
			diff[x * 3 + 2] = 0.f;
		}

		uint nextSampleN = tempResultN[x] + resultN[x];
		if (nextSampleN > resultN[x])
		{
			float resultInf = 1.f / (float)(nextSampleN);
			float oldInf = resultN[x] * resultInf;

			result[x * 3] = max(resultInf * tempResult[x * 3] + oldInf * result[x * 3], 0.f);
			result[x * 3 + 1] = max(resultInf * tempResult[x * 3 + 1] + oldInf * result[x * 3 + 1], 0.f);
			result[x * 3 + 2] = max(resultInf * tempResult[x * 3 + 2] + oldInf * result[x * 3 + 2], 0.f);

			resultN[x] = resultN[x] + tempResultN[x];

			pos[x * 3] = resultInf * tempPos[x * 3] + oldInf * pos[x * 3];
			pos[x * 3 + 1] = resultInf * tempPos[x * 3 + 1] + oldInf * pos[x * 3 + 1];
			pos[x * 3 + 2] = resultInf * tempPos[x * 3 + 2] + oldInf * pos[x * 3 + 2];

			norm[x * 3] = resultInf * tempNorm[x * 3] + oldInf * norm[x * 3];
			norm[x * 3 + 1] = resultInf * tempNorm[x * 3 + 1] + oldInf * norm[x * 3 + 1];
			norm[x * 3 + 2] = resultInf * tempNorm[x * 3 + 2] + oldInf * norm[x * 3 + 2];

			diff[x * 3] = resultInf * tempDiff[x * 3] + oldInf * diff[x * 3];
			diff[x * 3 + 1] = resultInf * tempDiff[x * 3 + 1] + oldInf * diff[x * 3 + 1];
			diff[x * 3 + 2] = resultInf * tempDiff[x * 3 + 2] + oldInf * diff[x * 3 + 2];
		}
	}

	__global__ void pt_calculateSquareError_kernel(float* correctData, float* sampleData, uint* sampleNData, uint sampleN, float* resultData, uint dataSize)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dataSize)
			return;
		uint sampledN = sampleNData[x];
		float reductionFactor = (float)sampleN / (float)(sampleN + sampledN);
		if (sampleN + sampledN < sampledN)
			reductionFactor = 0;
		resultData[x] = reductionFactor * ((correctData[x * 3] - sampleData[x * 3]) * (correctData[x * 3] - sampleData[x * 3])
			+ (correctData[x * 3 + 1] - sampleData[x * 3 + 1]) * (correctData[x * 3 + 1] - sampleData[x * 3 + 1])
			+ (correctData[x * 3 + 2] - sampleData[x * 3 + 2]) * (correctData[x * 3 + 2] - sampleData[x * 3 + 2])
			) / 3.f;
	}

	void CleanMem()
	{
		freeLightPathMem();
		freeStreamMem();
		freeAllBVHCudaMem();
		CUFREE(g_devTempResultN);
		CUFREE(g_devSampleResultN);
		CUFREE(g_devResultVarData);
		CUFREE(g_devResultData);
		CUFREE(g_devAccResultData);
		CUFREE(g_devTempResultData);
		CUFREE(g_devFilteredResult);
		CUFREE(g_devFilterGaussianConst);
		CUFREE(g_devPositionData);
		CUFREE(g_devNormalData);
		CUFREE(g_devDiffuseData);
		CUFREE(g_devTempPositionData);
		CUFREE(g_devTempNormalData);
		CUFREE(g_devTempDiffuseData);
	}

	//struct ray_greater_compare
	//{
	//	__hd__ bool operator()(const PTPathVertex* vert1, const PTPathVertex* vert2)
	//	{
	//		int vert1Score = (vert1->pathOutDir.x > 0) + (vert1->pathOutDir.y > 0) + (vert1->pathOutDir.z > 0);
	//		int vert2Score = (vert2->pathOutDir.x > 0) + (vert2->pathOutDir.y > 0) + (vert2->pathOutDir.z > 0);
	//		return vert1Score > vert2Score;
	//	}
	//};

	struct is_temppathqueue_terminated
	{
		__hd__ bool operator()(const uint& vert)
		{
			return (vert + 1 == 0);
		}
	};

	struct is_active
	{
		__hd__ bool operator()(const PTPathVertex* vert)
		{
			return !vert->isTerminated;
		}
	};

	struct is_terminated
	{
		__hd__ bool operator()(const PTPathVertex* vert)
		{
			return vert->isTerminated;
		}
	};

	struct is_connectToLightPath
	{
		__hd__ bool operator()(const PTPathVertex* vert)
		{
			return vert->pathType == RAYTYPE_DIFF;
		}
	};

	template<typename Predicate>
	__global__ void pt_copyif_kernel(PTPathVertex** result, uint* atomicN, PTPathVertex** pathStream
		, uint activePathStreamSize, Predicate pred)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (ind >= activePathStreamSize)
			return;
		if (pred(pathStream[ind]))
		{
			result[atomicAggInc((int *)atomicN)] = pathStream[ind];
		}
	}

	void TracePathQueue(uint pathQueueSize)
	{
		dim3 block1(BLOCK_SIZE*BLOCK_SIZE, 1, 1);
		dim3 block2(BLOCK_SIZE, BLOCK_SIZE, 1);

		uint activePathStreamSize = 0;
		g_uPathQueueCur = 0;
		while (g_uPathQueueCur < pathQueueSize || activePathStreamSize > 0)
		{
			uint tempActivePathStreamSize = activePathStreamSize;
			int assignableStreamSlot = min((uint)PATHSTREAM_SIZE - activePathStreamSize, pathQueueSize - g_uPathQueueCur);
			if (assignableStreamSlot > 0)
				pt_assignPathStream_kernel << < dim3(ceil((float)assignableStreamSlot / (float)block1.x), 1, 1), block1 >> >
				(g_devPathStream, activePathStreamSize, g_devPathQueue, g_uPathQueueCur
				, pathQueueSize, assignableStreamSlot);

			//readjust activePathStreamSize
			activePathStreamSize += assignableStreamSlot;
			g_uPathQueueCur += assignableStreamSlot;

			//tracing process
			pt_traceSample_kernel << < dim3(ceil((float)activePathStreamSize / (float)block1.x), 1, 1), block1 >> > 
				(g_devVertices, g_devTriangles, g_devMaterials, g_devTextures, g_devPathStream, activePathStreamSize);
			HANDLE_KERNEL_ERROR();

			//compact pathstream and find activePathStreamSize value
			//PTPathVertex** compactedStreamEndItr = thrust::remove_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize, is_terminated());
			//activePathStreamSize = compactedStreamEndItr - g_devPathStream;

			HANDLE_ERROR(cudaMemset((void*)g_devAtomicN, 0, sizeof(uint)));
			pt_copyif_kernel << < dim3(ceil((float)activePathStreamSize / (float)block1.x), 1, 1), block1 >> >
				(g_devTempPathStream, g_devAtomicN, g_devPathStream, activePathStreamSize, is_active());
			HANDLE_KERNEL_ERROR();
			cudaMemcpy(&activePathStreamSize, g_devAtomicN, sizeof(uint), cudaMemcpyDeviceToHost);
			PTPathVertex** tempSwap = g_devPathStream;
			g_devPathStream = g_devTempPathStream;
			g_devTempPathStream = tempSwap;

			//gen connectionpathstream
			//PTPathVertex** conPathStreamEndItr = thrust::copy_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize
			//	, g_devEyeLightConPathStream, is_connectToLightPath());
			//uint activeConPathStreamSize = conPathStreamEndItr - g_devEyeLightConPathStream;

			uint activeConPathStreamSize = 0;
			if (activePathStreamSize > 0)
			{
				HANDLE_ERROR(cudaMemset((void*)g_devAtomicN, 0, sizeof(uint)));
				pt_copyif_kernel << < dim3(ceil((float)activePathStreamSize / (float)block1.x), 1, 1), block1 >> >
					(g_devEyeLightConPathStream, g_devAtomicN, g_devPathStream, activePathStreamSize, is_connectToLightPath());
				HANDLE_KERNEL_ERROR();
				cudaMemcpy(&activeConPathStreamSize, g_devAtomicN, sizeof(uint), cudaMemcpyDeviceToHost);
			}

			//connect eye and light path stream
			if (activeConPathStreamSize > 0)
			{
				pt_connectEyeLightPath_kernel << < dim3(ceil((float)activeConPathStreamSize / (float)block1.x), 1, 1), block1 >> >
					(g_devEyeLightConPathStream, activeConPathStreamSize, g_devLightVertices, g_uLightVerticesSize);
				HANDLE_KERNEL_ERROR();
			}

		}
	}

	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene* scene
		, float width, float height, float* result)
	{
		// Check and allocate everything
		if (!scene || !scene->GetCompactBVH()->IsValid())
			return false;

		NPMathHelper::Vec3 camRight = camDir.cross(camUp).normalize();
		camUp = camRight.cross(camDir).normalize();

		g_matLastCamMat = g_matCurCamMat;
		g_matCurCamMat = NPMathHelper::Mat4x4::lookAt(camPos, camPos + camDir, camUp);
		g_uCurFrameN = (g_matLastCamMat != g_matCurCamMat) ? 0 : g_uCurFrameN + 1;

		if (!g_bIsCudaInit || scene->GetIsCudaDirty())
		{
			CleanMem();
			g_matLastCamMat = g_matCurCamMat;
			g_uCurFrameN = 0;
			initAllSceneCudaMem(scene);
			allocateStreamMem(width * height);
			allocateLightPathMem();
			updateLightTriCudaMem(scene);

			size_t mem_tot;
			size_t mem_free;
			cudaMemGetInfo(&mem_free, &mem_tot);
			std::cout << "Memory Used : " << mem_tot - mem_free << "/" << mem_tot << " -> Free " << mem_free << std::endl;
		}
		else if (scene->GetIsCudaMaterialDirty())
		{
			updateAllSceneMaterialsCudaMem(scene);
			updateLightTriCudaMem(scene);
			g_uCurFrameN = 0;
		}

		if (!g_bIsCudaInit)
			return false;

		if (!g_devResultData || !g_devAccResultData || !g_devTempResultData || g_resultDataSize != (sizeof(float) * 3 * width * height) || !g_devFilteredResult)
		{
			g_resultDataSize = sizeof(float) * 3 * width * height;
			CUFREE(g_devResultData);
			cudaMalloc((void**)&g_devResultData, g_resultDataSize);
			CUFREE(g_devAccResultData);
			cudaMalloc((void**)&g_devAccResultData, g_resultDataSize);
			CUFREE(g_devTempResultData);
			cudaMalloc((void**)&g_devTempResultData, g_resultDataSize);
			CUFREE(g_devPositionData);
			cudaMalloc((void**)&g_devPositionData, g_resultDataSize);
			CUFREE(g_devNormalData);
			cudaMalloc((void**)&g_devNormalData, g_resultDataSize);
			CUFREE(g_devDiffuseData);
			cudaMalloc((void**)&g_devDiffuseData, g_resultDataSize);
			CUFREE(g_devTempPositionData);
			cudaMalloc((void**)&g_devTempPositionData, g_resultDataSize);
			CUFREE(g_devTempNormalData);
			cudaMalloc((void**)&g_devTempNormalData, g_resultDataSize);
			CUFREE(g_devTempDiffuseData);
			cudaMalloc((void**)&g_devTempDiffuseData, g_resultDataSize);
			CUFREE(g_devResultVarData);
			cudaMalloc((void**)&g_devResultVarData, sizeof(float) * width * height); 
			CUFREE(g_devSampleResultN);
			cudaMalloc((void**)&g_devSampleResultN, sizeof(uint) * width * height);
			CUFREE(g_devTempResultN);
			cudaMalloc((void**)&g_devTempResultN, sizeof(uint) * width * height);
			CUFREE(g_devFilteredResult);
			cudaMalloc((void**)&g_devFilteredResult, g_resultDataSize);
		}

		if (!g_devFilterGaussianConst)
		{
			CUFREE(g_devFilterGaussianConst);
			cudaMalloc((void**)&g_devFilterGaussianConst, sizeof(uint) * GAUSSIANCOST_N);
			cudaBilateralFilter::updateGaussian(g_devFilterGaussianConst, *g_fFilterColorEuD.GetFloat(), *g_uiFilterRadius.GetUint());
		}

		float3 f3CamPos = V32F3(camPos);
		float3 f3CamUp = V32F3(camUp);
		float3 f3CamDir = V32F3(camDir);
		float3 f3CamRight = V32F3(camRight);
		dim3 block1(BLOCK_SIZE*BLOCK_SIZE, 1, 1);
		dim3 block2(BLOCK_SIZE, BLOCK_SIZE, 1);
		dim3 renderGrid(ceil(width / (float)block2.x), ceil(height / (float)block2.y), 1);

		// light paths
		if (g_uCurFrameN % 3 == 0)
		{
			uint lightPathStreamSizeCap = min((uint)PATHSTREAM_SIZE, (uint)(LIGHTVERTEX_N / LIGHTRAY_BOUND_MAX));
			pt_genLightPathQueue_kernel << < dim3(ceil((float)lightPathStreamSizeCap / (float)block1.x), 1, 1), block1 >> >
				(g_uCurFrameN, WangHash(g_uCurFrameN), g_devLightTri, g_lightTriN, g_devVertices, g_devTriangles, g_devMaterials, g_devTextures, g_devPathQueue, lightPathStreamSizeCap
				, g_devLightVertices, 0);

			uint activePathStreamSize = 0;
			g_uLightVerticesSize = lightPathStreamSizeCap;
			g_uPathQueueCur = 0;
			while (g_uPathQueueCur < lightPathStreamSizeCap || activePathStreamSize > 0)
			{
				uint tempActivePathStreamSize = activePathStreamSize;
				int assignableStreamSlot = min(lightPathStreamSizeCap - activePathStreamSize, lightPathStreamSizeCap - g_uPathQueueCur);
				if (assignableStreamSlot > 0)
				{
					pt_assignPathStream_kernel << < dim3(ceil((float)assignableStreamSlot / (float)block1.x), 1, 1), block1 >> >(g_devPathStream, activePathStreamSize, g_devPathQueue, g_uPathQueueCur
						, g_uLightVerticesSize, assignableStreamSlot);
				}
				//readjust activePathStreamSize
				activePathStreamSize += assignableStreamSlot;
				g_uPathQueueCur += assignableStreamSlot;

				pt_traceLight_kernel << < dim3(ceil((float)activePathStreamSize / (float)block1.x), 1, 1), block1 >> > (g_devVertices, g_devTriangles, g_devMaterials, g_devTextures, g_devPathStream, activePathStreamSize
					, g_devLightVertices, g_uLightVerticesSize);
				g_uLightVerticesSize += activePathStreamSize;
				//compact pathstream and find activePathStreamSize value
				//PTPathVertex** compactedStreamEndItr = thrust::remove_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize, is_terminated());
				//activePathStreamSize = compactedStreamEndItr - g_devPathStream;
				HANDLE_ERROR(cudaMemset((void*)g_devAtomicN, 0, sizeof(uint)));
				pt_copyif_kernel << < dim3(ceil((float)activePathStreamSize / (float)block1.x), 1, 1), block1 >> >
					(g_devTempPathStream, g_devAtomicN, g_devPathStream, activePathStreamSize, is_active());
				HANDLE_KERNEL_ERROR();
				cudaMemcpy(&activePathStreamSize, g_devAtomicN, sizeof(uint), cudaMemcpyDeviceToHost);
				PTPathVertex** tempSwap = g_devPathStream;
				g_devPathStream = g_devTempPathStream;
				g_devTempPathStream = tempSwap;
			}
			//std::cout << "Generated light vertices size: " << g_uLightVerticesSize << std::endl;
		}

		if (g_uCurFrameN < 5)
		{
			//float time;
			//cudaEvent_t start, stop;
			//HANDLE_ERROR(cudaEventCreate(&start));
			//HANDLE_ERROR(cudaEventCreate(&stop));

			uint useQueueSize = width * height;

			//HANDLE_ERROR(cudaEventRecord(start, 0));
			// eye paths
			pt_genPathQueue_kernel << < renderGrid, block2 >> > (f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height
				, g_uCurFrameN, WangHash(g_uCurFrameN), g_devPathQueue);
			//HANDLE_ERROR(cudaEventRecord(stop, 0));
			//HANDLE_ERROR(cudaEventSynchronize(stop));
			//HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			//std::cout << "gen path: " << time << std::endl;

			//HANDLE_ERROR(cudaEventRecord(start, 0));
			// trace path queue
			TracePathQueue(useQueueSize);
			//HANDLE_ERROR(cudaEventRecord(stop, 0));
			//HANDLE_ERROR(cudaEventSynchronize(stop));
			//HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			//std::cout << "trace path: " << time << std::endl;

			//HANDLE_ERROR(cudaEventRecord(start, 0));
			pt_applyPathQueueResult_kernel << < dim3(ceil((float)useQueueSize / (float)block1.x), 1, 1), block1 >> >
				(g_devPathQueue, useQueueSize, width, height, g_uCurFrameN, g_devResultData, g_devAccResultData, g_devResultVarData
				, g_devPositionData, g_devNormalData, g_devDiffuseData, g_devSampleResultN);
			//HANDLE_ERROR(cudaEventRecord(stop, 0));
			//HANDLE_ERROR(cudaEventSynchronize(stop));
			//HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			//std::cout << "accum path: " << time << std::endl;
		}
		else
		{
#ifdef PERFBREAKDOWN
			float time;
			cudaEvent_t start, stop;
			HANDLE_ERROR(cudaEventCreate(&start));
			HANDLE_ERROR(cudaEventCreate(&stop));
			HANDLE_ERROR(cudaEventRecord(start, 0));
#endif
			if (*g_bFilterDiffuse.GetBool())
			{
				cudaBilateralFilter::bilaterial_posnormemit_kernel << < renderGrid, block2 >> >
					(g_devResultData, g_devPositionData, g_devNormalData, g_devDiffuseData,
					width, height, *g_fFilterColorEuD.GetFloat(), *g_fFilterPosEuD.GetFloat(), *g_fFilterNormEuD.GetFloat()
					, *g_fFilterDiffEuD.GetFloat(), *g_uiFilterRadius.GetUint(), g_devFilterGaussianConst, g_devFilteredResult);
			}
			else
			{
				cudaBilateralFilter::bilaterial_posnorm_kernel << < renderGrid, block2 >> >
					(g_devResultData, g_devPositionData, g_devNormalData,
					width, height, *g_fFilterColorEuD.GetFloat(), *g_fFilterPosEuD.GetFloat(), *g_fFilterNormEuD.GetFloat()
					, *g_uiFilterRadius.GetUint(), g_devFilterGaussianConst, g_devFilteredResult);
			}

#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(stop, 0));
			HANDLE_ERROR(cudaEventSynchronize(stop));
			HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			g_fAvgProcessTimeA.attrFloat = (g_fAvgProcessTimeA.attrFloat * (g_uCurFrameN - 5) + time) / (g_uCurFrameN - 5 + 1);
			//std::cout << "filter: " << time << std::endl;
			HANDLE_ERROR(cudaEventRecord(start, 0));
#endif

			// calculate sampling map from converged result
			pt_calculateSquareError_kernel << < dim3(ceil((float)(width * height) / (float)block1.x), 1, 1), block1 >> >
				(g_devFilteredResult, g_devResultData, g_devSampleResultN, *g_uiDesiredSamplingN.GetUint(), g_devResultVarData, (uint)(width * height));
			//thrust::sort(thrust::device, g_devResultVarData, g_devResultVarData + (uint)(width * height));
			float sumMSE = thrust::reduce(thrust::device, g_devResultVarData, g_devResultVarData + (uint)(width * height), 0.f, thrust::plus<float>());
			//float maxMSE = thrust::reduce(thrust::device, g_devResultVarData, g_devResultVarData + (uint)(width * height), 0.f, thrust::maximum<float>());
			//float meanMSE = sumMSE / (width * height);
			//std::cout << "sumMSE: " << sumMSE << "\n";
			//std::cout << "maxMSE: " << maxMSE << "\n";
			//std::cout << "meanMSE: " << meanMSE << "\n";

#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(stop, 0));
			HANDLE_ERROR(cudaEventSynchronize(stop));
			HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			g_fAvgProcessTimeB.attrFloat = (g_fAvgProcessTimeB.attrFloat * (g_uCurFrameN - 5) + time) / (g_uCurFrameN - 5 + 1);
			//std::cout << "MSE cal: " << time << std::endl;
#endif

			//if (g_uCurFrameN == 1)
			//{
			//	float* tempDiffData = new float[(uint)width * (uint)height];
			//	cudaMemcpy(tempDiffData, g_devResultVarData, (uint)(width * height) * sizeof(float), cudaMemcpyDeviceToHost);
			//	NPConfFileHelper::txtConfFile conf("adapCheat_diffData.txt");
			//	for (uint j = 0; j < width * height; j++)
			//	{
			//		conf.WriteRaw<float>(tempDiffData[j]);
			//		conf.WriteRaw("\n");
			//	}
			//	conf.SyncDataToFile();
			//	DELETE_ARRAY(tempDiffData);
			//}

#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(start, 0));
#endif
			// gen adaptive eye paths
			uint useQueueSize = width * height * PATHQUEUE_MUL;
			HANDLE_ERROR(cudaMemset((void*)g_devAtomicN, 0, sizeof(uint)));

			pt_genAdapPathQueue_kernel << < dim3(ceil((float)(width * height) / (float)block1.x), 1, 1), block1 >> >
				(f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height
				, g_uCurFrameN, WangHash(g_uCurFrameN), g_devAtomicN, g_devResultVarData, sumMSE, g_devPathQueue, useQueueSize);
			cudaMemcpy(&useQueueSize, g_devAtomicN, sizeof(uint), cudaMemcpyDeviceToHost);
			//std::cout << "AtomicN : " << useQueueSize << std::endl;
#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(stop, 0));
			HANDLE_ERROR(cudaEventSynchronize(stop));
			HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			g_fAvgProcessTimeC.attrFloat = (g_fAvgProcessTimeC.attrFloat * (g_uCurFrameN - 5) + time) / (g_uCurFrameN - 5 + 1);
			//std::cout << "gen path: " << time << std::endl;
#endif


#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(start, 0));
#endif
			TracePathQueue(useQueueSize);
#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(stop, 0));
			HANDLE_ERROR(cudaEventSynchronize(stop));
			HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			g_fAvgProcessTimeD.attrFloat = (g_fAvgProcessTimeD.attrFloat * (g_uCurFrameN - 5) + time) / (g_uCurFrameN - 5 + 1);
			//std::cout << "trace path: " << time << std::endl;
#endif

#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(start, 0));
#endif
			HANDLE_ERROR(cudaMemset((void*)g_devTempResultData, 0, sizeof(float) * 3 * height * width));
			HANDLE_ERROR(cudaMemset((void*)g_devTempPositionData, 0, sizeof(float) * 3 * height * width));
			HANDLE_ERROR(cudaMemset((void*)g_devTempNormalData, 0, sizeof(float) * 3 * height * width));
			HANDLE_ERROR(cudaMemset((void*)g_devTempDiffuseData, 0, sizeof(float) * 3 * height * width));
			HANDLE_ERROR(cudaMemset((void*)g_devTempResultN, 0, sizeof(uint) * height * width));
			if (*g_enumDebugMode.GetUint() == 1)
			{
				pt_debugTracedPathQueueResult_kernel << < dim3(ceil((float)useQueueSize / (float)block1.x), 1, 1), block1 >> >
					(g_devPathQueue, useQueueSize, width, height, g_devTempResultData, g_devTempResultN);
			}
			else
			{
				pt_atomicApplyPathQueueResult_kernel << < dim3(ceil((float)useQueueSize / (float)block1.x), 1, 1), block1 >> >
					(g_devPathQueue, useQueueSize, width, height, g_devTempResultData, g_devTempResultN
					, g_devTempPositionData, g_devTempNormalData, g_devTempDiffuseData);
			}

			pt_accumTempResultToResult_kernel << < dim3(ceil((float)(width * height) / (float)block1.x), 1, 1), block1 >> >
				(width, height, g_uCurFrameN, g_devTempResultData, g_devTempResultN
				, g_devTempPositionData, g_devTempNormalData, g_devTempDiffuseData
				, g_devResultData, g_devSampleResultN, g_devPositionData, g_devNormalData, g_devDiffuseData);
#ifdef PERFBREAKDOWN
			HANDLE_ERROR(cudaEventRecord(stop, 0));
			HANDLE_ERROR(cudaEventSynchronize(stop));
			HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
			g_fAvgProcessTimeE.attrFloat = (g_fAvgProcessTimeE.attrFloat * (g_uCurFrameN - 5) + time) / (g_uCurFrameN - 5 + 1);
			//std::cout << "accum path: " << time << std::endl;
#endif

		}
		if (*g_enumDebugMode.GetUint() == 2 || *g_enumDebugMode.GetUint() == 3)
		{
			pt_applyPixelProbToResult_kernel << < renderGrid, block2 >> >(width, height, g_devFilteredResult, g_devResultVarData, (*g_enumDebugMode.GetUint() == 3) ? *g_fMinTraceProb.GetFloat() : 0.f);
		}

		// Copy result to host
		if (*g_enumDebugMode.GetUint() == 4 || *g_enumDebugMode.GetUint() == 2 || *g_enumDebugMode.GetUint() == 3)
		{
			cudaMemcpy(result, g_devFilteredResult, g_resultDataSize, cudaMemcpyDeviceToHost);
		}
		else
		{
			cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
		}
		return true;
	}
}