#include "cudaRTCommon.h"

#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 16
#define NORMALRAY_BOUND_MAX 5
#define PATHSTREAM_SIZE 1E4*64

#define LIGHTRAY_BOUND_MAX 5
#define LIGHTVERTEX_N 640

namespace cudaRTBDPTStream
{
	CUDA_RT_COMMON_ATTRIBS_N(0)
	CUDA_RT_COMMON_ATTRIBS_BGN
	CUDA_RT_COMMON_ATTRIBS_END

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

		__hd__ LightVertex()
		{
			pos = norm = irrad = irradDir = make_float3(0.f, 0.f, 0.f);
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
		{}
	};

	PTPathVertex* g_devPathQueue = nullptr;
	uint g_uPathQueueCur = 0;
	uint g_uPathQueueSize = 0;
	PTPathVertex** g_devPathStream = nullptr;
	PTPathVertex** g_devEyeLightConPathStream = nullptr;
	uint g_uPathStreamSize = PATHSTREAM_SIZE;

	void freeStreamMem()
	{
		g_uPathQueueCur = g_uPathQueueSize = 0;
		CUFREE(g_devPathQueue);
		CUFREE(g_devPathStream);
		CUFREE(g_devEyeLightConPathStream);
	}

	void allocateStreamMem(uint queueSize = 480000)
	{
		g_uPathQueueSize = queueSize;
		HANDLE_ERROR(cudaMalloc((void**)&g_devPathQueue, sizeof(PTPathVertex) * g_uPathQueueSize));
		HANDLE_ERROR(cudaMemset((void*)g_devPathQueue, 0, sizeof(PTPathVertex) * g_uPathQueueSize));

		HANDLE_ERROR(cudaMalloc((void**)&g_devPathStream, sizeof(PTPathVertex*) * g_uPathStreamSize));
		HANDLE_ERROR(cudaMemset((void*)g_devPathStream, 0, sizeof(PTPathVertex*) * g_uPathStreamSize));

		HANDLE_ERROR(cudaMalloc((void**)&g_devEyeLightConPathStream, sizeof(PTPathVertex*) * g_uPathStreamSize));
		HANDLE_ERROR(cudaMemset((void*)g_devEyeLightConPathStream, 0, sizeof(PTPathVertex*) * g_uPathStreamSize));
	}

#pragma region SHADING_FUNC
	__device__ float3 Diffuse_Lambert(float3 DiffuseColor)
	{
		return DiffuseColor * (1 / M_PI);
	}

	__device__ float Vis_SmithJointApprox(float Roughness, float NoV, float NoL)
	{
		float a = Roughness * Roughness;
		float Vis_SmithV = NoL * (NoV * (1 - a) + a);
		float Vis_SmithL = NoV * (NoL * (1 - a) + a);
		return 0.5 * rcpf(Vis_SmithV + Vis_SmithL);
		//float k = (Roughness * Roughness) / 2.0f; // (Roughness + 1) * (Roughness + 1) / 8.f;
		//return (NoV / (NoV * (1 - k) + k))*(NoL / (NoL * (1 - k) + k));
	}

	__device__ float D_GGX(float Roughness, float NoH)
	{
		float m = Roughness * Roughness;
		float m2 = m*m;
		float d = (NoH * m2 - NoH) * NoH + 1;
		return m2 / (M_PI*d*d);
	}

	__device__ float3 F_Schlick(float3 SpecularColor, float VoH)
	{
		float Fc = pow(1 - VoH, 5);
		float firstTerm = saturate(50.0 * SpecularColor.z) * Fc;
		return make_float3(firstTerm, firstTerm, firstTerm) + (1 - Fc) * SpecularColor;
	}

	__device__ float3 ImportanceSampleGGX(float2 Xi, float Roughness, float3 N)
	{
		float a = Roughness * Roughness;
		float Phi = 2 * M_PI * Xi.x;
		float CosTheta = sqrt((1 - Xi.y) / (1 + (a*a - 1) * Xi.y));
		float SinTheta = sqrt(1 - CosTheta * CosTheta);
		float3 H;
		H.x = SinTheta * cos(Phi);
		H.y = SinTheta * sin(Phi);
		H.z = CosTheta;
		//float3 UpVector = abs(N.z) < 0.999 ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
		//float3 TangentX = normalize(vecCross(UpVector, N));
		//float3 TangentY = vecCross(N, TangentX);

		float3 w = N;
		float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = vecCross(w, u);
		u = vecCross(v, w);

		// Tangent to world space
		return (u * H.x + v * H.y + w * H.z);
	}

	__device__ float3 Diffuse(float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH)
	{
		return Diffuse_Lambert(DiffuseColor);
	}

	__device__ float Distribution(float Roughness, float NoH)
	{
		return D_GGX(Roughness, NoH);
	}

	__device__ float GeometricVisibility(float Roughness, float NoV, float NoL, float VoH)
	{
		return Vis_SmithJointApprox(Roughness, NoV, NoL);
	}

	__device__ float3 Fresnel(float3 SpecularColor, float VoH)
	{
		return F_Schlick(SpecularColor, VoH);
	}

#pragma endregion SHADING_FUNC

	float* g_devResultData = nullptr;
	float* g_devAccResultData = nullptr;

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
			{
				// Get some random microfacet
				float3 hDir = ImportanceSampleGGX(make_float2(curand_uniform(&procVertex->randState), curand_uniform(&procVertex->randState)), roughness, nl);

				// Calculate flesnel
				float voH = vecDot(-1 * ray.dir, hDir);
				float3 f0 = vecLerp(0.08 * make_float3(specular, specular, specular), diff, metallic);
				float3 brdf_f = Fresnel(f0, voH);

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
					if (cos2t < 0.f)
					{
						refrProb = 0.f;
					}
					else
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
				if (reflProb > 0 && curand_uniform(&procVertex->randState) < reflProb)
				{
					nextRay = CURay(ray.orig + (traceResult.dist - M_FLT_BIAS_EPSILON) * ray.dir, reflDir);
					// ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

					// Microfacet specular = D*G*F / (4*NoL*NoV)
					// pdf = D * NoH / (4 * VoH)
					// (G * F * VoH) / (NoV * NoH)
					float VoH = vecDot(-1 * ray.dir, hDir);
					float NoV = vecDot(nl, -1 * ray.dir);
					float NoH = vecDot(nl, hDir);
					float NoL = vecDot(nl, reflDir);
					float G = GeometricVisibility(roughness, NoV, NoL, VoH);
					//shadeResult = vecMul((brdf_f * G * VoH) / (NoV * NoH * reflProb) , nextRayResult.light) + emissive;
					lightMulTerm = (brdf_f * G * VoH) / (NoV * NoH * reflProb);
					nextRayType = RAYTYPE_SPEC;
				}

				// Diffused or Transmited
				else
				{
					// Transmited
					if (refrProb > 0 && curand_uniform(&procVertex->randState) < refrProb)
					{
						nextRay = CURay(ray.orig + (traceResult.dist + M_FLT_BIAS_EPSILON) * ray.dir + refrDir * M_FLT_BIAS_EPSILON, refrDir);
						//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
						float cosine = vecDot(-1 * nl, refrDir);
						//shadeResult = (cosine * vecMul(diff, nextRayResult.light)) / (refrProb * (1 - reflProb)) + emissive;
						lightMulTerm = cosine * diff / (refrProb * (1 - reflProb));
						nextRayType = RAYTYPE_SPEC;
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
					}
				}

				procVertex->pathSample = emissive + vecMul(procVertex->pathSample, lightMulTerm);

				float pixelContrib = length(procVertex->pathOutMulTerm) * length(lightMulTerm);

				if ((procVertex->pathType == RAYTYPE_DIFF && nextRayType == RAYTYPE_SPEC) || length(emissive) > 0.f)
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
					if (cos2t < 0.f)
					{
						refrProb = 0.f;
					}
					else
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
				if (reflProb > 0 && curand_uniform(&procVertex->randState) < reflProb)
				{
					nextRay = CURay(ray.orig + (traceResult.dist - M_FLT_BIAS_EPSILON) * ray.dir, reflDir);
					// ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

					// Microfacet specular = D*G*F / (4*NoL*NoV)
					// pdf = D * NoH / (4 * VoH)
					// (G * F * VoH) / (NoV * NoH)
					float VoH = vecDot(-1 * ray.dir, hDir);
					float NoV = vecDot(nl, -1 * ray.dir);
					float NoH = vecDot(nl, hDir);
					float NoL = vecDot(nl, reflDir);
					float G = GeometricVisibility(roughness, NoV, NoL, VoH);
					//shadeResult = vecMul((brdf_f * G * VoH) / (NoV * NoH * reflProb) , nextRayResult.light) + emissive;
					lightMulTerm = (brdf_f * G * VoH) / (NoV * NoH * reflProb);
					nextRayType = RAYTYPE_SPEC;
				}

				// Diffused or Transmited
				else
				{
					// Transmited
					if (refrProb > 0 && curand_uniform(&procVertex->randState) < refrProb)
					{
						nextRay = CURay(ray.orig + (traceResult.dist + M_FLT_BIAS_EPSILON) * ray.dir + refrDir * M_FLT_BIAS_EPSILON, refrDir);
						//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
						float cosine = vecDot(-1 * nl, refrDir);
						//shadeResult = (cosine * vecMul(diff, nextRayResult.light)) / (refrProb * (1 - reflProb)) + emissive;
						lightMulTerm = cosine * diff / (refrProb * (1 - reflProb));
						nextRayType = RAYTYPE_SPEC;
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
					}
				}

				procVertex->pathSample = procVertex->pathSample + vecMul(emissive , procVertex->pathOutMulTerm);

				procVertex->origDiff = diff;
				procVertex->pathInDir = -1 * ray.dir;
				procVertex->origNorm = nl;
				procVertex->origRoughness = roughness;
				procVertex->origMetallic = metallic;
				procVertex->origSpecular = specular;
				procVertex->origTrans = trans;
				procVertex->pathInMulTerm = procVertex->pathOutMulTerm;

				float pixelContrib = length(procVertex->pathOutMulTerm) * length(lightMulTerm);

				if ((procVertex->pathType == RAYTYPE_DIFF && nextRayType == RAYTYPE_SPEC) || length(emissive) > 0.f)
					pixelContrib = 0.f;

				if (curand_uniform(&procVertex->randState) > pixelContrib || procVertex->pathSampleDepth + 1 >= NORMALRAY_BOUND_MAX)
				{
					procVertex->pathAccumSample = procVertex->pathAccumSample + procVertex->pathSample;
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

		pathQueue[ind] = PTPathVertex(false, make_uint2(x,y), dir, camPos, RAYTYPE_EYE, randstate);
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

	__device__ void  GetLightFromRandLightVertices(float3 pos, float3 norm, LightVertex* lightVertices, uint lightVerticesSize, curandState* randstate, float3& irrad, float3& irradDir)
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
			if (length(lightVertex->irradDir) > M_FLT_EPSILON)
				irrad = GetShadingResult(-1 * toLightVertexDir, lightVertex->irradDir, lightVertex->irrad, lightVertex->norm
				, lightVertex->diff, lightVertex->metallic, lightVertex->roughness, lightVertex->specular, make_float2(1.f, 1.f)) + lightVertex->emissive;
			else
				irrad = lightVertex->irrad;
			irrad = irrad;
			irradDir = toLightVertexDir;
		}
	}


	__global__ void pt_connectEyeLightPath_kernel(PTPathVertex** eyeStream, uint eyeStreamSize, LightVertex* lightVertices, uint lightVerticesSize)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (ind >= eyeStreamSize) return;

		PTPathVertex* eyePath = eyeStream[ind];
		float3 lightFromLightVertex = make_float3(0.f, 0.f, 0.f);
		float3 toLightVertexDir = make_float3(0.f, 0.f, 0.f);
		GetLightFromRandLightVertices(eyePath->pathVertexPos + eyePath->origNorm * M_FLT_BIAS_EPSILON, eyePath->origNorm
			, lightVertices, lightVerticesSize, &eyePath->randState, lightFromLightVertex, toLightVertexDir);
		float3 lightContribFromLightVertex = vecMax(make_float3(0.f, 0.f, 0.f)
			, GetShadingResult(eyePath->pathInDir, toLightVertexDir, lightFromLightVertex, eyePath->origNorm
			, eyePath->origDiff, eyePath->origMetallic, eyePath->origRoughness, eyePath->origSpecular
			, make_float2(1.f - eyePath->origTrans, 1.f)));

		if (length(lightContribFromLightVertex) > 0.f)
		{
			eyePath->pathAccumSample = eyePath->pathAccumSample + vecMul(lightContribFromLightVertex, eyePath->pathInMulTerm);
			eyePath->pathSampleN += 4;
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

	__global__ void pt_applyPathQueueResult_kernel(PTPathVertex* pathQueue, uint pathQueueSize, uint width, uint height, uint frameN, float* result, float* accResult)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;

		if (x >= pathQueueSize) return;

		// add calculating sample to the result
		if (!pathQueue[x].isTerminated)
		{
			pathQueue[x].pathAccumSample = pathQueue[x].pathAccumSample + pathQueue[x].pathSample;
			pathQueue[x].pathSampleN++;
		}

		if (pathQueue[x].pathSampleN > 0)
		{
			float3 sampleResult = pathQueue[x].pathAccumSample / (float)pathQueue[x].pathSampleN;
			float resultInf = 1.f / (float)(frameN + 1);
			float oldInf = 1.f - resultInf;
			uint ind = pathQueue[x].pathPixel.y * width + pathQueue[x].pathPixel.x;
			result[ind * 3] = max(resultInf * sampleResult.x + oldInf * result[ind * 3], 0.f);
			result[ind * 3 + 1] = max(resultInf * sampleResult.y + oldInf * result[ind * 3 + 1], 0.f);
			result[ind * 3 + 2] = max(resultInf * sampleResult.z + oldInf * result[ind * 3 + 2], 0.f);
		}
	}

	void CleanMem()
	{
		freeLightPathMem();
		freeStreamMem();
		freeAllBVHCudaMem();
		CUFREE(g_devResultData);
		CUFREE(g_devAccResultData);
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
			g_matLastCamMat = g_matCurCamMat;
			g_uCurFrameN = 0;
			initAllSceneCudaMem(scene);
			allocateStreamMem(width * height);
			allocateLightPathMem();
			updateLightTriCudaMem(scene);

			size_t mem_tot;
			size_t mem_free;
			cudaMemGetInfo(&mem_free, &mem_tot);
			std::cout << "Memory Used : " << mem_tot-mem_free << "/" << mem_tot << " -> Free " << mem_free << std::endl;
		}
		else if (scene->GetIsCudaMaterialDirty())
		{
			updateAllSceneMaterialsCudaMem(scene);
			updateLightTriCudaMem(scene);
			g_uCurFrameN = 0;
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
				PTPathVertex** compactedStreamEndItr = thrust::remove_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize, is_terminated());
				activePathStreamSize = compactedStreamEndItr - g_devPathStream;
			}
			std::cout << "Generated light vertices size: " << g_uLightVerticesSize << std::endl;
		}

		// eye paths
		pt_genPathQueue_kernel << < renderGrid, block2 >> > (f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height
			, g_uCurFrameN, WangHash(g_uCurFrameN), g_devPathQueue);

		uint activePathStreamSize = 0;
		g_uPathQueueCur = 0;
		while (g_uPathQueueCur < g_uPathQueueSize || activePathStreamSize > 0)
		{
			uint tempActivePathStreamSize = activePathStreamSize;
			int assignableStreamSlot = min((uint)PATHSTREAM_SIZE - activePathStreamSize, g_uPathQueueSize - g_uPathQueueCur);
			if (assignableStreamSlot > 0)
				pt_assignPathStream_kernel << < dim3(ceil((float)assignableStreamSlot / (float)block1.x), 1, 1), block1 >> >(g_devPathStream, activePathStreamSize, g_devPathQueue, g_uPathQueueCur
				, g_uPathQueueSize, assignableStreamSlot);

			//readjust activePathStreamSize
			activePathStreamSize += assignableStreamSlot;
			g_uPathQueueCur += assignableStreamSlot;

			//tracing process
			pt_traceSample_kernel << < dim3(ceil((float)activePathStreamSize / (float)block1.x), 1, 1), block1 >> > (g_devVertices, g_devTriangles, g_devMaterials, g_devTextures, g_devPathStream, activePathStreamSize);

			//compact pathstream and find activePathStreamSize value
			PTPathVertex** compactedStreamEndItr = thrust::remove_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize, is_terminated());
			activePathStreamSize = compactedStreamEndItr - g_devPathStream;

			//gen connectionpathstream
			PTPathVertex** conPathStreamEndItr = thrust::copy_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize, g_devEyeLightConPathStream, is_connectToLightPath());
			uint activeConPathStreamSize = conPathStreamEndItr - g_devEyeLightConPathStream;

			//connect eye and light path stream
			if (activeConPathStreamSize > 0)
			{
				pt_connectEyeLightPath_kernel << < dim3(ceil((float)activeConPathStreamSize / (float)block1.x), 1, 1), block1 >> >
					(g_devEyeLightConPathStream, activeConPathStreamSize, g_devLightVertices, g_uLightVerticesSize);
			}

		}
		pt_applyPathQueueResult_kernel << < dim3(ceil((float)g_uPathQueueSize / (float)block1.x), 1, 1), block1 >> >(g_devPathQueue, g_uPathQueueSize, width, height, g_uCurFrameN, g_devResultData, g_devAccResultData);

		// Copy result to host
		cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
		return true;
	}
}