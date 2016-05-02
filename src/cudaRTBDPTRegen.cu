#include "cudaRTCommon.h"

#define BLOCK_SIZE 16
#define NORMALRAY_BOUND_MAX 5

#define LIGHTRAY_BOUND_MAX 5
#define LIGHTVERTEX_N 640

namespace cudaRTBDPTRegen
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
	uint* g_devLightTri = nullptr;
	uint g_lightTriN = 0;

	float* g_devResultData = nullptr;
	float* g_devAccResultData = nullptr;
	uint* g_devSampleResultN = nullptr;

	NPMathHelper::Mat4x4 g_matLastCamMat;
	NPMathHelper::Mat4x4 g_matCurCamMat;
	uint32 g_uCurFrameN;
	size_t g_resultDataSize = 0;

	struct LightTracerData
	{
		LightVertex* lightVertices;
		uint* lightTri;
		uint lightTriN;
		RTVertex* vertices;
		RTTriangle* triangles;
		RTMaterial* materials;
		CURTTexture* textures;
		curandState *randstate;

		__device__ LightTracerData(LightVertex* _lightVertices, uint* _lightTri, uint _lightTriN,
			RTVertex* _vertices, RTTriangle* _triangles, RTMaterial* _materials, CURTTexture* _textures, curandState* _randState)
			: lightVertices(_lightVertices)
			, lightTri(_lightTri)
			, lightTriN(_lightTriN)
			, vertices(_vertices)
			, triangles(_triangles)
			, materials(_materials)
			, textures(_textures)
			, randstate(_randState)
		{}
	};

	enum RAYTYPE
	{
		RAYTYPE_EYE = 0,
		RAYTYPE_DIFF = 1,
		RAYTYPE_SPEC = 2,
		RAYTYPE_LIGHT = 3
	};

	template <int depth = 0>
	__device__ void bdpt_light_ray(const CURay& ray, RAYTYPE rayType, float3 rayIrrad, uint lightSlot, LightTracerData& tracerData)
	{
		CURay nextRay = CURay();
		float3 lightMulTerm = make_float3(1.f, 1.f, 1.f);
		RAYTYPE nextRayType = RAYTYPE_LIGHT;

		LightVertex resultVertex = LightVertex();

		TracePrimitiveResult traceResult;

		//Regenerate ray
		CURay lightRay = ray;
		if (!lightRay.isValid || length(rayIrrad) < M_FLT_BIAS_EPSILON)
		{
			uint lightTri = tracerData.lightTriN * curand_uniform(tracerData.randstate);
			float lightW = curand_uniform(tracerData.randstate);
			float lightU = curand_uniform(tracerData.randstate);
			if (lightW + lightU > 1.0f)
			{
				lightW = 1.f - lightW;
				lightU = 1.f - lightU;
			}
			float lightV = 1.f - lightW - lightU;

			uint triId = tracerData.lightTri[lightTri];
			RTTriangle* tri = &tracerData.triangles[triId];
			RTMaterial* mat = &tracerData.materials[tri->matInd];
			RTVertex* v0 = &tracerData.vertices[tri->vertInd0];
			RTVertex* v1 = &tracerData.vertices[tri->vertInd1];
			RTVertex* v2 = &tracerData.vertices[tri->vertInd2];
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
			GetMaterialColors(mat, uv, tracerData.textures, diff, triNorm, emissive, trans, specular, metallic, roughness
				, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss);

			float3 w = triNorm;
			float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = vecCross(w, u);
			u = vecCross(v, w);

			float r1 = 2.f * M_PI * curand_uniform(tracerData.randstate);
			float r2cos = sqrtf(curand_uniform(tracerData.randstate));
			float r2sin = 1.f - r2cos*r2cos;
			float3 diffDir = normalize(w * r2cos + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

			nextRay = CURay(triPos + triNorm * M_FLT_BIAS_EPSILON, diffDir);
			rayIrrad = emissive;// *(float)tracerData.lightTriN;
			nextRayType = RAYTYPE_LIGHT;

			resultVertex.irrad = rayIrrad;
			resultVertex.irradDir = make_float3(0.f,0.f,0.f);
			resultVertex.norm = triNorm;
			resultVertex.pos = triPos;
			resultVertex.diff = diff;
			resultVertex.emissive = emissive;
			resultVertex.specular = specular;
			resultVertex.metallic = metallic;
			resultVertex.roughness = roughness;
		}
		else if (TracePrimitive(lightRay, traceResult, M_INF, M_FLT_BIAS_EPSILON, false))
		{
			RTTriangle* tri = &tracerData.triangles[traceResult.triId];
			RTMaterial* mat = &tracerData.materials[tri->matInd];
			RTVertex* v0 = &tracerData.vertices[tri->vertInd0];
			RTVertex* v1 = &tracerData.vertices[tri->vertInd1];
			RTVertex* v2 = &tracerData.vertices[tri->vertInd2];
			float2 uv0 = make_float2(v0->tex._x, v0->tex._y);
			float2 uv1 = make_float2(v1->tex._x, v1->tex._y);
			float2 uv2 = make_float2(v2->tex._x, v2->tex._y);
			float2 uv = uv0 * traceResult.w + uv1 * traceResult.u + uv2 * traceResult.v;
			float3 n0 = V32F3(v0->norm);
			float3 n1 = V32F3(v1->norm);
			float3 n2 = V32F3(v2->norm);
			float3 norm = n0 * traceResult.w + n1 * traceResult.u + n2 * traceResult.v;
			float3 pos = V32F3(v0->pos) * traceResult.w + V32F3(v1->pos) * traceResult.u + V32F3(v2->pos) * traceResult.v;

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
			GetMaterialColors(mat, uv, tracerData.textures, diff, norm, emissive, trans, specular, metallic, roughness
				, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss);
			float3 nl = vecDot(norm, lightRay.dir) < 0.f ? norm : -1 * norm;

			// Get some random microfacet
			float3 hDir = ImportanceSampleGGX(make_float2(curand_uniform(tracerData.randstate), curand_uniform(tracerData.randstate)), roughness, nl);

			// Calculate flesnel
			float voH = vecDot(-1 * lightRay.dir, hDir);
			float3 f0 = vecLerp(0.08 * make_float3(specular, specular, specular), diff, metallic);
			float3 brdf_f = Fresnel(f0, voH);

			// Reflected or Refracted
			float reflProb = lerp(length(brdf_f), 1.0f, metallic);
			float refrProb = trans;
			float3 reflDir;
			float3 refrDir;

			if (refrProb > 0)
			{
				bool into = vecDot(nl, norm) > 0.f;
				float nt = specular * 0.8f + 1.f;
				float nc = 1.0f;
				float nnt = into ? nc / nt : nt / nc;
				float ddn = vecDot(hDir, lightRay.dir);
				float cos2t = 1.f - nnt * nnt *(1.f - ddn * ddn);
				if (cos2t < 0.f)
				{
					reflProb = 1.0f;// refrProb = 0.f;
				}
				else
				{
					refrDir = normalize(lightRay.dir * nnt - hDir * (ddn*nnt + sqrtf(cos2t)));
				}
			}

			if (reflProb > 0)
			{
				reflDir = normalize(lightRay.dir - hDir * 2 * vecDot(hDir, lightRay.dir));
				if (vecDot(reflDir, nl) < 0.f)
					reflProb = 0.f;
			}

			// Reflected
			if (ProbabilityRand(tracerData.randstate, reflProb))
			{
				nextRay = CURay(lightRay.orig + (traceResult.dist - M_FLT_BIAS_EPSILON) * lightRay.dir, reflDir);
				// ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

				// Microfacet specular = D*G*F / (4*NoL*NoV)
				// pdf = D * NoH / (4 * VoH)
				// (G * F * VoH) / (NoV * NoH)
				float VoH = vecDot(-1 * lightRay.dir, hDir);
				float NoV = vecDot(nl, -1 * lightRay.dir);
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
				if (ProbabilityRand(tracerData.randstate, refrProb))
				{
					nextRay = CURay(lightRay.orig + (traceResult.dist + M_FLT_BIAS_EPSILON) * lightRay.dir + refrDir * M_FLT_BIAS_EPSILON, refrDir);
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

					float r1 = 2.f * M_PI * curand_uniform(tracerData.randstate);
					float r2cos = sqrtf(curand_uniform(tracerData.randstate));
					float r2sin = 1.f - r2cos*r2cos;
					float3 diffDir = normalize(w * r2cos + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

					nextRay = CURay(lightRay.orig + traceResult.dist * lightRay.dir + diffDir * M_FLT_BIAS_EPSILON, diffDir);
					//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

					float VoH = vecDot(-1 * lightRay.dir, hDir);
					float NoV = vecDot(nl, -1 * lightRay.dir);
					float NoL = vecDot(nl, diffDir);
					//shadeResult = (M_PI * vecMul(Diffuse(diff, roughness, NoV, NoL, VoH), nextRayResult.light)) / ((1 - refrProb) * (1 - reflProb)) + emissive;
					lightMulTerm = M_PI * Diffuse(diff, roughness, NoV, NoL, VoH) / ((1 - refrProb) * (1 - reflProb));
					nextRayType = RAYTYPE_DIFF;
				}
			}

			//if (rayType == RAYTYPE_DIFF && nextRayType == RAYTYPE_SPEC)
			//	lightMulTerm = make_float3(0.f,0.f,0.f);

			lightMulTerm = vecMul(lightMulTerm, rayIrrad);
			//if (curand_uniform(tracerData.randstate) >= length(lightMulTerm))
			//{
			//	nextRay = CURay();
			//}

			resultVertex.irrad = rayIrrad;
			resultVertex.irradDir = -1 * lightRay.dir;
			resultVertex.norm = norm;
			resultVertex.pos = pos;
			resultVertex.diff = diff;
			resultVertex.emissive = emissive;
			resultVertex.specular = specular;
			resultVertex.metallic = metallic;
			resultVertex.roughness = roughness;
		}

		tracerData.lightVertices[lightSlot] = resultVertex;
		bdpt_light_ray<depth + 1>(nextRay, nextRayType, lightMulTerm, lightSlot + 1, tracerData);
	}

	template <>
	__device__ void bdpt_light_ray<LIGHTRAY_BOUND_MAX>(const CURay& ray, RAYTYPE rayType, float3 rayIrrad, uint lightSlot, LightTracerData& tracerData)
	{ }

	__global__ void bdpt_light_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
		float width, float height, LightVertex* lightVertices, uint* lightTri, uint lightTriN, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures
		, uint32 frameN, uint32 hashedFrameN)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint result_slot = x * LIGHTRAY_BOUND_MAX;

		curandState randstate;
		curand_init(hashedFrameN + x, 0, 0, &randstate);

		LightTracerData tracerData(lightVertices, lightTri, lightTriN, vertices, triangles, materials, textures, &randstate);
		bdpt_light_ray(CURay(), RAYTYPE_LIGHT, make_float3(0.f,0.f,0.f), result_slot, tracerData);
	}

	struct TracerData
	{
		float2* winSize;
		float fov;
		float3* camOrig;
		float3* camU;
		float3* camR;
		float3* camD;
		LightVertex* lightVertices;
		RTVertex* vertices;
		RTTriangle* triangles;
		RTMaterial* materials;
		CURTTexture* textures;
		curandState *randstate;

		__device__ TracerData(float2* _winSize, float _fov, float3* _camOrig, float3* _camU, float3* _camR, float3* _camD
			, LightVertex* _lightVertices, RTVertex* _vertices, RTTriangle* _triangles, RTMaterial* _materials, CURTTexture* _textures, curandState* _randState)
			: winSize(_winSize)
			, fov(_fov)
			, camOrig(_camOrig)
			, camU(_camU)
			, camR(_camR)
			, camD(_camD)
			, lightVertices(_lightVertices)
			, vertices(_vertices)
			, triangles(_triangles)
			, materials(_materials)
			, textures(_textures)
			, randstate(_randState)
		{}
	};

	struct PTSample
	{
		uint2 pixel;
		float2 uv;
		float pixelContrib;
		uint pathDepth;
		uint sampleTime;
		float3 sampleResult;
		float3 accumResult;

		__device__ PTSample(uint2 _pixel, float2 _uv)
			: pixel(_pixel), uv(_uv), pixelContrib(1.0f), pathDepth(0), sampleTime(0), accumResult()
		{
			accumResult = sampleResult = make_float3(0.f, 0.f, 0.f);
		}
	};

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
		return vecMul(lightInIrrad*noL,diffspec.y*specIrrad + diffspec.x*diffIrrad);
	}

	__device__ void  GetLightFromRandLightVertices(float3 pos, float3 norm, TracerData& tracerData, float3& irrad, float3& irradDir)
	{
		//LightVertex dummy;
		//dummy.diff = make_float3(1.f, 1.f, 1.f);
		//dummy.irrad = make_float3(1.f, 0.f, 0.f);
		//dummy.pos = make_float3(0.f, 0.f, 0.f);
		//dummy.norm = dummy.irradDir = normalize(pos - dummy.pos);
		//dummy.roughness = 0.5f;
		//dummy.specular = 0.5f;
		//dummy.metallic = 0.f;

		irrad = make_float3(0.f,0.f,0.f);
		uint lightVert = curand_uniform(tracerData.randstate) * LIGHTVERTEX_N;
		LightVertex* lightVertex = &tracerData.lightVertices[lightVert];
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

	template <int depth = 0>
	__device__ void pt0_normalRay(const CURay& ray, RAYTYPE rayType, PTSample& sample, TracerData& tracerData)
	{
		TracePrimitiveResult traceResult;
		if (TracePrimitive(ray, traceResult, M_INF, M_FLT_BIAS_EPSILON, false))
		{
			RTTriangle* tri = &tracerData.triangles[traceResult.triId];
			RTMaterial* mat = &tracerData.materials[tri->matInd];
			RTVertex* v0 = &tracerData.vertices[tri->vertInd0];
			RTVertex* v1 = &tracerData.vertices[tri->vertInd1];
			RTVertex* v2 = &tracerData.vertices[tri->vertInd2];
			float2 uv0 = make_float2(v0->tex._x, v0->tex._y);
			float2 uv1 = make_float2(v1->tex._x, v1->tex._y);
			float2 uv2 = make_float2(v2->tex._x, v2->tex._y);
			float2 uv = uv0 * traceResult.w + uv1 * traceResult.u + uv2 * traceResult.v;
			float3 n0 = V32F3(v0->norm);
			float3 n1 = V32F3(v1->norm);
			float3 n2 = V32F3(v2->norm);
			float3 norm = n0 * traceResult.w + n1 * traceResult.u + n2 * traceResult.v;
			float3 pos = V32F3(v0->pos) * traceResult.w + V32F3(v1->pos) * traceResult.u + V32F3(v2->pos) * traceResult.v;

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
			GetMaterialColors(mat, uv, tracerData.textures, diff, norm, emissive, trans, specular, metallic, roughness
				, anisotropic, sheen, sheenTint, clearcoat, clearcoatGloss);
			float3 nl = vecDot(norm, ray.dir) < 0.f ? norm : -1 * norm;
			{

				// Get some random microfacet
				float3 hDir = ImportanceSampleGGX(make_float2(curand_uniform(tracerData.randstate), curand_uniform(tracerData.randstate)), roughness, nl);

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
				RAYTYPE nextRayType = rayType;

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
						reflProb = 1.0f;// refrProb = 0.f;
					}
					else
					{
						refrDir = normalize(ray.dir * nnt - hDir * (ddn*nnt + sqrtf(cos2t)));
					}
				}

				if (reflProb > 0)
				{
					reflDir = normalize(ray.dir - hDir * 2 * vecDot(hDir,ray.dir));
					if (vecDot(reflDir, nl) < 0.f)
						reflProb = 0.f;
				}

				// Reflected
				if (ProbabilityRand(tracerData.randstate, reflProb))
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
					if (ProbabilityRand(tracerData.randstate, refrProb))
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

						float r1 = 2.f * M_PI * curand_uniform(tracerData.randstate);
						float r2cos = sqrtf(curand_uniform(tracerData.randstate));
						float r2sin = 1.f - r2cos*r2cos;
						float3 diffDir = normalize(w * r2cos + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

						nextRay = CURay(ray.orig + traceResult.dist * ray.dir + diffDir * M_FLT_BIAS_EPSILON, diffDir);
						//ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

						float VoH = vecDot(-1 * ray.dir, hDir);
						float NoV = vecDot(nl, -1 * ray.dir);
						float NoL = vecDot(nl, diffDir);
						//shadeResult = (M_PI * vecMul(Diffuse(diff, roughness, NoV, NoL, VoH), nextRayResult.light)) / ((1 - refrProb) * (1 - reflProb)) + emissive;
						lightMulTerm = M_PI * Diffuse(diff, roughness, NoV, NoL, VoH)/ ((1 - refrProb) * (1 - reflProb));
						nextRayType = RAYTYPE_DIFF;
					}
				}

				sample.pixelContrib = sample.pixelContrib * length(lightMulTerm);
				float nextRayResult;

				float3 lightContribFromLightVertex = make_float3(0.f, 0.f, 0.f);

				if ((rayType == RAYTYPE_DIFF && nextRayType == RAYTYPE_SPEC) || length(emissive) > 0.f)
					sample.pixelContrib = 0.f;

				bool isAccum = (sample.pathDepth == 0);
				if (!ProbabilityRand(tracerData.randstate,sample.pixelContrib))
				{
					sample.pixelContrib = 1.0f;
					sample.pathDepth = 0;
					sample.sampleTime++;
					nextRayType = RAYTYPE_EYE;
					float au = sample.uv.x + (curand_uniform(tracerData.randstate) - 0.5f) / tracerData.winSize->y * tan(tracerData.fov * 0.5f);
					float av = sample.uv.y + (curand_uniform(tracerData.randstate) - 0.5f) / tracerData.winSize->y * tan(tracerData.fov * 0.5f);
					float3 dir = normalize(*tracerData.camR * au + *tracerData.camU * av + *tracerData.camD);
					nextRay = CURay(*tracerData.camOrig, dir);
				}
				else
				{
					sample.pathDepth++;
					if (nextRayType == RAYTYPE_DIFF)
					{
						float3 lightFromLightVertex = make_float3(0.f, 0.f, 0.f);
						float3 toLightVertexDir = make_float3(0.f, 0.f, 0.f);
						GetLightFromRandLightVertices(pos + nl * M_FLT_BIAS_EPSILON, nl
							, tracerData, lightFromLightVertex, toLightVertexDir);
						lightContribFromLightVertex = vecMax(make_float3(0.f, 0.f, 0.f)
							, GetShadingResult(-1 * ray.dir, toLightVertexDir, lightFromLightVertex, nl, diff, metallic, roughness, specular
							, make_float2(1.f - trans, 1.f)));
						if (length(lightContribFromLightVertex) > 0.f)
						{
							sample.sampleTime += 4;
						}
					}
				}

				pt0_normalRay<depth + 1>(nextRay, nextRayType, sample, tracerData);

				sample.sampleResult = vecMul(lightMulTerm, sample.sampleResult) + emissive + lightContribFromLightVertex;

				if (isAccum && sample.sampleTime > 0)
				{
					sample.accumResult = sample.accumResult + sample.sampleResult;
					sample.sampleResult = make_float3(0.f, 0.f, 0.f);
				}
			}
		}
		else
		{
			sample.sampleTime++;
			sample.sampleResult = make_float3(0.f, 0.f, 0.f);
		}
	}

	template <>
	__device__ void pt0_normalRay<NORMALRAY_BOUND_MAX>(const CURay& ray, RAYTYPE rayType, PTSample& sample, TracerData& tracerData)
	{
		sample.sampleTime++;
		sample.sampleResult = make_float3(0.f, 0.f, 0.f);
	}

	__global__ void pt0_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
		float width, float height,LightVertex* lightVertices, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures
		, uint32 frameN, uint32 hashedFrameN, float* result, float* accResult, uint* sampleResultN)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= width || y >= height) 
			return;

		uint ind = (y * width + x);
		int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


		float u = (2.f * ((float)x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
		float v = (2.f * ((float)y + 0.5f) / height - 1.f) * tan(fov * 0.5f);

		curandState randstate;
		curand_init(hashedFrameN + ind, 0, 0, &randstate);
		float au = u + (curand_uniform(&randstate) - 0.5f) / height * tan(fov * 0.5f);
		float av = v + (curand_uniform(&randstate) - 0.5f) / height * tan(fov * 0.5f);

		float3 dir = normalize(camRight * au + camUp * av + camDir);
		CURay ray(camPos, dir);

		PTSample sampleResult(make_uint2(x,y), make_float2(u, v));

		float2 winSize = make_float2(width, height);
		TracerData tracerData(&winSize, fov, &camPos, &camUp, &camRight, &camDir, lightVertices, vertices, triangles, materials, textures, &randstate);
		pt0_normalRay(ray, RAYTYPE_EYE, sampleResult, tracerData);

		if (!frameN)
		{
			sampleResultN[ind] = 0;
		}
		uint tempNextSampleResultN = sampleResultN[ind] + sampleResult.sampleTime;
		float resultInf = 1.f / (float)(tempNextSampleResultN);
		float oldInf = sampleResultN[ind] * resultInf;
		result[ind * 3] = max(resultInf * sampleResult.accumResult.x + oldInf * result[ind * 3], 0.f);
		result[ind * 3 + 1] = max(resultInf * sampleResult.accumResult.y + oldInf * result[ind * 3 + 1], 0.f);
		result[ind * 3 + 2] = max(resultInf * sampleResult.accumResult.z + oldInf * result[ind * 3 + 2], 0.f);
		sampleResultN[ind] = tempNextSampleResultN;
	}

	void CleanMem()
	{
		freeAllBVHCudaMem();
		CUFREE(g_devResultData);
		CUFREE(g_devAccResultData);
		CUFREE(g_devLightVertices);
		CUFREE(g_devLightTri);
		CUFREE(g_devSampleResultN);
	}

	void initLightVerticesCudaMem(RTScene* scene)
	{
		HANDLE_ERROR(cudaMalloc((void**)&g_devLightVertices, sizeof(LightVertex) * LIGHTVERTEX_N));
		LightVertex tempLightVertices[LIGHTVERTEX_N];
		for (uint i = 0; i < LIGHTVERTEX_N; i++)
		{
			tempLightVertices[i] = LightVertex();
		}
		HANDLE_ERROR(cudaMemcpy(g_devLightVertices, tempLightVertices, sizeof(LightVertex) * LIGHTVERTEX_N, cudaMemcpyHostToDevice));
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

	uint32 WangHash(uint32 a) {
		a = (a ^ 61) ^ (a >> 16);
		a = a + (a << 3);
		a = a ^ (a >> 4);
		a = a * 0x27d4eb2d;
		a = a ^ (a >> 15);
		return a;
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
			g_matLastCamMat = g_matCurCamMat;
			g_uCurFrameN = 0;
			initAllSceneCudaMem(scene);
			initLightVerticesCudaMem(scene);
			updateLightTriCudaMem(scene);
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
			CUFREE(g_devSampleResultN);
			cudaMalloc((void**)&g_devSampleResultN, sizeof(uint) * width * height);
		}

		float3 f3CamPos = V32F3(camPos);
		float3 f3CamUp = V32F3(camUp);
		float3 f3CamDir = V32F3(camDir);
		float3 f3CamRight = V32F3(camRight);

		if (g_uCurFrameN%3 == 0)
		{
			dim3 block(BLOCK_SIZE, 1, 1);
			dim3 grid(LIGHTVERTEX_N / (block.x * LIGHTRAY_BOUND_MAX), 1, 1);
			bdpt_light_kernel << < grid, block >> > (f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height, g_devLightVertices, g_devLightTri, g_lightTriN, g_devVertices, g_devTriangles, g_devMaterials, g_devTextures
				, g_uCurFrameN, WangHash(g_uCurFrameN));
		}

		// Kernel go here
		dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
		dim3 grid(ceil(width / (float)block.x), ceil(height / (float)block.y), 1);
		pt0_kernel << < grid, block >> > (f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height, g_devLightVertices, g_devVertices, g_devTriangles, g_devMaterials, g_devTextures
			, g_uCurFrameN, WangHash(g_uCurFrameN), g_devResultData, g_devAccResultData, g_devSampleResultN);

		// Copy result to host
		cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
		return true;
	}
}