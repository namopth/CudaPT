#include "cudaRTCommon.h"

#define BLOCK_SIZE 16
#define NORMALRAY_BOUND_MAX 3
namespace cudaRTPT
{

	CUDA_RT_COMMON_ATTRIBS_N(0)
	CUDA_RT_COMMON_ATTRIBS_BGN
	CUDA_RT_COMMON_ATTRIBS_END

	float* g_devResultData = nullptr;
	float* g_devAccResultData = nullptr;

	NPMathHelper::Mat4x4 g_matLastCamMat;
	NPMathHelper::Mat4x4 g_matCurCamMat;
	uint32 g_uCurFrameN;
	size_t g_resultDataSize = 0;

	struct ShootRayResult
	{
		float3 light;
	};

	template <int depth = 0>
	__device__ ShootRayResult pt0_normalRay(const CURay& ray, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures, curandState *randstate)
	{
		ShootRayResult rayResult;
		if (depth > 5)
		{
			rayResult.light = make_float3(0.f, 0.f, 0.f);
			return rayResult;
		}

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
			float3 shadeResult = make_float3(0.f,0.f,0.f);
			float3 nl = vecDot(norm, ray.dir) < 0.f ? norm : -1 * norm;
#define MICROFACET_MODEL
#ifdef MICROFACET_MODEL
			{
				// Get some random microfacet
				float3 hDir = ImportanceSampleGGX(make_float2(curand_uniform(randstate), curand_uniform(randstate)), roughness, nl);

				// Calculate flesnel
				float voH = vecDot(-1 * ray.dir, hDir);
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
					reflDir = normalize(ray.dir - hDir * 2 * vecDot(hDir,ray.dir));
					if (vecDot(reflDir, nl) < 0.f)
						reflProb = 0.f;
				}

				// Reflected
				if (reflProb > 0 && curand_uniform(randstate) < reflProb)
				{
					CURay nextRay(ray.orig + (traceResult.dist - M_FLT_BIAS_EPSILON) * ray.dir, reflDir);
					ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
					// Microfacet specular = D*G*F / (4*NoL*NoV)
					// pdf = D * NoH / (4 * VoH)
					// (G * F * VoH) / (NoV * NoH)
					float VoH = vecDot(-1 * ray.dir, hDir);
					float NoV = vecDot(nl, -1 * ray.dir);
					float NoH = vecDot(nl, hDir);
					float NoL = vecDot(nl, reflDir);
					float G = GeometricVisibility(roughness, NoV, NoL, VoH);
					shadeResult = vecMul((brdf_f * G * VoH) / (NoV * NoH * reflProb) , nextRayResult.light) + emissive;
				}

				// Diffused or Transmited
				else
				{
					// Transmited
					if (refrProb > 0 && curand_uniform(randstate) < refrProb)
					{
						CURay nextRay(ray.orig + (traceResult.dist + M_FLT_BIAS_EPSILON) * ray.dir + refrDir * M_FLT_BIAS_EPSILON, refrDir);
						ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
						float cosine = vecDot(-1 * nl, refrDir);
						shadeResult = (cosine * vecMul(diff, nextRayResult.light)) / (refrProb * (1 - reflProb)) + emissive;
					}
					// Diffused
					else
					{
						float3 w = nl;
						float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
						float3 v = vecCross(w, u);
						u = vecCross(v, w);

						float r1 = 2.f * M_PI * curand_uniform(randstate);
						float r2cos = sqrtf(curand_uniform(randstate));
						float r2sin = 1.f - r2cos*r2cos;
						float3 diffDir = normalize(w * r2cos + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

						CURay nextRay(ray.orig + traceResult.dist * ray.dir + diffDir * M_FLT_BIAS_EPSILON, diffDir);
						ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);

						float VoH = vecDot(-1 * ray.dir, hDir);
						float NoV = vecDot(nl, -1 * ray.dir);
						float NoL = vecDot(nl, diffDir);
						shadeResult = (M_PI * vecMul(Diffuse(diff, roughness, NoV, NoL, VoH), nextRayResult.light)) / ((1 - refrProb) * (1 - reflProb)) + emissive;
					}
				}
			}
#else
			float refrProb = curand_uniform(randstate);
			float3 refrDir = ray.dir;
			if (refrProb < trans)
			{
				bool into = vecDot(nl, norm) > 0.f;
				float nt = specular * 0.8f + 1.f;
				float nc = 1.0f;
				float nnt = into ? nc / nt : nt / nc;
				float ddn = vecDot(nl, ray.dir);
				float cos2t = 1.f - nnt * nnt *(1.f - ddn * ddn);
				if (cos2t < 0.f)
				{
					float3 refDir = normalize(ray.dir - norm * 2 * vecDot(norm, ray.dir));
					CURay nextRay(ray.orig + traceResult.dist * ray.dir + refDir * M_FLT_BIAS_EPSILON, refDir);
					ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
					float cosine = vecDot(nl, refDir);
					shadeResult = (cosine * nextRayResult.light) / trans + emissive;
				}
				else
				{
					refrDir = normalize(ray.dir * nnt - norm * ((into ? 1 : -1)*(ddn*nnt + sqrtf(cos2t))));
					float a = nt - nc;
					float b = nt + nc;
					float r0 = a * a / (b * b);
					float c = 1.f - (into ? -ddn : vecDot(refrDir, norm));
					float re = r0 + (1.f - r0)*c*c*c*c*c;
					float tr = 1.f - re;
					float p = re;
					float reflProb = curand_uniform(randstate);
					if (reflProb < p)
					{
						float3 refDir = normalize(ray.dir - norm * 2 * vecDot(norm, ray.dir));
						CURay nextRay(ray.orig + traceResult.dist * ray.dir + refDir * M_FLT_BIAS_EPSILON, refDir);
						ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
						float cosine = vecDot(nl, refDir);
						shadeResult = (re * cosine * nextRayResult.light) / (trans * p) + emissive;
					}
					else
					{
						CURay nextRay(ray.orig + (traceResult.dist + M_FLT_BIAS_EPSILON) * ray.dir + refrDir * M_FLT_BIAS_EPSILON, refrDir);
						ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
						float cosine = vecDot(-1 * nl, refrDir);
						shadeResult = (tr * cosine * nextRayResult.light) / (trans * (1.f - p)) + emissive;
					}
				}
			}
			else
			{
				float3 w = nl;
				float3 u = normalize(vecCross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
				float3 v = vecCross(w, u);
				u = vecCross(v, w);

				//float r1 = 2.f * M_PI * curand_uniform(randstate);
				//float r2 = curand_uniform(randstate);
				//float r2s = sqrtf(r2);
				//float3 refDir = normalize(u*cosf(r1)*r2s + v*sinf(r1)*r2s + w*sqrtf(1.f - r2));

				float r1 = 2.f * M_PI * curand_uniform(randstate);
				float r2 = 0.5f * M_PI * curand_uniform(randstate);
				float r2sin = sinf(r2);
				float3 refDir = normalize(w * cosf(r2) + u * r2sin * cosf(r1) + v * r2sin * sinf(r1));

				CURay nextRay(ray.orig + traceResult.dist * ray.dir + refDir * M_FLT_BIAS_EPSILON, refDir);
				ShootRayResult nextRayResult = pt0_normalRay<depth + 1>(nextRay, vertices, triangles, materials, textures, randstate);
				nextRayResult.light = nextRayResult.light;
				float cosine = vecDot(nl, refDir);
				shadeResult = (M_PI * cosine * vecMul(diff, nextRayResult.light)) / (1 - trans) + emissive;
			}
#endif
			rayResult.light = shadeResult;
		}
		else
		{
			rayResult.light.x = 1.f;
			rayResult.light.y = 1.f;
			rayResult.light.z = 1.f;
		}

		return rayResult;
	}

	template <>
	__device__ ShootRayResult pt0_normalRay<NORMALRAY_BOUND_MAX>(const CURay& ray, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures
		, curandState *randstate)
	{
		ShootRayResult rayResult;
		rayResult.light.x = rayResult.light.y = rayResult.light.z = 0.f;
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

	__global__ void pt0_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov,
		float width, float height, RTVertex* vertices, RTTriangle* triangles, RTMaterial* materials, CURTTexture* textures
		, uint32 frameN, uint32 hashedFrameN, float* result, float* accResult)
	{
		uint x = blockIdx.x * blockDim.x + threadIdx.x;
		uint y = blockIdx.y * blockDim.y + threadIdx.y;
		uint ind = (y * width + x) * 3;
		int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;


		float u = (2.f * ((float)x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
		float v = (2.f * ((float)y + 0.5f) / height - 1.f) * tan(fov * 0.5f);

		curandState randstate;
		curand_init(hashedFrameN + ind, 0, 0, &randstate);
		u = u + (curand_uniform(&randstate) - 0.5f) / width;
		v = v + (curand_uniform(&randstate) - 0.5f) / height;

		float3 dir = normalize(camRight * u + camUp * v + camDir);
		CURay ray(camPos, dir);

		ShootRayResult rayResult = pt0_normalRay(ray, vertices, triangles, materials, textures, &randstate);

		float resultInf = 1.f / (float)(frameN + 1);
		float oldInf = 1.f - resultInf;
		result[ind] = max(resultInf * rayResult.light.x + oldInf * result[ind], 0.f);
		result[ind + 1] = max(resultInf * rayResult.light.y + oldInf * result[ind + 1], 0.f);
		result[ind + 2] = max(resultInf * rayResult.light.z + oldInf * result[ind + 2], 0.f);
	}

	void CleanMem()
	{
		freeAllBVHCudaMem();
		CUFREE(g_devResultData);
		CUFREE(g_devAccResultData);
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
		}
		else if (scene->GetIsCudaMaterialDirty())
		{
			updateAllSceneMaterialsCudaMem(scene);
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

		// Kernel go here
		dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
		dim3 grid(width / block.x, height / block.y, 1);
		pt0_kernel << < grid, block >> > (f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height, g_devVertices, g_devTriangles, g_devMaterials, g_devTextures
			, g_uCurFrameN, WangHash(g_uCurFrameN), g_devResultData, g_devAccResultData);

		// Copy result to host
		cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
		return true;
	}
}