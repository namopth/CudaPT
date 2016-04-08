#include "cudaRTCommon.h"

#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 16
#define NORMALRAY_BOUND_MAX 10
#define PATHSTREAM_SIZE 1E4*64
namespace cudaRTPTStreamRegen
{
	CUDA_RT_COMMON_ATTRIBS_N(0)
	CUDA_RT_COMMON_ATTRIBS_BGN
	CUDA_RT_COMMON_ATTRIBS_END

	enum RAYTYPE
	{
		RAYTYPE_EYE = 0,
		RAYTYPE_DIFF = 1,
		RAYTYPE_SPEC = 2
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

		__device__ PTPathVertex(uint _isTerminated, uint2 _pathPixel, float3 _pathOutDir, float3 _pathVertexPos, RAYTYPE _pathType, curandState _randState)
			: isTerminated(_isTerminated)
			, pathPixel(_pathPixel)
			, pathOutDir(_pathOutDir)
			, pathVertexPos(_pathVertexPos)
			, pathOutMulTerm(make_float3(1.f,1.f,1.f))
			, pathType(_pathType)
			, pathSample(make_float3(0.f, 0.f, 0.f))
			, pathAccumSample(make_float3(0.f, 0.f, 0.f))
			, pathSampleN(0)
			, pathSampleDepth(0)
			, randState(_randState)
		{}
	};

	PTPathVertex* g_devPathQueue = nullptr;
	uint g_uPathQueueCur = 0;
	uint g_uPathQueueSize = 0;
	PTPathVertex** g_devPathStream = nullptr;
	uint g_uPathStreamSize = PATHSTREAM_SIZE;
	PTPathVertex** g_devPathQueueRegen = nullptr;
	uint g_uPathQueueRegenSize = 0;

	void freeStreamMem()
	{
		g_uPathQueueCur = g_uPathQueueSize = 0;
		CUFREE(g_devPathQueue);
		CUFREE(g_devPathStream);
		CUFREE(g_devPathQueueRegen);
	}

	void allocateStreamMem(uint queueSize = 480000)
	{
		g_uPathQueueSize = queueSize;
		HANDLE_ERROR(cudaMalloc((void**)&g_devPathQueue, sizeof(PTPathVertex) * g_uPathQueueSize));
		HANDLE_ERROR(cudaMemset((void*)g_devPathQueue, 0, sizeof(PTPathVertex) * g_uPathQueueSize));

		HANDLE_ERROR(cudaMalloc((void**)&g_devPathStream, sizeof(PTPathVertex*) * g_uPathStreamSize));
		HANDLE_ERROR(cudaMemset((void*)g_devPathStream, 0, sizeof(PTPathVertex*) * g_uPathStreamSize));

		g_uPathQueueRegenSize = 0;
		HANDLE_ERROR(cudaMalloc((void**)&g_devPathQueueRegen, sizeof(PTPathVertex*) * g_uPathQueueSize));
		HANDLE_ERROR(cudaMemset((void*)g_devPathQueueRegen, 0, sizeof(PTPathVertex*) * g_uPathQueueSize));
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
					procVertex->pathVertexPos = nextRay.orig;
					procVertex->pathType = nextRayType;
					procVertex->pathSampleDepth++;
				}
			}
		}
		else
		{
			procVertex->pathAccumSample = procVertex->pathAccumSample + procVertex->pathSample;
			procVertex->pathSampleN++;
			procVertex->isTerminated = true;
		}
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

	__global__ void pt_assignPathStream_kernel(PTPathVertex** pathStream, uint pathStreamSize, PTPathVertex* pathQueue, uint pathQueueCur, uint pathQueueSize)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (ind >= pathStreamSize)
		{
			int pathQueueInd = pathQueueCur + ind - pathStreamSize;
			PTPathVertex* assignSample = nullptr;
			if (pathQueueInd < pathQueueSize)
			{
				assignSample = &pathQueue[pathQueueInd];
			}
			pathStream[ind] = assignSample;
		}
	}

	__global__ void pt_reinitRegenPathStream_kernel(PTPathVertex** pathQueueRegen, uint pathQueueRegenReinitSize,
		float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov, float width, float height)
	{
		uint ind = blockIdx.x * blockDim.x + threadIdx.x;
		if (ind < pathQueueRegenReinitSize)
		{
			PTPathVertex* assignSample = pathQueueRegen[ind];

			float u = (2.f * ((float)assignSample->pathPixel.x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
			float v = (2.f * ((float)assignSample->pathPixel.y + 0.5f) / height - 1.f) * tan(fov * 0.5f);

			float au = u + (curand_uniform(&assignSample->randState) - 0.5f) / height * tan(fov * 0.5f);
			float av = v + (curand_uniform(&assignSample->randState) - 0.5f) / height * tan(fov * 0.5f);

			float3 dir = normalize(camRight * au + camUp * av + camDir);

			*assignSample = PTPathVertex(false, assignSample->pathPixel, dir, camPos, RAYTYPE_EYE, assignSample->randState);
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
		freeStreamMem();
		freeAllBVHCudaMem();
		CUFREE(g_devResultData);
		CUFREE(g_devAccResultData);
	}

	struct is_terminated
	{
		__hd__ bool operator()(const PTPathVertex* vert)
		{
			return vert->isTerminated;
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

			size_t mem_tot;
			size_t mem_free;
			cudaMemGetInfo(&mem_free, &mem_tot);
			std::cout << "Memory Used : " << mem_tot-mem_free << "/" << mem_tot << " -> Free " << mem_free << std::endl;
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
		dim3 block1(BLOCK_SIZE*BLOCK_SIZE, 1, 1);
		dim3 block2(BLOCK_SIZE, BLOCK_SIZE, 1);
		dim3 renderGrid(ceil(width / (float)block2.x), ceil(height / (float)block2.y), 1);
		pt_genPathQueue_kernel << < renderGrid, block2 >> > (f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height
			, g_uCurFrameN, WangHash(g_uCurFrameN), g_devPathQueue);
		cudaDeviceSynchronize();

		uint maxActivePathStreamSize = min((uint)PATHSTREAM_SIZE, g_uPathQueueSize);
		uint activePathStreamSize = 0;
		uint activeRegenPathStreamSize = 0;
		g_uPathQueueCur = 0;
		g_uPathQueueRegenSize = 0;
		while (g_uPathQueueCur < g_uPathQueueSize || activePathStreamSize > 0)
		{
			uint tempActivePathStreamSize = activePathStreamSize;
			pt_assignPathStream_kernel << < dim3(ceil((float)PATHSTREAM_SIZE / (float)block1.x), 1, 1), block1 >> >(g_devPathStream, activePathStreamSize, g_devPathQueue, g_uPathQueueCur, g_uPathQueueSize);
			//readjust activePathStreamSize
			activePathStreamSize = min((uint)PATHSTREAM_SIZE, activePathStreamSize + (g_uPathQueueSize - g_uPathQueueCur));
			g_uPathQueueCur += activePathStreamSize - tempActivePathStreamSize;
			cudaDeviceSynchronize();

			//reinit regen path
			//if (activePathStreamSize + activeRegenPathStreamSize < maxActivePathStreamSize)
			//{
			//	uint reinitPathStreamSize = min((uint)maxActivePathStreamSize - (activePathStreamSize + activeRegenPathStreamSize), g_uPathQueueRegenSize - activeRegenPathStreamSize);
			//	pt_reinitRegenPathStream_kernel << < dim3(ceil((float)reinitPathStreamSize / (float)block1.x), 1, 1), block1 >> > 
			//		(g_devPathQueueRegen, reinitPathStreamSize, f3CamPos, f3CamDir, f3CamUp, f3CamRight, fov, width, height);
			//	activeRegenPathStreamSize += reinitPathStreamSize;
			//}

			pt_traceSample_kernel << < dim3(ceil((float)activePathStreamSize / (float)block1.x), 1, 1), block1 >> > (g_devVertices, g_devTriangles, g_devMaterials, g_devTextures, g_devPathStream, activePathStreamSize);
			cudaDeviceSynchronize();

			//prepare regen list
			//PTPathVertex** regenStreamEndItr = thrust::copy_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize, g_devPathQueueRegen + g_uPathQueueRegenSize, is_terminated());
			//g_uPathQueueRegenSize = regenStreamEndItr - g_devPathQueueRegen;

			//compact regenpathstream and find regenpathstreamsize value
			//if (activePathStreamSize < maxActivePathStreamSize)
			//{
			//	PTPathVertex** compactRegenStreamEndItr = thrust::remove_if(thrust::device, g_devPathQueueRegen, g_devPathQueueRegen + g_uPathQueueRegenSize, is_terminated());
			//	uint terminatedRegenPathStreamSize = g_uPathQueueRegenSize - (compactRegenStreamEndItr - g_devPathQueueRegen);
			//	activeRegenPathStreamSize -= terminatedRegenPathStreamSize;
			//	g_uPathQueueRegenSize -= terminatedRegenPathStreamSize;
			//}

			//compact pathstream and find activePathStreamSize value
			PTPathVertex** compactedStreamEndItr = thrust::remove_if(thrust::device, g_devPathStream, g_devPathStream + activePathStreamSize, is_terminated());
			activePathStreamSize = compactedStreamEndItr - g_devPathStream;
		}
		pt_applyPathQueueResult_kernel << < dim3(ceil((float)g_uPathQueueSize / (float)block1.x), 1, 1), block1 >> >(g_devPathQueue, g_uPathQueueSize, width, height, g_uCurFrameN, g_devResultData, g_devAccResultData);

		// Copy result to host
		cudaMemcpy(result, g_devResultData, g_resultDataSize, cudaMemcpyDeviceToHost);
		return true;
	}
}