#include "cudaRTCommon.h"


texture<float4, 1, cudaReadModeElementType> g_bvhMinMaxBounds;
texture<uint1, 1, cudaReadModeElementType> g_bvhOffsetTriStartN;
texture<float4, 1, cudaReadModeElementType> g_triIntersectionData;

CURTTexture* g_devTextures = nullptr;
std::vector<CURTTexture> g_cuRTTextures; // to CUFREE on CPU Side

RTVertex* g_devVertices = nullptr;
RTTriangle* g_devTriangles = nullptr;
RTMaterial* g_devMaterials = nullptr;
float4* g_devBVHMinMaxBounds = nullptr;
uint1* g_devBVHOffsetTriStartN = nullptr;
float4* g_devTriIntersectionData = nullptr;

bool g_bIsCudaInit = false;

__hd__ CURTTexture::CURTTexture() {}

__hd__ CURay::CURay(float3 _orig, float3 _dir) : orig(_orig), dir(_dir) {}
__hd__ float CURay::IntersectAABB(const float3& _min, const float3& _max) const
{
	float3 modDir = dir;
	modDir.x = escapeZero(modDir.x, M_EPSILON);
	modDir.y = escapeZero(modDir.y, M_EPSILON);
	modDir.z = escapeZero(modDir.z, M_EPSILON);
	modDir = vecRcp(modDir);
	float3 tmin = vecMul((_min - orig), modDir);
	float3 tmax = vecMul((_max - orig), modDir);
	float3 real_min = vecMin(tmin, tmax);
	float3 real_max = vecMax(tmin, tmax);
	float minmax = fminf(fminf(real_max.x, real_max.y), real_max.z);
	float maxmin = fmaxf(fmaxf(real_min.x, real_min.y), real_min.z);
	if (minmax >= maxmin)
		return (maxmin > M_EPSILON) ? maxmin : 0;
	return M_INF;
}

__hd__ float CURay::IntersectTri(const float3& _p0, const float3& _e0, const float3& _e1, float& w, float& u, float& v, float epsilon, bool cullback) const
{
	if (cullback && vecDot(vecCross(_e0, _e1), dir) > 0.f)
		return M_INF;
	float3 de2 = vecCross(dir, _e1);
	float divisor = vecDot(de2, _e0);
	if (fabs(divisor) < M_EPSILON)
		return M_INF;
	divisor = rcpf(divisor);
	float3 t = (orig + epsilon * dir) - _p0;
	float3 te1 = vecCross(t, _e0);
	float rT = vecDot(te1, _e1) * divisor;
	if (rT < 0.f)
		return M_INF;
	u = vecDot(de2, t) * divisor;
	if (u < 0.f || u > 1.f)
		return M_INF;
	v = vecDot(te1, dir) * divisor;
	if (v < 0.f || (u + v) > 1.f)
		return M_INF;
	w = 1 - u - v;
	return rT;
}

__device__ bool TracePrimitive(const CURay &ray, TracePrimitiveResult& result, const float maxDist, const float rayEpsilon, bool cullback)
{
	float minIntersect = maxDist;
	uint32 tracedTriId = 0;
	float w, u, v;
	uint32 traceCmd[BVH_DEPTH_MAX];
	traceCmd[0] = 0;
	int32 traceCmdPointer = 0;
	while (traceCmdPointer >= 0)
	{
		uint32 curInd = traceCmd[traceCmdPointer--];
		float4 boundMin = tex1Dfetch(g_bvhMinMaxBounds, curInd * 2);
		float4 boundMax = tex1Dfetch(g_bvhMinMaxBounds, curInd * 2 + 1);
		float min = ray.IntersectAABB(make_float3(boundMin.x, boundMin.y, boundMin.z),
			make_float3(boundMax.x, boundMax.y, boundMax.z));
		if (min >= 0 && min < minIntersect)
		{
			uint1 offOrTs = tex1Dfetch(g_bvhOffsetTriStartN, curInd * 2);
			uint1 tN = tex1Dfetch(g_bvhOffsetTriStartN, curInd * 2 + 1);
			if (tN.x == 0)
			{
				if (traceCmdPointer < BVH_DEPTH_MAX - 2)
				{
					traceCmd[++traceCmdPointer] = curInd + 1;
					traceCmd[++traceCmdPointer] = curInd + offOrTs.x;
				}
			}
			else
			{
				for (uint32 i = offOrTs.x; i < offOrTs.x + tN.x; i++)
				{
					float _w, _u, _v;
					float4 p0 = tex1Dfetch(g_triIntersectionData, i * 3);
					float4 e0 = tex1Dfetch(g_triIntersectionData, i * 3 + 1);
					float4 e1 = tex1Dfetch(g_triIntersectionData, i * 3 + 2);
					float triIntersect = ray.IntersectTri(make_float3(p0.x, p0.y, p0.z),
						make_float3(e0.x, e0.y, e0.z), make_float3(e1.x, e1.y, e1.z),
						_w, _u, _v, rayEpsilon, cullback);
					if (triIntersect >= 0 && triIntersect < minIntersect)
					{
						minIntersect = triIntersect;
						tracedTriId = i;
						w = _w; u = _u; v = _v;
					}
				}
			}
		}
	}

	if (minIntersect < maxDist)
	{
		result.dist = minIntersect;
		result.triId = tracedTriId;
		result.w = w;
		result.u = u;
		result.v = v;
		return true;
	}

	return false;
}

__hd__ float4 V32F4(const NPMathHelper::Vec3& vec3)
{
	return make_float4(vec3._x, vec3._y, vec3._z, 0.f);
}
__hd__ float3 V32F3(const NPMathHelper::Vec3& vec3)
{
	return make_float3(vec3._x, vec3._y, vec3._z);
}

__device__ void GetMaterialColors(const RTMaterial* mat, const float2 uv, const CURTTexture* textures, float4 &diff, float3 &ambient, float3 &specular, float3 &emissive)
{
	ambient = V32F3(mat->ambient);
	if (mat->diffuseTexId >= 0)
	{
		float4 texValue = tex2D<float4>(textures[mat->diffuseTexId].texObj, uv.x, uv.y);
		diff = texValue;//vecMul(diff, texValue);
	}
	else
	{
		diff = make_float4(mat->diffuse._x, mat->diffuse._y, mat->diffuse._z, 1.0f);
	}
	if (mat->specularTexId >= 0)
	{
		float4 texValue = tex2D<float4>(textures[mat->specularTexId].texObj, uv.x, uv.y);
		specular = make_float3(texValue.x, texValue.y, texValue.z);
	}
	else
	{
		specular = V32F3(mat->specular);
	}
	if (mat->emissiveTexId >= 0)
	{
		float4 texValue = tex2D<float4>(textures[mat->emissiveTexId].texObj, uv.x, uv.y);
		emissive = make_float3(texValue.x, texValue.y, texValue.z);
	}
	else
	{
		emissive = V32F3(mat->emissive);
	}
}

template<class T, int dim, enum cudaTextureReadMode readMode>
void BindCudaTexture(texture<T, dim, readMode> *tex, void* data, size_t size, uint32 filterMode)
{
	tex->normalized = false;
	tex->filterMode = (cudaTextureFilterMode)filterMode;
	tex->addressMode[0] = cudaAddressModeWrap;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	HANDLE_ERROR(cudaBindTexture(0, *tex, data, channelDesc, size));
}

CURTTexture CreateCURTTexture(const RTTexture &cpuTex)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	HANDLE_ERROR(cudaMallocArray(&cuArray, &channelDesc, cpuTex.width, cpuTex.height));
	HANDLE_ERROR(cudaMemcpyToArray(cuArray, 0, 0, cpuTex.data, cpuTex.width*cpuTex.height*sizeof(float4), cudaMemcpyHostToDevice));

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObj = 0;
	HANDLE_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

	CURTTexture cuRTTexture;
	cuRTTexture.texObj = texObj;
	cuRTTexture.cuArray = cuArray;
	cuRTTexture.width = cpuTex.width;
	cuRTTexture.height = cpuTex.height;

	return cuRTTexture;
}

void freeAllBVHCudaMem()
{
	HANDLE_ERROR(cudaUnbindTexture(g_bvhMinMaxBounds));
	HANDLE_ERROR(cudaUnbindTexture(g_bvhOffsetTriStartN));
	HANDLE_ERROR(cudaUnbindTexture(g_triIntersectionData));
	CUFREE(g_devBVHMinMaxBounds);
	CUFREE(g_devBVHOffsetTriStartN);
	CUFREE(g_devTriIntersectionData);
	CUFREE(g_devVertices);
	CUFREE(g_devTriangles);
	CUFREE(g_devMaterials);

	for (auto &cuRTTex : g_cuRTTextures)
	{
		cudaDestroyTextureObject(cuRTTex.texObj);
		CUFREEARRAY(cuRTTex.cuArray);
	}
	g_cuRTTextures.clear();

	g_bIsCudaInit = false;
}

void initAllBVHCudaMem(RTScene* scene)
{
	if (g_bIsCudaInit)
	{
		freeAllBVHCudaMem();
	}

	// Texture
	CURTTexture* tempCURTTextures = new CURTTexture[scene->m_pTextures.size()];
	for (uint32 i = 0; i < scene->m_pTextures.size(); i++)
	{
		tempCURTTextures[i] = CreateCURTTexture(scene->m_pTextures[i].second);
		g_cuRTTextures.push_back(tempCURTTextures[i]);
	}

	uint vertSize = scene->m_pVertices.size();
	RTVertex* tempVertices = new RTVertex[vertSize];
	{
		auto f = [&](const tbb::blocked_range< int >& range) {
			for (unsigned int i = range.begin(); i < range.end(); i++)
			{
				tempVertices[i] = scene->m_pVertices[i];
			}
		};
		tbb::parallel_for(tbb::blocked_range< int >(0, vertSize), f);
	}


	uint triSize = scene->m_pTriangles.size();
	RTTriangle* tempTriangles = new RTTriangle[triSize];
	float4* tempTriIntersectionData = new float4[triSize * 3];
	{
		auto f = [&](const tbb::blocked_range< int >& range) {
			for (unsigned int i = range.begin(); i < range.end(); i++)
			{
				tempTriIntersectionData[i * 3] = V32F4((*scene->GetTriIntersectData())[i * 3]);
				tempTriIntersectionData[i * 3 + 1] = V32F4(((*scene->GetTriIntersectData())[i * 3 + 1] - (*scene->GetTriIntersectData())[i * 3]));
				tempTriIntersectionData[i * 3 + 2] = V32F4(((*scene->GetTriIntersectData())[i * 3 + 2] - (*scene->GetTriIntersectData())[i * 3]));
				tempTriangles[i] = scene->m_pTriangles[i];
			}
		};
		tbb::parallel_for(tbb::blocked_range< int >(0, triSize), f);
	}

	RTMaterial* tempMaterials = new RTMaterial[scene->m_pMaterials.size()];
	{
		auto f = [&](const tbb::blocked_range< int >& range) {
			for (unsigned int i = range.begin(); i < range.end(); i++)
			{
				tempMaterials[i] = scene->m_pMaterials[i];
			}
		};
		tbb::parallel_for(tbb::blocked_range< int >(0, scene->m_pMaterials.size()), f);
	}

	uint bvhNodeN = scene->GetCompactBVH()->nodeN;
	float4* tempBVHMinMaxBounds = new float4[bvhNodeN * 2];
	uint1* tempBVHOffsetTriStartN = new uint1[bvhNodeN * 2];
	{
		auto f = [&](const tbb::blocked_range< int >& range) {
			for (unsigned int i = range.begin(); i < range.end(); i++)
			{
				tempBVHMinMaxBounds[i * 2] = V32F4(scene->GetCompactBVH()->bounds[i].minPoint);
				tempBVHMinMaxBounds[i * 2 + 1] = V32F4(scene->GetCompactBVH()->bounds[i].maxPoint);
				tempBVHOffsetTriStartN[i * 2].x = scene->GetCompactBVH()->offOrTSTN[i * 2];
				tempBVHOffsetTriStartN[i * 2 + 1].x = scene->GetCompactBVH()->offOrTSTN[i * 2 + 1];
			}
		};
		tbb::parallel_for(tbb::blocked_range< int >(0, bvhNodeN), f);
	}

	// Create Dev Data
	HANDLE_ERROR(cudaMalloc((void**)&g_devTextures, sizeof(CURTTexture) * scene->m_pTextures.size()));
	HANDLE_ERROR(cudaMalloc((void**)&g_devMaterials, sizeof(RTMaterial) * scene->m_pMaterials.size()));
	HANDLE_ERROR(cudaMalloc((void**)&g_devTriangles, sizeof(RTTriangle) * triSize));
	HANDLE_ERROR(cudaMalloc((void**)&g_devVertices, sizeof(RTVertex) * vertSize));
	HANDLE_ERROR(cudaMalloc((void**)&g_devTriIntersectionData, sizeof(float4) * triSize * 3));
	HANDLE_ERROR(cudaMalloc((void**)&g_devBVHOffsetTriStartN, sizeof(uint1) * bvhNodeN * 2));
	HANDLE_ERROR(cudaMalloc((void**)&g_devBVHMinMaxBounds, sizeof(float4) * bvhNodeN * 2));

	// MemCpy Dev Data
	HANDLE_ERROR(cudaMemcpy(g_devTextures, tempCURTTextures, sizeof(CURTTexture) * scene->m_pTextures.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g_devMaterials, tempMaterials, sizeof(RTMaterial) * scene->m_pMaterials.size(), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g_devTriangles, tempTriangles, sizeof(RTTriangle) * triSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g_devVertices, tempVertices, sizeof(RTVertex) * vertSize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g_devTriIntersectionData, tempTriIntersectionData, sizeof(float4) * triSize * 3, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g_devBVHOffsetTriStartN, tempBVHOffsetTriStartN, sizeof(uint1) * bvhNodeN * 2, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g_devBVHMinMaxBounds, tempBVHMinMaxBounds, sizeof(float4) * bvhNodeN * 2, cudaMemcpyHostToDevice));

	// Del Temp Data
	DEL_ARRAY(tempBVHOffsetTriStartN);
	DEL_ARRAY(tempBVHMinMaxBounds);
	DEL_ARRAY(tempTriIntersectionData);
	DEL_ARRAY(tempTriangles);
	DEL_ARRAY(tempMaterials);
	DEL_ARRAY(tempVertices);
	DEL_ARRAY(tempCURTTextures);

	// Bind Dev Data To Texture
	BindCudaTexture(&g_bvhMinMaxBounds, g_devBVHMinMaxBounds, sizeof(float4) * bvhNodeN * 2);
	BindCudaTexture(&g_bvhOffsetTriStartN, g_devBVHOffsetTriStartN, sizeof(uint1) * bvhNodeN * 2);
	BindCudaTexture(&g_triIntersectionData, g_devTriIntersectionData, sizeof(float4) * triSize * 3);

	g_bIsCudaInit = true;
	scene->SetIsCudaDirty();
}