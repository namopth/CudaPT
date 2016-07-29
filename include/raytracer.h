#ifdef RT_XGEN
DEFINE_RT(PTNaive, cudaRTPT)
//DEFINE_RT(PTNaiveRR, cudaRTPTRR)
//DEFINE_RT(Regen, cudaRTPTRegen)
DEFINE_RT(PTSC, cudaRTPTStream)
#ifdef FULLSPECTRAL
DEFINE_RT(PTSCSpec, cudaRTPTStreamSpec)
#endif
//DEFINE_RT(PTSCRegen, cudaRTPTStreamRegen)
//DEFINE_RT(BDPTRegen, cudaRTBDPTRegen)
DEFINE_RT(BDPTSC, cudaRTBDPTStream)
DEFINE_RT(BDPTSC Bilaterial, cudaRTBDPTStreamBilat)
//DEFINE_RT(BDPTSC Adap PDF, cudaRTBDPTStreamAdapPDF)
//DEFINE_RT(BDPTSC Adap Cheat Binary, cudaRTBDPTStreamAdapCheatBinary)
//DEFINE_RT(BDPTSC Adap Cheat Sort, cudaRTBDPTStreamAdapCheatSort)
//DEFINE_RT(BDPTSC Adap Empirical, cudaRTBDPTStreamAdapEmp)
//DEFINE_RT(BDPTSC Adap Bilat, cudaRTBDPTStreamAdapBilat)
//DEFINE_RT(BDPTSC Adap Bilat Sort, cudaRTBDPTStreamAdapBilatSort)
DEFINE_RT(BDPTSC Adap Bilat Binary, cudaRTBDPTStreamAdapBilatBinary)
DEFINE_RT(Cuda Debug, cudaRTDebug)
DEFINE_RT(Cuda BVH Debug, cudaRTBVHDebug)
#endif

#ifndef RAYTRACER_H
#define RAYTRACER_H

//#define FULLSPECTRAL
#include "mathhelper.h"
#include "rayhelper.h"
#include "bvhhelper.h"
#include "atbhelper.h"

#include <vector>

#ifdef FULLSPECTRAL
#include "cudaspechelper.h"
#endif

struct RTVertex
{
	NPMathHelper::Vec3 pos;
	NPMathHelper::Vec3 norm;
	NPMathHelper::Vec3 tan;
	NPMathHelper::Vec2 tex;
};

struct RTTriangle
{
	uint32 vertInd0;
	uint32 vertInd1;
	uint32 vertInd2;
	uint32 matInd;
};

enum RTMAT_TYPE
{
	RTMAT_TYPE_DIFFUSE,
	RTMAT_TYPE_SPECULAR,
	RTMAT_TYPE_REFRACT,
	RTMAT_TYPE_N
};

struct RTMaterial
{
	NPMathHelper::Vec3 diffuse;
	NPMathHelper::Vec3 emissive;
	int32 diffuseTexId;
	int32 normalTexId;
	int32 emissiveTexId;
	RTMAT_TYPE matType;

	float transparency;
	float subsurface;
	float specular;
	float metallic;
	float roughness;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;

#ifdef FULLSPECTRAL
	float directivity;
	bool isChangeWavelength;
	float specPara[NPCudaSpecHelper::c_u32SampleN*NPCudaSpecHelper::c_u32SampleN];
	bool isUseSpecIOR;
	float specIOR[NPCudaSpecHelper::c_u32SampleN];
#endif

	RTMaterial() 
		: diffuse(1.0f, 1.0f, 1.0f), emissive(1.0f, 1.0f, 1.0f)
		, diffuseTexId(-1), normalTexId(-1), emissiveTexId(-1)
		, matType(RTMAT_TYPE_DIFFUSE)
		, transparency(0.0f)
		, subsurface(0.f)
		, specular(0.5f)
		, metallic(0.f)
		, roughness(0.5f)
		, anisotropic(0.f)
		, sheen(0.f)
		, sheenTint(0.f)
		, clearcoat(0.f)
		, clearcoatGloss(0.f)
	{
#ifdef FULLSPECTRAL
		directivity = 0.f;
		for (uint32 i = 0; i < NPCudaSpecHelper::c_u32SampleN * NPCudaSpecHelper::c_u32SampleN; i++)
		{
			uint32 vertPos = i / NPCudaSpecHelper::c_u32SampleN;
			uint32 horiPos = i - vertPos * NPCudaSpecHelper::c_u32SampleN;
			specPara[i] = (vertPos == horiPos) ? 1.0f : 0.f;
		}
		isUseSpecIOR = false;
		for (uint32 i = 0; i < NPCudaSpecHelper::c_u32SampleN; i++)
		{
			specIOR[i] = 0.0f;
		}
#endif
	}

	bool operator==(const RTMaterial& rhs)
	{

#ifdef FULLSPECTRAL
		for (uint32 i = 0; i < NPCudaSpecHelper::c_u32SampleN*NPCudaSpecHelper::c_u32SampleN; i++)
		{
			if (specPara[i] != rhs.specPara[i])
			{
				return false;
			}
		}
		if (isChangeWavelength != rhs.isChangeWavelength)
		{
			return false;
		}
		if (isUseSpecIOR != rhs.isUseSpecIOR)
		{
			return false;
		}
		if (directivity != rhs.directivity)
		{
			return false;
		}
		for (uint32 i = 0; i < NPCudaSpecHelper::c_u32SampleN; i++)
		{
			if (specIOR[i] != rhs.specIOR[i])
			{
				return false;
			}
		}
#endif
		return (diffuse == rhs.diffuse) && (emissive == rhs.emissive)
			&& (diffuseTexId == rhs.diffuseTexId) && (normalTexId == rhs.normalTexId)
			&& (emissiveTexId == rhs.emissiveTexId) && (matType == rhs.matType)
			&& (transparency == rhs.transparency) && (specular == rhs.specular)
			&& (metallic == rhs.metallic) && (roughness == rhs.roughness)
			&& (anisotropic == rhs.anisotropic) && (sheen == rhs.sheen)
			&& (sheenTint == rhs.sheenTint) && (clearcoat == rhs.clearcoat)
			&& (clearcoatGloss == rhs.clearcoatGloss) && (subsurface == rhs.subsurface);
	}
	bool operator!=(const RTMaterial& rhs)
	{
		return !(*this == rhs);
	}
	void operator=(const RTMaterial& rhs)
	{
#ifdef FULLSPECTRAL
		directivity = rhs.directivity;
		for (uint32 i = 0; i < NPCudaSpecHelper::c_u32SampleN*NPCudaSpecHelper::c_u32SampleN; i++)
		{
			specPara[i] = rhs.specPara[i];
		}
		isChangeWavelength = rhs.isChangeWavelength;
		for (uint32 i = 0; i < NPCudaSpecHelper::c_u32SampleN; i++)
		{
			specIOR[i] = rhs.specIOR[i];
		}
		isUseSpecIOR = rhs.isUseSpecIOR;
#endif
		diffuse = rhs.diffuse;
		emissive = rhs.emissive;
		diffuseTexId = rhs.diffuseTexId;
		normalTexId = rhs.normalTexId;
		emissiveTexId = rhs.emissiveTexId;
		matType = rhs.matType;
		transparency = rhs.transparency;
		specular = rhs.specular;
		metallic = rhs.metallic;
		roughness = rhs.roughness;
		anisotropic = rhs.anisotropic;
		sheen = rhs.sheen;
		sheenTint = rhs.sheenTint;
		clearcoat = rhs.clearcoat;
		clearcoatGloss = rhs.clearcoatGloss;
		subsurface = rhs.subsurface;
	}
};

struct RTTexture
{
	float* data;
	uint32 width;
	uint32 height;

	RTTexture() : data(0), width(0), height(0) {}
};

class RTScene
{
public:
	enum RTOBJTYPE
	{
		OBJ_SPHERE = 0,
		OBJ_BOX = 1,
		OBJ_TRI = 2,
		OBJ_N
	};

	struct HitResult
	{
		NPMathHelper::Vec3 hitPosition;
		NPMathHelper::Vec3 hitNormal;
		unsigned int objType;
		unsigned int objId;
		unsigned int subObjId;
	};
	RTScene() : m_bIsCudaDirty(true), m_bIsCudaMaterialDirty(true), m_pMaterialBar(nullptr)
		, m_iCurrentMaterialId(-1)
	{}

	inline const NPBVHHelper::CompactBVH* GetCompactBVH() const { return &m_compactBVH; }
	inline const std::vector<NPMathHelper::Vec3>* GetTriIntersectData() const { return &m_triIntersectData; }
	inline bool GetIsCudaDirty() const { return m_bIsCudaDirty; }
	inline void SetIsCudaDirty(const bool dirty = false) { m_bIsCudaDirty = dirty; }
	inline bool GetIsCudaMaterialDirty() const { return m_bIsCudaMaterialDirty; }
	inline void SetIsCudaMaterialDirty(const bool dirty = false) { m_bIsCudaMaterialDirty = dirty; }

	bool Trace(const NPRayHelper::Ray &r, HitResult& result);
	bool TracePixel(const float x, const float y, const unsigned int width, const unsigned int height
		, NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, HitResult& result);
	bool AddModel(const char* filename);
	void UpdateMaterialsDirtyFlag();

	void SetTWMaterialBar(const int matId = -1);

#ifdef FULLSPECTRAL
	bool BrowseSpecReflFile(float wavelengthMul = 1.0f, float powerMul = 1.0f);
	bool BrowseBispecReflFile();
	bool BrowseSpecIORFile(float wavelengthMul = 1.0f, float iorMul = 1.0f);
#endif

	std::vector<std::string> m_vLoadedModelPaths;
	std::vector<RTVertex> m_pVertices;
	std::vector<RTTriangle> m_pTriangles;
	std::vector<RTMaterial> m_pMaterials;
	std::vector<std::pair<std::string,RTTexture>> m_pTextures;
protected:
	std::vector<RTMaterial> m_pLastMaterials;
	void updateTWBar();

	NPBVHHelper::BVHNode m_bvhRootNode;
	NPBVHHelper::CompactBVH m_compactBVH;
	std::vector<NPMathHelper::Vec3> m_triIntersectData;
	bool m_bIsCudaDirty;
	bool m_bIsCudaMaterialDirty;

	TwBar* m_pMaterialBar;
	int m_iCurrentMaterialId;
};

class RTRenderer
{
public:
	enum RENDERER_MODE
	{
		RENDERER_MODE_CPU_DEBUG,
#define RT_XGEN
#define DEFINE_RT(__name__, __codename__) RENDERER_MODE_##__codename__,
#include "raytracer.h"
#undef DEFINE_RT(__name__, __codename__)
#undef RT_XGEN
		RENDERER_MODE_N
	};
	RTRenderer();
	~RTRenderer();
	inline const RENDERER_MODE GetRendererMode() const { return m_renderer; }
	void SetRendererMode(const RENDERER_MODE mode);
	bool Init(const unsigned int width, const unsigned int height);
	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);
	inline const float* GetResult() { return m_pResult; }
protected:
	bool renderCPU(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);
	void updateTWBar();

	RENDERER_MODE m_renderer;
	RTScene* m_pScene;
	float* m_pResult;
	unsigned int m_uSizeW;
	unsigned int m_uSizeH;

	TwBar* m_pRenderBar;
};
#endif