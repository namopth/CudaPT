#ifdef RT_XGEN
DEFINE_RT(Cuda Pathtracing, cudaRTPT)
DEFINE_RT(Cuda Debug, cudaRTDebug)
DEFINE_RT(Cuda BVH Debug, cudaRTBVHDebug)
#endif

#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "mathhelper.h"
#include "rayhelper.h"
#include "bvhhelper.h"
#include "atbhelper.h"

#include <vector>

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
	NPMathHelper::Vec3 ambient;
	NPMathHelper::Vec3 specular;
	NPMathHelper::Vec3 emissive;
	float opacity;
	int32 diffuseTexId;
	int32 specularTexId;
	int32 emissiveTexId;
	RTMAT_TYPE matType;

	RTMaterial() 
		: diffuse(1.0f, 1.0f, 1.0f), ambient(1.0f, 1.0f, 1.0f)
		, specular(1.0f, 1.0f, 1.0f), emissive(1.0f, 1.0f, 1.0f)
		, opacity(1.0f)
		, diffuseTexId(-1), specularTexId(-1), emissiveTexId(-1)
		, matType(RTMAT_TYPE_DIFFUSE) 
	{}
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
	RTScene() : m_bIsCudaDirty(true)
	{}

	inline const NPBVHHelper::CompactBVH* GetCompactBVH() const { return &m_compactBVH; }
	inline const std::vector<NPMathHelper::Vec3>* GetTriIntersectData() const { return &m_triIntersectData; }
	inline bool GetIsCudaDirty() const { return m_bIsCudaDirty; }
	inline void SetIsCudaDirty(bool dirty = false) { m_bIsCudaDirty = dirty; }

	bool Trace(const NPRayHelper::Ray &r, HitResult& result);
	bool AddModel(const char* filename);

	std::vector<RTVertex> m_pVertices;
	std::vector<RTTriangle> m_pTriangles;
	std::vector<RTMaterial> m_pMaterials;
	std::vector<std::pair<std::string,RTTexture>> m_pTextures;
protected:
	std::vector<NPRayHelper::Sphere> m_vSpheres;
	NPBVHHelper::BVHNode m_bvhRootNode;
	NPBVHHelper::CompactBVH m_compactBVH;
	std::vector<NPMathHelper::Vec3> m_triIntersectData;
	bool m_bIsCudaDirty;
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
	TwBar* m_pMaterialBar;
};
#endif