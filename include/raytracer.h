#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "mathhelper.h"
#include "rayhelper.h"
#include "cudarayhelper.h"
#include "bvhhelper.h"

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

struct RTMaterial
{

};

class RTScene
{
public:
	enum RTOBJTYPE
	{
		OBJ_SPHERE = 0,
		OBJ_BOX = 1,
		OBJ_MESH = 2,
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

	void AddSphere(NPRayHelper::Sphere sphere) { m_vSpheres.push_back(sphere); }
	bool AddModel(const char* filename);

	std::vector<RTVertex> m_pVertices;
	std::vector<RTTriangle> m_pTriangles;
	std::vector<RTMaterial> m_pMaterials;
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
	enum RENDERER
	{
		RENDERER_CPU,
		RENDERER_CUDA,
		RENDERER_N
	};
	RTRenderer();
	~RTRenderer();
	bool Init(const unsigned int width, const unsigned int height);
	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);
	bool RenderCUDA(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);
	bool RenderCPU(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);
	bool Render2(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov
		, NPCudaRayHelper::Scene &scene);

	inline const float* GetResult() { return m_pResult; }
protected:
	RENDERER m_renderer;
	RTScene* m_pScene;
	float* m_pResult;
	unsigned int m_uSizeW;
	unsigned int m_uSizeH;
};
#endif