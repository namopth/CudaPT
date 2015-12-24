#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "mathhelper.h"
#include "rayhelper.h"

#include <vector>

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
	RTScene()
	{}

	bool Trace(const NPRayHelper::Ray &r, HitResult& result);

	void AddSphere(NPRayHelper::Sphere sphere) { m_vSpheres.push_back(sphere); }

protected:
	std::vector<NPRayHelper::Sphere> m_vSpheres;

};

class RTRenderer
{
public:
	RTRenderer();
	~RTRenderer();
	bool Init(const unsigned int width, const unsigned int height);
	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);
	bool RenderCUDA(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);
	bool RenderCPU(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene);

	inline const float* GetResult() { return m_pResult; }
protected:
	RTScene* m_pScene;
	float* m_pResult;
	unsigned int m_uSizeW;
	unsigned int m_uSizeH;
};
#endif