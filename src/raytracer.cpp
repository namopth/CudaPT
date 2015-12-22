#include "raytracer.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>


bool RTScene::Trace(const NPRayHelper::Ray &r, HitResult& result)
{
	float minIntersect = M_INF;
	for (int i = 0; i < m_vSpheres.size(); i++)
	{
		NPRayHelper::Sphere sphere = m_vSpheres[i];
		NPMathHelper::Vec3 pos, norm;
		if (sphere.intersect(r, pos, norm))
		{
			float dist = (pos - r.origPoint).length();
			if (dist < minIntersect)
			{
				minIntersect = dist;
				result.hitPosition = pos;
				result.hitNormal = norm;
				result.objId = i;
				result.objType = OBJ_SPHERE;
			}
		}
	}

	NPRayHelper::Tri test = NPRayHelper::Tri(NPMathHelper::Vec3(0.f, 0.f, -2.f)
		, NPMathHelper::Vec3(0.f, 2.f, -2.f), NPMathHelper::Vec3(-2.f, 0.f, -2.f));

	NPMathHelper::Vec3 pos, norm;
	float w, u, v;
	if (test.intersect(r, pos, norm, w, u, v))
	{
		float dist = (pos - r.origPoint).length();
		if (dist < minIntersect)
		{
			minIntersect = dist;
			result.hitPosition = pos;
			result.hitNormal = NPMathHelper::Vec3(w, u, v);
			result.objId = 0;
			result.objType = OBJ_SPHERE;
		}
	}

	NPRayHelper::AABBBox test2 = NPRayHelper::AABBBox(NPMathHelper::Vec3(0.f, -4.f, 0.f)
		, NPMathHelper::Vec3(2.f, -2.f, 2.f));
	if (test2.intersect(r, pos, norm))
	{
		float dist = (pos - r.origPoint).length();
		if (dist < minIntersect)
		{
			minIntersect = dist;
			result.hitPosition = pos;
			result.hitNormal = norm;
			result.objId = 0;
			result.objType = OBJ_SPHERE;
		}
	}

	return (minIntersect < M_INF);
}

RTRenderer::RTRenderer()
	: m_pResult(0)
	, m_pScene(0)
{

}
RTRenderer::~RTRenderer()
{
	if (m_pResult)
	{
		delete m_pResult;
		m_pResult = nullptr;
	}
}

bool RTRenderer::Init(const unsigned int width, const unsigned int height)
{
	if (m_pResult && (m_uSizeW != width || m_uSizeH != height))
	{
		delete m_pResult;
		m_pResult = nullptr;
	}

	m_uSizeW = width;
	m_uSizeH = height;

	if (!m_pResult)
	{
		m_pResult = new float[width * height * 3];
	}

	return true;
}

bool RTRenderer::Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene)
{
	auto f = [&](const tbb::blocked_range2d< int, int >& range) {
		for (unsigned int i = range.cols().begin(); i < range.cols().end(); i++)
		{
			for (unsigned int j = range.rows().begin(); j < range.rows().end(); j++)
			{
				unsigned int ind = (i + j * m_uSizeW) * 3.f;
				float u = 2.f * ((float)i - 0.5f * m_uSizeW + 0.5f) / m_uSizeW * tan(fov * 0.5f);
				float v = 2.f * ((float)j - 0.5f * m_uSizeH + 0.5f) / m_uSizeH * tan(fov * 0.5f) / (float)m_uSizeW * (float)m_uSizeH;
				NPMathHelper::Vec3 camRight = camDir.cross(camUp).normalize();
				NPMathHelper::Vec3 dir = (camRight * u + camUp * v + camDir).normalize();
				NPRayHelper::Ray ray(camPos, dir);

				RTScene::HitResult hitResult;
				if (scene.Trace(ray, hitResult))
				{
					m_pResult[ind] = hitResult.hitNormal._x;
					m_pResult[ind + 1] = hitResult.hitNormal._y;
					m_pResult[ind + 2] = hitResult.hitNormal._z;
				}
				else
				{
					m_pResult[ind] = 0;
					m_pResult[ind + 1] = 0;
					m_pResult[ind + 2] = 0;
				}
			}
		}
	};

	tbb::parallel_for(tbb::blocked_range2d< int, int >(0, m_uSizeH, 0, m_uSizeW), f);

	return true;
}
