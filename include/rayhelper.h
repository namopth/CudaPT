#ifndef RAYHELPER_H
#define RAYHELPER_H

#include "mathhelper.h"

namespace NPRayHelper
{
	struct Ray
	{
		NPMathHelper::Vec3 origPoint;
		NPMathHelper::Vec3 dir;
		Ray(NPMathHelper::Vec3 o = NPMathHelper::Vec3(), NPMathHelper::Vec3 d = NPMathHelper::Vec3())
			: origPoint(o), dir(d) 
		{}
	};

	struct TraceableObject
	{
	public:
		virtual bool intersect(const Ray &r, NPMathHelper::Vec3 &hitPoint, NPMathHelper::Vec3 &hitNormal) = 0;
	};

	struct Sphere : public TraceableObject
	{
		NPMathHelper::Vec3 centerPoint;
		float radius;
		Sphere(const NPMathHelper::Vec3 c = NPMathHelper::Vec3(), const float r = 0.f)
			: centerPoint(c), radius(r)
		{}

		virtual bool intersect(const Ray &r, NPMathHelper::Vec3 &hitPoint, NPMathHelper::Vec3 &hitNormal) override
		{
			NPMathHelper::Vec3 co = r.origPoint - centerPoint;
			float a = r.dir.dot(r.dir);
			if (abs(a) < M_EPSILON)
				return false;
			float b = 2.f * r.dir.dot(co);
			float c = co.dot(co) - radius * radius;
			float root = b * b - 4.f * a * c;
			if (root < 0)
				return false;
			root = sqrtf(root);
			float t = fmax((-b + root) / 2.f * a, (-b - root) / 2.f * a);
			if (t < 0)
				return false;
			hitPoint = r.origPoint + r.dir * t;
			hitNormal = (hitPoint - r.origPoint).normalize();
			return true;
		}
	};

	struct AABBBox
	{
		NPMathHelper::Vec3 minPoint;
		NPMathHelper::Vec3 maxPoint;
		AABBBox(NPMathHelper::Vec3 min = NPMathHelper::Vec3(), NPMathHelper::Vec3 max = NPMathHelper::Vec3())
			:minPoint(min), maxPoint(max)
		{}
	};

	struct Tri
	{
		NPMathHelper::Vec3 p0;
		NPMathHelper::Vec3 p1;
		NPMathHelper::Vec3 p2;
	};
}

#endif