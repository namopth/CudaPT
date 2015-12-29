#ifndef RAYHELPER_H
#define RAYHELPER_H

#include "mathhelper.h"
#include <algorithm>

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

	struct Sphere
	{
		NPMathHelper::Vec3 centerPoint;
		float radius;
		Sphere(const NPMathHelper::Vec3 c = NPMathHelper::Vec3(), const float r = 0.f)
			: centerPoint(c), radius(r)
		{}

		bool intersect(const Ray &r, NPMathHelper::Vec3 &hitPoint, NPMathHelper::Vec3 &hitNormal)
		{
			NPMathHelper::Vec3 co = r.origPoint - centerPoint;
			float a = r.dir.dot(r.dir);
			if (fabs(a) < M_EPSILON)
				return false;
			float b = 2.f * r.dir.dot(co);
			float c = co.dot(co) - radius * radius;
			float root = b * b - 4.f * a * c;
			if (root < 0)
				return false;
			root = sqrtf(root);
			float t0 = fmax((-b + root) / 2.f * a, 0.f);
			float t1 = fmax((-b - root) / 2.f * a, 0.f);
			float t = fmin(t0, t1);
			if (t <= 0)
				return false;
			hitPoint = r.origPoint + r.dir * t;
			hitNormal = (hitPoint - centerPoint).normalize();
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

		bool intersect(const Ray &r, NPMathHelper::Vec3 &hitPoint, NPMathHelper::Vec3 &hitNormal)
		{
			NPMathHelper::Vec3 modDir = r.dir;
			modDir._x = escapeZero(modDir._x, M_EPSILON);
			modDir._y = escapeZero(modDir._y, M_EPSILON);
			modDir._z = escapeZero(modDir._z, M_EPSILON);
			NPMathHelper::Vec3 tmin = (minPoint - r.origPoint) / modDir;
			NPMathHelper::Vec3 tmax = (maxPoint - r.origPoint) / modDir;
			NPMathHelper::Vec3 real_min = vecmin(tmin, tmax);
			NPMathHelper::Vec3 real_max = vecmax(tmin, tmax);
			float minmax = std::min(std::min(real_max._x, real_max._y), real_max._z);
			float maxmin = std::max(std::max(real_min._x, real_min._y), real_min._z);
			if (minmax >= maxmin && maxmin > M_EPSILON)
			{
				hitPoint = r.origPoint + r.dir * maxmin;
				hitNormal = (maxmin == real_min._x) ? NPMathHelper::Vec3(1.f, 0.f, 0.f) :
					(maxmin == real_min._y) ? NPMathHelper::Vec3(0.f, 1.f, 0.f) : NPMathHelper::Vec3(0.f, 0.f, 1.f);
				if (hitNormal.dot(r.dir) > 0.f)
					hitNormal = -1 * hitNormal;
				return true;
			}
			return false;
		}

	protected:
		float escapeZero(const float value, const float epsilon)
		{
			float result = value;
			if (fabs(result) < epsilon)
				result = (result > 0) ? result + epsilon : result - epsilon;
			return result;
		}
		NPMathHelper::Vec3 vecmin(const NPMathHelper::Vec3& lhs, const NPMathHelper::Vec3& rhs)
		{
			return NPMathHelper::Vec3(std::min(lhs._x, rhs._x), std::min(lhs._y, rhs._y), std::min(lhs._z, rhs._z));
		}
		NPMathHelper::Vec3 vecmax(const NPMathHelper::Vec3& lhs, const NPMathHelper::Vec3& rhs)
		{
			return NPMathHelper::Vec3(std::max(lhs._x, rhs._x), std::max(lhs._y, rhs._y), std::max(lhs._z, rhs._z));
		}
	};

	struct Tri
	{
		// CCW
		NPMathHelper::Vec3 p0;
		NPMathHelper::Vec3 p1;
		NPMathHelper::Vec3 p2;

		Tri(NPMathHelper::Vec3 a = NPMathHelper::Vec3()
			, NPMathHelper::Vec3 b = NPMathHelper::Vec3()
			, NPMathHelper::Vec3 c = NPMathHelper::Vec3())
			: p0(a)
			, p1(b)
			, p2(c)
		{

		}

		bool intersect(const Ray &r, NPMathHelper::Vec3 &hitPoint, NPMathHelper::Vec3 &hitNormal, float &w, float &u, float &v)
		{
			NPMathHelper::Vec3 e1 = p1 - p0;
			NPMathHelper::Vec3 e2 = p2 - p0;
			if (e1.cross(e2).dot(r.dir) > 0.f) return false;
			NPMathHelper::Vec3 de2 = r.dir.cross(e2);
			float divisor = de2.dot(e1);
			if (fabs(divisor) < M_EPSILON)
				return false;
			NPMathHelper::Vec3 t = r.origPoint - p0;
			NPMathHelper::Vec3 te1 = t.cross(e1);
			float rT = te1.dot(e2) / divisor;
			if (rT < 0.f)
				return false;
			u = de2.dot(t) / divisor;
			if (u < 0.0f || u > 1.0f)
				return false;
			v = te1.dot(r.dir) / divisor;
			if (v < 0.0f || (u + v) > 1.0f)
				return false;
			w = 1 - u - v;
			hitPoint = r.origPoint + rT * r.dir;
			hitNormal = e1.cross(e2).normalize();
			return true;
		}
	};
}

#endif