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

	struct Sphere
	{
		NPMathHelper::Vec3 centerPoint;
		float radius;
		Sphere(NPMathHelper::Vec3 c = NPMathHelper::Vec3(), float r = 0.f)
			: centerPoint(c), radius(r)
		{}
	};

	struct Box
	{
		NPMathHelper::Vec3 centerPoint;
		NPMathHelper::Vec3 size;
		Box(NPMathHelper::Vec3 c = NPMathHelper::Vec3(), NPMathHelper::Vec3 s = NPMathHelper::Vec3())
			:centerPoint(c), size(s)
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