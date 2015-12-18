#ifndef GEOHELPER_H
#define GEOHELPER_H

#include <vector>

#include "mathhelper.h"

namespace NPGeoHelper
{
	struct vertex
	{
		NPMathHelper::Vec3 pos;
		NPMathHelper::Vec3 norm;
		NPMathHelper::Vec3 binorm;
		NPMathHelper::Vec3 tan;
		NPMathHelper::Vec2 tex;
	};

	struct Geometry
	{
		std::vector<vertex> vertices;
		std::vector<unsigned int> indices;
	};
	Geometry GetSlicedHemisphereShape(const float radius, const unsigned int vertSlice, unsigned int horiSlice);
	Geometry GetFloorPlaneShape(const float width, const float height, const float uvmultiplier = 1.f);
	Geometry GetBoxShape(const float width, const float height, const float depth);
}

#endif