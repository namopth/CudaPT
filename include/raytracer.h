#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "mathhelper.h"

class RTScene
{
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
		unsigned int objType;
		unsigned int objId;
		unsigned int subObjId;
	};
public:
protected:
};

class RTRenderer
{
public:
protected:
};
#endif