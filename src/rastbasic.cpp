#include "glhelper.h"
#include "camhelper.h"
#include "mathhelper.h"
#include "macrohelper.h"
#include "attrhelper.h"
#include "raytracer.h"

namespace rastBasic
{
	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp
		, float fov, RTScene* scene, float width, float height, float* result)
	{
		return true;
	}
	void CleanMem()
	{

	}
	NPAttrHelper::Attrib* GetAttribute(unsigned __int32 ind, std::string &name)
	{
		return nullptr;
	}
	unsigned __int32 GetAttribsN()
	{
		return 0;
	}
}