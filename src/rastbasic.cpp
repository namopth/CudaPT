#include "glhelper.h"
#include "camhelper.h"
#include "mathhelper.h"
#include "macrohelper.h"
#include "attrhelper.h"
#include "raytracer.h"

namespace rastBasic
{
	RT_ATTRIBS_N(0)
	RT_ATTRIBS_BGN
	RT_ATTRIBS_END

	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp
		, float fov, RTScene* scene, float width, float height, float* result)
	{
		return true;
	}
	void CleanMem()
	{

	}
}