#ifndef BVHHELPER_H
#define BVHHELPER_H

#include "rayhelper.h"

#include <vector>

namespace NPBVHHelper
{

	struct BVHNode
	{
		BVHNode* childNodes[2];
		NPRayHelper::AABBBox bound;
		uint32 triStart;
		uint32 triN;
		uint32 desN;
		float boundFaceArea[3];
		BVHNode() : triStart(0), triN(0), desN(0)
			, bound(NPMathHelper::Vec3(M_INF, M_INF, M_INF), NPMathHelper::Vec3(M_MIN_INF, M_MIN_INF, M_MIN_INF))
		{
			childNodes[0] = childNodes[1] = 0; boundFaceArea[0] = boundFaceArea[1] = boundFaceArea[2] = 0.f;
		}

		void Clear()
		{
			triStart = 0;
			triN = 0;
			desN = 0;
			bound = NPRayHelper::AABBBox(NPMathHelper::Vec3(M_INF, M_INF, M_INF), NPMathHelper::Vec3(M_MIN_INF, M_MIN_INF, M_MIN_INF));
			if (childNodes[0])
			{
				childNodes[0]->Clear();
				delete childNodes[0];
			}
			if (childNodes[1])
			{
				childNodes[1]->Clear();
				delete childNodes[1];
			}
			childNodes[0] = childNodes[1] = nullptr;
		}
	};

	// return reordered tri order
	std::vector<uint32> CreateBVH(BVHNode* root, const std::vector<uint32> &tri, const std::vector<NPMathHelper::Vec3> &vert, uint32 maxDepth = 32);

	struct CompactBVH
	{
		NPRayHelper::AABBBox* bounds;
		float* boundsFaceArea;
		uint32* offTSTN;
		uint32 nodeN;

		CompactBVH() : bounds(0), boundsFaceArea(0), offTSTN(0), nodeN(0) {}
		bool IsValid() const { return (bounds && offTSTN && nodeN > 0); }
		bool InitialCompactBVH(const BVHNode* bvhRoot);
	};
}
#endif