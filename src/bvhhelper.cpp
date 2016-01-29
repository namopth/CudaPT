#include "bvhhelper.h"
#include "macrohelper.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <iostream>

#define BVH_SAH_BIN_N 16

namespace NPBVHHelper
{
	struct BVHTriangle
	{
		uint32 vertInd[3];
		NPMathHelper::Vec3 centroid;
		NPRayHelper::AABBBox bound;
	};

	struct BVHBIN
	{
		uint32 triN;
		//NPRayHelper::AABBBox bound;
		NPRayHelper::AABBBox centroidBound;

		BVHBIN()
			: triN(0)
			//, bound(NPMathHelper::Vec3(M_INF, M_INF, M_INF), NPMathHelper::Vec3(M_MIN_INF, M_MIN_INF, M_MIN_INF))
			, centroidBound(NPMathHelper::Vec3(M_INF, M_INF, M_INF), NPMathHelper::Vec3(M_MIN_INF, M_MIN_INF, M_MIN_INF))
		{}
	};

	float AABBSurfaceArea(const NPRayHelper::AABBBox &bound)
	{
		NPMathHelper::Vec3 boundDia = bound.maxPoint - bound.minPoint;
		if (boundDia._x < 0 || boundDia._y < 0 || boundDia._z < 0 || boundDia.length() < M_EPSILON)
			return 0;
		return boundDia._x * boundDia._y + boundDia._x * boundDia._z + boundDia._y * boundDia._z;
	}

	void InitialBVHLeafNode(BVHNode* node, uint32 triStart, uint32 triN)
	{
		node->triStart = triStart;
		node->triN = triN;
		node->childNodes[0] = node->childNodes[1] = nullptr;
	}

	void InitialBVHNode(BVHNode* node
		, BVHTriangle* bvhTris, uint32* reorderedTriOrder, uint32 triStart, uint32 triN, uint32 depthBudget)
	{
		if (triN <= 0) 
			return;
		// calculate bound
		NPRayHelper::AABBBox centroidBound(NPMathHelper::Vec3(M_INF, M_INF, M_INF), NPMathHelper::Vec3(M_MIN_INF, M_MIN_INF, M_MIN_INF));
		for (uint32 i = 0; i < triN; i++)
		{
			uint32 ind = reorderedTriOrder[triStart + i];
			node->bound = node->bound.merge(bvhTris[ind].bound);
			centroidBound = centroidBound.merge(bvhTris[ind].centroid);
		}
		//std::cout << "==node==" << std::endl;
		//for (uint32 i = 0; i != 24; i++)
		//	std::cout << reorderedTriOrder[i] << " ";
		//std::cout << std::endl;

		NPMathHelper::Vec3 boundDia = centroidBound.maxPoint - centroidBound.minPoint;
		if (boundDia._x < 0 || boundDia._y < 0 || boundDia._z < 0)
			return;
		if (triN == 1 || depthBudget <= 1 || boundDia.length() < M_EPSILON)
			return InitialBVHLeafNode(node, triStart, triN);

		// choose split axis
		uint32 splitAxis = 0;
		if (abs(boundDia._y) > abs(boundDia._x)) splitAxis = 1;
		if (abs(boundDia._z) > abs(boundDia._y) && abs(boundDia._z) > abs(boundDia._x)) splitAxis = 2;
		float longestAxisLength = abs(boundDia._e[splitAxis]);

		// calculate bins
		BVHBIN binData[BVH_SAH_BIN_N];
		{
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
			//for (unsigned int i = 0; i < BVH_SAH_BIN_N; i++)
				{
					float binRangeStart = centroidBound.minPoint._e[splitAxis] + (float)i * longestAxisLength / (float)BVH_SAH_BIN_N;
					float binRangeEnd = centroidBound.minPoint._e[splitAxis] + (float)(i + 1) * longestAxisLength / (float)BVH_SAH_BIN_N;
					if (i + 1 == BVH_SAH_BIN_N)
						binRangeEnd = centroidBound.maxPoint._e[splitAxis] + M_FLT_BIAS_EPSILON;
					for (uint32 j = 0; j < triN; j++)
					{
						uint32 ind = reorderedTriOrder[triStart + j];
						float triPos = bvhTris[ind].centroid._e[splitAxis];
						if (triPos >= binRangeStart && triPos < binRangeEnd)
						{
							binData[i].triN++;
							//binData[i].bound = binData[i].bound.merge(bvhTris[ind].bound);
							binData[i].centroidBound = binData[i].centroidBound.merge(bvhTris[ind].centroid);
						}
					}
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, BVH_SAH_BIN_N), f);
		}

		//Calculate heuristic value for each case
		float nodeSurfaceArea = AABBSurfaceArea(node->bound);
		float binHeu[BVH_SAH_BIN_N];
		{
			// two thread from head to mid and last to mid
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
			//for (unsigned int i = 0; i < 2; i++)
				{
					uint32 startBin = 0;
					uint32 endBin = BVH_SAH_BIN_N / 2;
					uint32 step = 1;
					if (i == 1)
					{
						startBin = BVH_SAH_BIN_N - 1;
						endBin = BVH_SAH_BIN_N / 2 - 1;
						step = -1;
					}
					uint32 mainAccumN = 0;
					NPRayHelper::AABBBox mainAccumBound(NPMathHelper::Vec3(M_INF, M_INF, M_INF), NPMathHelper::Vec3(M_MIN_INF, M_MIN_INF, M_MIN_INF));
					for (uint32 j = startBin; j != endBin; j += step)
					{
						mainAccumN += binData[j].triN;
						mainAccumBound = mainAccumBound.merge(binData[j].centroidBound);

						uint32 opN = 0;
						NPRayHelper::AABBBox opBount(NPMathHelper::Vec3(M_INF, M_INF, M_INF), NPMathHelper::Vec3(M_MIN_INF, M_MIN_INF, M_MIN_INF));
						for (uint32 k = j + step; k >= 0 && k < BVH_SAH_BIN_N; k += step)
						{
							opN += binData[k].triN;
							opBount = opBount.merge(binData[k].centroidBound);
						}
						binHeu[j] = 0.125f + (mainAccumN * AABBSurfaceArea(mainAccumBound) + opN * AABBSurfaceArea(opBount)) / nodeSurfaceArea;
					}
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, 2), f);
		}

		//Choose the best splitting bin
		float minBinCost = M_INF;
		float minBinN = BVH_SAH_BIN_N - 1;
		for (uint32 i = 0; i < BVH_SAH_BIN_N; i++)
		{
			if (binHeu[i] <= minBinCost)
			{
				minBinCost = binHeu[i];
				minBinN = i;
			}
		}

		//If leaf node is more efficient
		if (minBinN == BVH_SAH_BIN_N - 1)
			return InitialBVHLeafNode(node, triStart, triN);

		//re order tri order
		uint32 mid = triStart;
		{
			float binSplit = centroidBound.minPoint._e[splitAxis] + (float)(minBinN + 1) * longestAxisLength / (float)BVH_SAH_BIN_N;

			uint32 left = triStart;
			uint32 right = triStart + triN - 1;
			while (left < right)
			{
				while (bvhTris[reorderedTriOrder[left]].centroid._e[splitAxis] <= binSplit && left < right) left++;
				while (bvhTris[reorderedTriOrder[right]].centroid._e[splitAxis] > binSplit && left < right) right--;
				if (left < right)
				{
					XORSWAP(reorderedTriOrder[left], reorderedTriOrder[right]);
					left++;
					right--;
				}
			}
			if (bvhTris[reorderedTriOrder[left]].centroid._e[splitAxis] <= binSplit)
				mid = left;
			else
				mid = left - 1;


			//std::cout << "==reordered split at" << binSplit << "==" << std::endl;
			//for (uint32 i = 0; i != 24; i++)
			//{
			//	if (triStart == i)
			//		std::cout << "[";
			//	std::cout << bvhTris[reorderedTriOrder[i]].centroid._e[splitAxis] << " ";
			//	if (i == triStart + triN - 1)
			//		std::cout << "]";
			//	if (i == mid)
			//		std::cout << "|";
			//}
			//std::cout << std::endl;
		}

		//recursive
		BVHNode* leftNode = new BVHNode();
		BVHNode* rightNode = new BVHNode();
		node->childNodes[0] = leftNode;
		node->childNodes[1] = rightNode;
		InitialBVHNode(leftNode, bvhTris, reorderedTriOrder, triStart, mid - triStart + 1, depthBudget-1);
		InitialBVHNode(rightNode, bvhTris, reorderedTriOrder, mid + 1, triN - (mid - triStart + 1), depthBudget-1);
		node->desN = 2 + leftNode->desN + rightNode->desN;
		return;

	}

	std::vector<uint32> CreateBVH(BVHNode* root, const std::vector<uint32> &tri, const std::vector<NPMathHelper::Vec3> &vert, uint32 maxDepth)
	{
		if (tri.size() == 0 || !root)
			return std::vector<uint32>();

		uint32 triSize = tri.size() / 3;
		uint32* rawReorderedTriOrder = new uint32[triSize];
		BVHTriangle* rawBVHTris = new BVHTriangle[triSize];
		{
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
			//for (unsigned int i = 0; i < triSize; i++)
				{
					rawReorderedTriOrder[i] = i;
					rawBVHTris[i].vertInd[0] = tri[i * 3];
					rawBVHTris[i].vertInd[1] = tri[i * 3 + 1];
					rawBVHTris[i].vertInd[2] = tri[i * 3 + 2];
					rawBVHTris[i].centroid = (vert[tri[i * 3]] + vert[tri[i * 3 + 1]] + vert[tri[i * 3 + 2]]) / 3.f;
					rawBVHTris[i].bound.minPoint = vert[tri[i * 3 + 2]].min(vert[tri[i * 3 + 1]].min(vert[tri[i * 3]]));
					rawBVHTris[i].bound.maxPoint = vert[tri[i * 3 + 2]].max(vert[tri[i * 3 + 1]].max(vert[tri[i * 3]]));
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, triSize), f);
		}

		// Our journey start here!
		InitialBVHNode(root, rawBVHTris, rawReorderedTriOrder, 0, triSize, maxDepth);
		// Ok its done guys!

		std::vector<uint32> reorderedTriOrder(triSize);
		{
			auto f = [&](const tbb::blocked_range< int >& range) {
				for (unsigned int i = range.begin(); i < range.end(); i++)
				{
					reorderedTriOrder[i] = rawReorderedTriOrder[i];
				}
			};
			tbb::parallel_for(tbb::blocked_range< int >(0, triSize), f);
		}

		delete[] rawReorderedTriOrder; rawReorderedTriOrder = nullptr;
		delete[] rawBVHTris; rawBVHTris = nullptr;

		return reorderedTriOrder;
	}

	uint32 InitialCompactBVHOnNode(const BVHNode* node, CompactBVH* compact, uint32 curInd)
	{
		compact->bounds[curInd] = node->bound;
		if (!node->childNodes[0] || !node->childNodes[1])
		{
			compact->offOrTSTN[curInd * 2] = node->triStart;
			compact->offOrTSTN[curInd * 2 + 1] = node->triN;
			return 1;
		}
		compact->offOrTSTN[curInd * 2 + 1] = 0;
		uint32 firstChildDes = InitialCompactBVHOnNode(node->childNodes[0], compact, curInd + 1);
		compact->offOrTSTN[curInd * 2] = firstChildDes + 1;
		return InitialCompactBVHOnNode(node->childNodes[1], compact, curInd + firstChildDes + 1) + firstChildDes + 1;
	}

	bool CompactBVH::InitialCompactBVH(const BVHNode* bvhRoot)
	{
		if (!bvhRoot) return false;
		DELETE_ARRAY(bounds);
		DELETE_ARRAY(offOrTSTN);
		nodeN = bvhRoot->desN + 1;
		if (nodeN <= 0) return false;

		bounds = new NPRayHelper::AABBBox[nodeN];
		offOrTSTN = new uint32[nodeN * 2];
		if (InitialCompactBVHOnNode(bvhRoot, this, 0) == nodeN)
			return true;
		return false;
	}
}