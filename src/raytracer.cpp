#include "raytracer.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

int cuda_test(int a, int b);
int cuda_test2(NPCudaRayHelper::Scene* scene);

void cuda_pt(float3 camPos, float3 camDir, float3 camUp, float fov, NPCudaRayHelper::Scene* scene
	, float width, float height, float* result);


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

__host__ void AssimpProcessNode(RTScene* mainScene, aiNode* node, const aiScene* scene)
{
	for (uint i = 0; i < node->mNumMeshes; i++)
	{
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		uint indicesOffset = mainScene->m_pVertices.size();
		for (uint j = 0; j < mesh->mNumVertices; j++)
		{
			RTVertex vertex;
			vertex.pos._x = mesh->mVertices[j].x;
			vertex.pos._y = mesh->mVertices[j].y;
			vertex.pos._z = mesh->mVertices[j].z;
			if (mesh->mNormals)
			{
				vertex.norm._x = mesh->mNormals[j].x;
				vertex.norm._y = mesh->mNormals[j].y;
				vertex.norm._z = mesh->mNormals[j].z;
			}
			if (mesh->mTangents)
			{
				vertex.tan._x = mesh->mTangents[j].x;
				vertex.tan._y = mesh->mTangents[j].y;
				vertex.tan._z = mesh->mTangents[j].z;
			}
			if (mesh->mTextureCoords && mesh->mTextureCoords[0])
			{
				vertex.tex._x = mesh->mTextureCoords[0][j].x;
				vertex.tex._y = mesh->mTextureCoords[0][j].y;
			}
			mainScene->m_pVertices.push_back(vertex);
		}

		for (uint j = 0; j < mesh->mNumFaces; j++)
		{
			aiFace face = mesh->mFaces[j];
			RTTriangle tri;
			for (uint k = 0; k < face.mNumIndices; k++)
			{
				uint ind = k % 3;
				if (ind == 0)
					tri.vertInd0 = indicesOffset + face.mIndices[k];
				else if (ind == 1)
					tri.vertInd1 = indicesOffset + face.mIndices[k];
				else
				{
					tri.vertInd2 = indicesOffset + face.mIndices[k];
					mainScene->m_pTriangles.push_back(tri);
				}
			}
		}
	}

	for (uint i = 0; i < node->mNumChildren; i++)
	{
		AssimpProcessNode(mainScene, node->mChildren[i], scene);
	}
}

bool RTScene::AddModel(const char* filename)
{
	Assimp::Importer importer;
	std::string sPath = filename;
	const aiScene* scene = importer.ReadFile(sPath.c_str(), aiProcess_Triangulate | aiProcess_FlipUVs
		| aiProcess_CalcTangentSpace | aiProcess_GenNormals);
	if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		return false;
	}
	std::string dir = sPath.substr(0, sPath.find_last_of('\\'));
	AssimpProcessNode(this, scene->mRootNode, scene);

	std::vector<uint32> tris;
	for (auto &tri : m_pTriangles)
	{
		tris.push_back(tri.vertInd0);
		tris.push_back(tri.vertInd1);
		tris.push_back(tri.vertInd2);
	}
	std::vector<NPMathHelper::Vec3> verts;
	for (auto &vert : m_pVertices)
	{
		verts.push_back(vert.pos);
	}
	bvhRootNode.Clear();
	std::vector<uint32> reorderedTriOrder = NPBVHHelper::CreateBVH(&bvhRootNode, tris, verts);

	return true;
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
	return RenderCPU(camPos, camDir, camUp, fov, scene);
}

bool RTRenderer::RenderCUDA(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene)
{
	std::cout << "test cuda :" << cuda_test(1, 1) << std::endl;
	return true;
}

bool RTRenderer::RenderCPU(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene)
{
	NPMathHelper::Vec3 camRight = camDir.cross(camUp).normalize();
	camUp = camRight.cross(camDir).normalize();
	auto f = [&](const tbb::blocked_range2d< int, int >& range) {
		for (unsigned int i = range.cols().begin(); i < range.cols().end(); i++)
		{
			for (unsigned int j = range.rows().begin(); j < range.rows().end(); j++)
			{
				unsigned int ind = (i + j * m_uSizeW) * 3.f;
				float u = (2.f * ((float)i + 0.5f) / (float)m_uSizeW - 1.f) * tan(fov * 0.5f) * (float)m_uSizeW / (float)m_uSizeH;
				float v = (2.f * ((float)j + 0.5f) / (float)m_uSizeH - 1.f) * tan(fov * 0.5f);
				//float u = 2.f * ((float)i - 0.5f * m_uSizeW + 0.5f) / m_uSizeW * tan(fov * 0.5f);
				//float v = 2.f * ((float)j - 0.5f * m_uSizeH + 0.5f) / m_uSizeH * tan(fov * 0.5f) / (float)m_uSizeW * (float)m_uSizeH;
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
					m_pResult[ind] = dir._x;
					m_pResult[ind + 1] = dir._y;
					m_pResult[ind + 2] = dir._z;
				}
			}
		}
	};

	tbb::parallel_for(tbb::blocked_range2d< int, int >(0, m_uSizeH, 0, m_uSizeW), f);

	return true;
}

bool RTRenderer::Render2(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp
	, float fov, NPCudaRayHelper::Scene &scene)
{
	//std::cout << "test cuda :" << cuda_test2(&scene) << std::endl;
	cuda_pt(make_float3(camPos._x, camPos._y, camPos._z), make_float3(camDir._x, camDir._y, camDir._z),
		make_float3(camUp._x, camUp._y, camUp._z), fov, &scene, m_uSizeW, m_uSizeH, m_pResult);
	return true;
}
