#include "raytracer.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <SOIL.h>

#include "cudahelper.h"
#include "attrhelper.h"

#define RT_XGEN
#define DEFINE_RT(__name__, __codename__) \
	namespace __codename__{ \
	bool Render(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp \
	, float fov, RTScene* scene \
	, float width, float height, float* result); \
	void CleanMem(); \
	NPAttrHelper::Attrib* GetAttribute(unsigned __int32 ind, std::string &name); \
	unsigned __int32 GetAttribsN(); \
	}
#include "raytracer.h"
#undef DEFINE_RT(__name__, __codename__)
#undef RT_XGEN


bool RTScene::Trace(const NPRayHelper::Ray &r, HitResult& result)
{
	float minIntersect = M_INF;

	if (m_compactBVH.IsValid())
	{
		uint32 traceCmd[128];
		traceCmd[0] = 0;
		int32 traceCmdPointer = 0;
		while (traceCmdPointer >= 0)
		{
			uint32 curInd = traceCmd[traceCmdPointer--];
			float min = m_compactBVH.bounds[curInd].intersect(r);
			if (min >= 0 && min < minIntersect)
			{
				if (m_compactBVH.offTSTN[curInd * 3] != 0)
				{
					if (traceCmdPointer < 127)
						traceCmd[++traceCmdPointer] = curInd + 1;
					if (traceCmdPointer < 127)
						traceCmd[++traceCmdPointer] = curInd + m_compactBVH.offTSTN[curInd * 3];
				}
				else
				{
					uint32 triStart = m_compactBVH.offTSTN[curInd * 3 + 1];
					uint32 triN = m_compactBVH.offTSTN[curInd * 3 + 2];
					for (uint32 i = triStart; i < triStart + triN; i++)
					{
						NPRayHelper::Tri tri;
						tri.p0 = m_triIntersectData[i * 4];
						tri.p1 = m_triIntersectData[i * 4 + 1];
						tri.p2 = m_triIntersectData[i * 4 + 2];

						NPMathHelper::Vec3 pos, norm;
						float w, u, v;
						if (tri.intersect(r, pos, norm, w, u, v))
						{
							float dist = (pos - r.origPoint).length();
							if (dist < minIntersect)
							{
								minIntersect = dist;
								result.hitPosition = pos;
								result.hitNormal = NPMathHelper::Vec3(w, u, v);
								result.objId = i;
								result.objType = OBJ_TRI;
							}
						}
					}
				}
			}
		}
	}

	return (minIntersect < M_INF);
}

void AssimpProcessNode(RTScene* mainScene, aiNode* node, const aiScene* scene)
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
			tri.matInd = mesh->mMaterialIndex + mainScene->m_pMaterials.size();
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

bool LoadTexture(RTTexture &texture, const std::string &name)
{
	int width, height;
	unsigned char* image = SOIL_load_image(name.c_str(), &width, &height, 0, SOIL_LOAD_RGB);
	if (!image || width <= 0 || height <= 0)
		return false;
	if (texture.data)
		delete[] texture.data;
	texture.height = height;
	texture.width = width;
	texture.data = new float[width * height * 4];
	for (uint i = 0; i < width * height; i++)
	{
		texture.data[i * 4] = (float)image[i * 3] / 255.f;
		texture.data[i * 4 + 1] = (float)image[i * 3 + 1] / 255.f;
		texture.data[i * 4 + 2] = (float)image[i * 3 + 2] / 255.f;
		texture.data[i * 4 + 3] = 1.f;
	}
	return true;
}

int32 GetTextureIndex(RTScene* mainScene, const std::string &name)
{
	for (uint32 i = 0; i < mainScene->m_pTextures.size(); i++)
	{
		if (name.compare(mainScene->m_pTextures[i].first) == 0)
			return i;
	}

	RTTexture addTex;
	if (LoadTexture(addTex, name))
	{
		mainScene->m_pTextures.push_back(std::make_pair(name,addTex));
		return mainScene->m_pTextures.size() - 1;
	}

	return -1;
}

int32 AssimpLoadTexture(RTScene* mainScene, const aiMaterial* mat, const std::string& dir, aiTextureType type)
{
	if (mat->GetTextureCount(type) > 0)
	{
		aiString str;
		mat->GetTexture(type, 0, &str);
		std::string texPath = dir + "\\" + str.C_Str();
		return GetTextureIndex(mainScene, texPath);
	}
	return -1;
}


void AssimpProcessSceneMaterial(RTScene* mainScene, const aiScene* scene, std::string& dir)
{
	uint32 curMatN = mainScene->m_pMaterials.size();
	for (uint i = 0; i < scene->mNumMaterials; i++)
	{
		RTMaterial ourMat;
		aiMaterial* material = scene->mMaterials[i];
		aiColor3D aiColor(0.f, 0.f, 0.f);
		float aiFloat;
		if (AI_SUCCESS == material->Get(AI_MATKEY_COLOR_DIFFUSE, aiColor))
			ourMat.diffuse = NPMathHelper::Vec3(aiColor.r, aiColor.g, aiColor.b);
		if (AI_SUCCESS == material->Get(AI_MATKEY_COLOR_AMBIENT, aiColor))
			ourMat.emissive = NPMathHelper::Vec3(aiColor.r, aiColor.g, aiColor.b);
		if (AI_SUCCESS == material->Get(AI_MATKEY_COLOR_EMISSIVE, aiColor))
			ourMat.emissive = ourMat.emissive + NPMathHelper::Vec3(aiColor.r, aiColor.g, aiColor.b);
		if (AI_SUCCESS == material->Get(AI_MATKEY_OPACITY, aiFloat))
			ourMat.transparency = 1.f - aiFloat;
		ourMat.diffuseTexId = AssimpLoadTexture(mainScene, material, dir, aiTextureType_DIFFUSE);
		ourMat.normalTexId = AssimpLoadTexture(mainScene, material, dir, aiTextureType_NORMALS);
		ourMat.emissiveTexId = AssimpLoadTexture(mainScene, material, dir, aiTextureType_EMISSIVE);

		mainScene->m_pMaterials.push_back(ourMat);
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
	AssimpProcessSceneMaterial(this, scene, dir);
	updateTWBar();

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
	m_bvhRootNode.Clear();
	std::vector<uint32> reorderedTriOrder = NPBVHHelper::CreateBVH(&m_bvhRootNode, tris, verts);
	std::vector<RTTriangle> tempTriOrder(reorderedTriOrder.size());
	m_triIntersectData.clear();
	m_triIntersectData.resize(reorderedTriOrder.size()*4);
	m_triArea.clear();
	m_triArea.resize(reorderedTriOrder.size());
	for (uint32 i = 0; i < reorderedTriOrder.size(); i++)
	{
		tempTriOrder[i] = m_pTriangles[reorderedTriOrder[i]];
		m_triIntersectData[i * 4] = m_pVertices[tempTriOrder[i].vertInd0].pos;
		m_triIntersectData[i * 4 + 1] = m_pVertices[tempTriOrder[i].vertInd1].pos;
		m_triIntersectData[i * 4 + 2] = m_pVertices[tempTriOrder[i].vertInd2].pos;

		NPMathHelper::Vec3 edge1 = m_triIntersectData[i * 3 + 1] - m_triIntersectData[i * 3];
		NPMathHelper::Vec3 edge2 = m_triIntersectData[i * 3 + 2] - m_triIntersectData[i * 3];
		m_triIntersectData[i * 4 + 3] = edge1.cross(edge2).normalize();
		float edge1L = edge1.length();
		float edge2L = edge2.length();
		float edgeDot = edge1.dot(edge2);
		m_triArea[i] = 0.5f * sqrtf(edge1L * edge1L * edge2L * edge2L - edgeDot * edgeDot);
	}
	m_pTriangles = tempTriOrder;
	SetIsCudaDirty(true);
	return m_compactBVH.InitialCompactBVH(&m_bvhRootNode);
}

void RTScene::UpdateMaterialsDirtyFlag()
{
	if (!m_bIsCudaMaterialDirty)
	{
		if (m_pLastMaterials.size() != m_pMaterials.size())
		{
			m_bIsCudaMaterialDirty = true;
			m_pLastMaterials.clear();
			for (uint32 i = 0; i < m_pMaterials.size(); i++)
			{
				m_pLastMaterials.push_back(m_pMaterials[i]);
			}
		}
		else
		{
			for (uint32 i = 0; i < m_pMaterials.size(); i++)
			{
				if (m_pMaterials[i] != m_pLastMaterials[i])
				{
					m_pLastMaterials[i] = m_pMaterials[i];
					m_bIsCudaMaterialDirty = true;
				}
			}
		}
	}
}

void RTScene::updateTWBar()
{
	if (m_pMaterialBar)
	{
		TwRemoveAllVars(m_pMaterialBar);
	}
	else
	{
		m_pMaterialBar = TwNewBar("Material Setting");
	}

	for (uint32 i = 0; i < m_pMaterials.size(); i++)
	{
		std::string matName = "mat" + std::to_string(i);
		std::string matPara = "group='" + matName + "'";
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Diffuse" + matName).c_str(), TW_TYPE_COLOR3F, m_pMaterials[i].diffuse._e, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("EmissiveR" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].emissive._x, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("EmissiveG" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].emissive._y, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("EmissiveB" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].emissive._z, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Transparency" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].transparency, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Specular" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].specular, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Metallic" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].metallic, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Roughness" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].roughness, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Anisotropic" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].anisotropic, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Sheen" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].sheen, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("SheenTint" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].sheenTint, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("Clearcoat" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].clearcoat, matPara.c_str()));
		ATB_ASSERT(TwAddVarRW(m_pMaterialBar, ("ClearcoatGloss" + matName).c_str(), TW_TYPE_FLOAT, &m_pMaterials[i].clearcoatGloss, matPara.c_str()));
	}
}

RTRenderer::RTRenderer()
	: m_pResult(0)
	, m_pScene(0)
	, m_renderer(RENDERER_MODE_CPU_DEBUG)
	, m_pRenderBar(nullptr)
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

void RTRenderer::SetRendererMode(RENDERER_MODE mode)
{
	switch (m_renderer) {
	case RENDERER_MODE_CPU_DEBUG:
		break;
#define RT_XGEN
#define DEFINE_RT(__name__, __codename__) case RENDERER_MODE_##__codename__: \
	__codename__::CleanMem();\
	break;
#include "raytracer.h"
#undef DEFINE_RT(__name__, __codename__)
#undef RT_XGEN
	}
	m_renderer = mode;
	updateTWBar();
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
	scene.UpdateMaterialsDirtyFlag();

	switch(m_renderer) {
	case RENDERER_MODE_CPU_DEBUG:
		return renderCPU(camPos, camDir, camUp, fov, scene);
#define RT_XGEN
#define DEFINE_RT(__name__, __codename__) case RENDERER_MODE_##__codename__: \
	return __codename__::Render(camPos, camDir, camUp, fov, &scene, m_uSizeW, m_uSizeH, m_pResult);
#include "raytracer.h"
#undef DEFINE_RT(__name__, __codename__)
#undef RT_XGEN
	}
	return false;
}

bool RTRenderer::renderCPU(NPMathHelper::Vec3 camPos, NPMathHelper::Vec3 camDir, NPMathHelper::Vec3 camUp, float fov, RTScene &scene)
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
				NPMathHelper::Vec3 dir = (camRight * u + camUp * v + camDir).normalize();
				NPRayHelper::Ray ray(camPos, dir);

				RTScene::HitResult hitResult;
				if (scene.Trace(ray, hitResult))
				{
					assert(hitResult.objId >= 0);
					RTTriangle hitTri = scene.m_pTriangles[hitResult.objId];
					assert(hitTri.matInd >= 0);
					RTMaterial hitMat = scene.m_pMaterials[hitTri.matInd];
					m_pResult[ind] = hitMat.diffuse._x;
					m_pResult[ind + 1] = hitMat.diffuse._y;
					m_pResult[ind + 2] = hitMat.diffuse._z;
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

void RTRenderer::updateTWBar()
{
	if (m_pRenderBar)
	{
		TwRemoveAllVars(m_pRenderBar);
	}
	else
	{
		m_pRenderBar = TwNewBar("Renderer Setting");
	}

#define RT_XGEN
#define DEFINE_RT(__name__, __codename__) if (m_renderer == RENDERER_MODE_##__codename__)\
	{\
		for (uint32 i = 0; i < __codename__::GetAttribsN(); i++)\
		{\
			std::string valName;\
			NPAttrHelper::Attrib* att = __codename__::GetAttribute(i, valName);\
			if (att->attrType == NPAttrHelper::ATTR_BOOL)\
			{\
				ATB_ASSERT(TwAddVarRW(m_pRenderBar, valName.c_str(), TW_TYPE_BOOLCPP, att->GetBool(),\
					"group='Render Parameters'"));\
			}\
			else if (att->attrType == NPAttrHelper::ATTR_ENUM)\
			{\
				TwEnumVal* enumEV = new TwEnumVal[att->attrLength];\
				for (uint32 j = 0; j < att->attrLength; j++)\
				{\
					enumEV[j].Value = j;\
					enumEV[j].Label = att->attrEnumName[j];\
				}\
				TwType renderType = TwDefineEnum(valName.c_str(), enumEV, att->attrLength);\
				ATB_ASSERT(TwAddVarRW(m_pRenderBar, valName.c_str(), renderType, att->GetUint(),\
					"group='Render Parameters'"));\
				delete[] enumEV;\
			}\
			else if (att->attrType == NPAttrHelper::ATTR_FLOAT)\
			{\
				ATB_ASSERT(TwAddVarRW(m_pRenderBar, valName.c_str(), TW_TYPE_FLOAT, att->GetFloat(),\
					"group='Render Parameters'"));\
			}\
			else if (att->attrType == NPAttrHelper::ATTR_INT)\
			{\
				ATB_ASSERT(TwAddVarRW(m_pRenderBar, valName.c_str(), TW_TYPE_INT32, att->GetInt(),\
					"group='Render Parameters'"));\
			}\
			else if (att->attrType == NPAttrHelper::ATTR_STR)\
			{\
			\
			}\
			else if (att->attrType == NPAttrHelper::ATTR_UINT)\
			{\
				ATB_ASSERT(TwAddVarRW(m_pRenderBar, valName.c_str(), TW_TYPE_UINT32, att->GetUint(),\
					"group='Render Parameters'"));\
			}\
		}\
	}
#include "raytracer.h"
#undef DEFINE_RT(__name__, __codename__)
#undef RT_XGEN
}