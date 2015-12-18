#include "ModelViewWindow.h"

#include <iostream>
#include <string>
#include <sstream>
#include <SOIL.h>

#include "geohelper.h"
#include "oshelper.h"
#include "atbhelper.h"
#include "samplinghelper.h"

#define ITR_COUNT 1

namespace BRDFModel
{
	void Mesh::Draw(NPGLHelper::Effect &effect)
	{
		unsigned int diffuseNr = 1;
		unsigned int specularNr = 1;
		for (unsigned int i = 0; i < m_textures.size(); i++)
		{
			glActiveTexture(GL_TEXTURE1 + i);
			std::string texName = m_textures[i].name;
			if (m_textures[i].type == 0)
			{
				texName += std::to_string(diffuseNr++);
			}
			else
			{
				texName += std::to_string(specularNr++);
			}
			glBindTexture(GL_TEXTURE_2D, m_textures[i].id);
			effect.SetInt(texName.c_str(), i + 1);
		}
		glBindVertexArray(m_iVAO);
		glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}

	void Mesh::SetupMesh()
	{
		glGenVertexArrays(1, &m_iVAO);
		glGenBuffers(1, &m_iVBO);
		glGenBuffers(1, &m_iEBO);

		glBindVertexArray(m_iVAO);
		glBindBuffer(GL_ARRAY_BUFFER, m_iVBO);
		glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(Vertex), m_vertices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices.size() * sizeof(GLuint), m_indices.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, tangent));
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, texCoords));

		glBindVertexArray(0);

		std::vector<NPMathHelper::Vec3> points;
		for (auto &vertex : m_vertices)
		{
			points.push_back(vertex.position);
		}
		m_space = SphericalSpace::CalcSpaceFromPoints(points);
	}


	Model::Model()
	{

	}

	void Model::Draw(NPGLHelper::Effect &effect)
	{
		for (auto &mesh : m_meshes)
		{
			mesh->Draw(effect);
		}
	}

	bool Model::LoadModel(const char* path)
	{
		Assimp::Importer importer;
		std::string sPath = path;
		const aiScene* scene = importer.ReadFile(sPath.c_str(), aiProcess_Triangulate | aiProcess_FlipUVs 
			| aiProcess_CalcTangentSpace | aiProcess_GenNormals);
		if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			NPOSHelper::CreateMessageBox("Cannot load model.", "Model loading", NPOSHelper::MSGBOX_OK);
			return false;
		}
		m_sDirectory = sPath.substr(0, sPath.find_last_of('\\'));
		ProcessNode(scene->mRootNode, scene);

		return true;
	}

	void Model::ProcessNode(aiNode* node, const aiScene* scene)
	{
		for (unsigned int i = 0; i < node->mNumMeshes; i++)
		{
			Mesh *loadedMesh = ProcessMesh(scene->mMeshes[node->mMeshes[i]], scene);
			m_meshes.push_back(loadedMesh);

			if (i == 0)
				m_space = loadedMesh->m_space;
			else
				m_space = m_space.Merge(loadedMesh->m_space);
		}

		for (unsigned int i = 0; i < node->mNumChildren; i++)
		{
			ProcessNode(node->mChildren[i], scene);
		}
	}

	Mesh* Model::ProcessMesh(aiMesh* mesh, const aiScene* scene)
	{
		std::vector<Vertex> vertices;
		std::vector<GLuint> indices;
		std::vector<Texture> textures;

		for (unsigned int i = 0; i < mesh->mNumVertices; i++)
		{
			Vertex vertex;
			vertex.position._x = mesh->mVertices[i].x;
			vertex.position._y = mesh->mVertices[i].y;
			vertex.position._z = mesh->mVertices[i].z;
			vertex.normal._x = mesh->mNormals[i].x;
			vertex.normal._y = mesh->mNormals[i].y;
			vertex.normal._z = mesh->mNormals[i].z;
			vertex.tangent._x = mesh->mTangents[i].x;
			vertex.tangent._y = mesh->mTangents[i].y;
			vertex.tangent._z = mesh->mTangents[i].z;
			if (mesh->mTextureCoords[0])
			{
				vertex.texCoords._x = mesh->mTextureCoords[0][i].x;
				vertex.texCoords._y = mesh->mTextureCoords[0][i].y;
			}
			vertices.push_back(vertex);
		}

		for (unsigned int i = 0; i < mesh->mNumFaces; i++)
		{
			aiFace face = mesh->mFaces[i];
			for (unsigned int j = 0; j < face.mNumIndices; j++)
			{
				indices.push_back(face.mIndices[j]);
			}
		}

		if (mesh->mMaterialIndex >= 0)
		{
			aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
			std::vector<Texture> diffuseMaps = LoadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse", 0);
			textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());
			std::vector<Texture> specularMaps = LoadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular", 1);
			textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());
			
		}

		return new Mesh(vertices, indices, textures);

	}

	std::vector<Texture> Model::LoadMaterialTextures(aiMaterial* mat, aiTextureType type, const char* name, const unsigned int typeId)
	{
		std::vector<Texture> textures;
		for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
		{
			aiString str;
			mat->GetTexture(type, i, &str);
			Texture texture;
			std::string fullpath = m_sDirectory + "\\" + str.C_Str();
			if (NPGLHelper::loadTextureFromFile(fullpath.c_str(), texture.id, GL_CLAMP, GL_CLAMP, GL_LINEAR, GL_LINEAR))
			{
				texture.name = name;
				texture.type = typeId;
				textures.push_back(texture);
			}
		}

		return textures;
	}
}

void TW_CALL BrowseModelButton(void* window)
{
	ModelViewWindow* appWin = (ModelViewWindow*)window;
	if (appWin)
		appWin->OpenModelData();
}

void TW_CALL BrowseCubemapButton(void* content)
{
	ModelViewWindow::CUBEMAPLOADCMD* cmd = (ModelViewWindow::CUBEMAPLOADCMD*)content;
	if (!cmd)
		return;
	if (cmd && cmd->win)
	{
		cmd->win->SetCubemap(cmd->side);
	}
}

void TW_CALL LoadCubemapButton(void* window)
{
	ModelViewWindow* appWin = (ModelViewWindow*)window;
	if (appWin)
		appWin->LoadCubemap();
}

void TW_CALL SetRenderingMethodCallback(const void *value, void *clientData)
{
	ModelViewWindow* appWin = (ModelViewWindow*)clientData;
	if (appWin)
	{
		ModelViewWindow::RENDERINGMETHODS method = *((ModelViewWindow::RENDERINGMETHODS*) value);
		appWin->SetRenderingMethod(method);
	}
}
void TW_CALL GetRenderingMethodCallback(void *value, void *clientData)
{
	ModelViewWindow* appWin = (ModelViewWindow*)clientData;
	if (appWin)
	{
		*(ModelViewWindow::RENDERINGMETHODS*)value = appWin->GetRenderingMethod();
	}
}

void TW_CALL RecordingPreviewButton(void* window)
{
	ModelViewWindow* appWin = (ModelViewWindow*)window;
	if (appWin)
		appWin->SetRecording(ModelViewWindow::REC_PREVIEW);
}

void TW_CALL RecordingStartButton(void* window)
{
	ModelViewWindow* appWin = (ModelViewWindow*)window;
	if (appWin)
		appWin->SetRecording(ModelViewWindow::REC_RECORDING);
}

void TW_CALL RecordingStopButton(void* window)
{
	ModelViewWindow* appWin = (ModelViewWindow*)window;
	if (appWin)
		appWin->SetRecording(ModelViewWindow::REC_NONE);
}


const unsigned int ModelViewWindow::SHADOW_WIDTH = 2048;
const unsigned int ModelViewWindow::SHADOW_HEIGHT = 2048;

ModelViewWindow::ModelViewWindow(const char* name, const int sizeW, const int sizeH)
	: Window(name, sizeW, sizeH)
	, m_Cam(1.f, 0.f, M_PI * 0.25f)
	, m_fCamSenX(0.01f)
	, m_fCamSenY(.005f)
	, m_fInSenX(.001f)
	, m_fInSenY(0.0005f)
	, m_fInPitch(0.f)
	, m_fInYaw(M_PI*0.25f)
	, m_bIsCamRotate(false)
	, m_bIsInRotate(false)
	, m_fZoomMin(0.25f)
	, m_fZoomMax(20.0f)
	, m_fZoomSen(0.1f)
	, m_bIsLoadTexture(false)
	, m_sBRDFTextureName("None")
	, m_bIsLoadModel(false)
	, m_sModelName("None")
	, m_pModel(nullptr)
	, m_v3LightColor(1.f,1.f,1.f)
	, m_bIsWireFrame(false)
	, m_bIsSceneGUI(true)
	, m_v3ModelPos(0.f, 0.f, 0.f)
	, m_fModelScale(1.0f)
	, m_v3ModelRot()
	, m_fLightIntMultiplier(1.0f)
	, m_bIsEnvMapDirty(true)
	, m_bIsEnvMapLoaded(false)
	, m_uiEnvMap(0)
	, m_pSkyboxEffect(nullptr)
	, m_eRenderingMethod(RENDERINGMETHOD_NONE)
	, m_uiVAOQuad(0)
	, m_uiVBOQuad(0)
	, m_uiHDRFBO(0)
	, m_uiHDRCB(0)
	, m_uiHDRDB(0)
	, m_pFinalComposeEffect(nullptr)
	, m_pDiffuseModelEffect(nullptr)
	, m_pBlinnPhongModelEffect(nullptr)
	, m_pBlinnPhongNormalModelEffect(nullptr)
	, m_pDiffuseNormalModelEffect(nullptr)
	, m_pBRDFModelEffect(nullptr)
	, m_pDiffuseEnvModelEffect(nullptr)
	, m_pBlinnPhongEnvModelEffect(nullptr)
	, m_pBRDFEnvModelEffect(nullptr)
	, m_pDepthEffect(nullptr)
	, m_fExposure(1.f)
	, m_fEnvMapMultiplier(1.f)
	, m_bIsShowFloor(true)
	, m_iFloorTex(0)
	, m_iFloorNormalTex(0)
	, m_uiMaxSampling(1024)
	, m_fRenderingProgress(0.f)
	, m_fShadowBiasMin(0.005f)
	, m_fShadowBiasMax(0.05f)
	, m_uiEnvShadowMaxSamp(2048)
	, m_recStatus(REC_NONE)
	, m_fRecFPS(24)
	, m_fRecCirSec(5.0f)
	, m_uiRecCurFrame(0)
	, m_sRecStorePath("Recorded\\seq")
{
}

ModelViewWindow::~ModelViewWindow()
{
}

int ModelViewWindow::OnInit()
{
	////////////////////
	// ANT INIT - BGN //
	////////////////////
	CHECK_GL_ERROR;
	ATB_ASSERT(NPTwInit(m_uiID, TW_OPENGL_CORE, nullptr));
	ATB_ASSERT(TwSetCurrentWindow(m_uiID));
	ATB_ASSERT(TwWindowSize(m_iSizeW, m_iSizeH));
	TwBar* mainBar = TwNewBar("ModelView");
	ATB_ASSERT(TwDefine(" ModelView help='These properties defines the visual appearance of the model' "));

	ATB_ASSERT(TwAddVarRO(mainBar, "brdfname", TW_TYPE_STDSTRING, &m_sBRDFTextureName,
		" label='Loaded BRDF' help='Loaded BRDF' group='BRDF File'"));

	ATB_ASSERT(TwAddButton(mainBar, "openmodel", BrowseModelButton, this, "label='Browse File' group='Model'"));
	ATB_ASSERT(TwAddVarRO(mainBar, "modelname", TW_TYPE_STDSTRING, &m_sModelName,
		" label='Loaded Model' help='Loaded Model' group='Model'"));

	ATB_ASSERT(TwAddVarRW(mainBar, "PosX", TW_TYPE_FLOAT, &m_v3ModelPos._x,
		" label='Pos X' help='Model Translation' group='Model'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "PosY", TW_TYPE_FLOAT, &m_v3ModelPos._y,
		" label='Pos Y' help='Model Translation' group='Model'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "PosZ", TW_TYPE_FLOAT, &m_v3ModelPos._z,
		" label='Pos Z' help='Model Translation' group='Model'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Rotation", TW_TYPE_QUAT4F, &m_v3ModelRot._e[0],
		" label='Rotation' help='Model Rotation' group='Model' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Scale", TW_TYPE_FLOAT, &m_fModelScale,
		" label='Scale' help='Model Scale' group='Model' step=0.1"));

	ATB_ASSERT(TwAddVarRW(mainBar, "wireframe", TW_TYPE_BOOLCPP, &m_bIsWireFrame,
		" label='Wireframe' help='Show Wireframe' group='Display'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "scenegui", TW_TYPE_BOOLCPP, &m_bIsSceneGUI,
		" label='Scene GUI' help='Show Scene GUI' group='Display'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Exposure", TW_TYPE_FLOAT, &m_fExposure,
		" label='Exposure' help='View Exposure' group='Display' step=0.1"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Show Floor", TW_TYPE_BOOLCPP, &m_bIsShowFloor,
		" label='Show Floor' help='Show Floor' group='Display'"));
	
	TwEnumVal renderEV[] = { 
		{ RENDERINGMETHOD_DIFFUSEDIRLIGHT, "Diffuse DirLight" },
		{ RENDERINGMETHOD_BLINNPHONGDIRLIGHT, "Blinn-Phong DirLight" },
		{ RENDERINGMETHOD_BRDFDIRLIGHT, "BRDF DirLight" },
		{ RENDERINGMETHOD_DIFFUSEENVMAP, "Diffuse EnvMap" },
		{ RENDERINGMETHOD_BLINNPHONGENVMAP, "Blinn-Phong EnvMap" },
		{ RENDERINGMETHOD_BRDFENVMAP, "BRDF EnvMap" },
		{ RENDERINGMETHOD_DIFFUSEENVMAPS, "Diffuse EnvMap Shadow" },
		{ RENDERINGMETHOD_BLINNPHONGENVMAPS, "Blinn-Phong EnvMap Shadow" },
		{ RENDERINGMETHOD_BRDFENVMAPS, "BRDF EnvMap Shadow" }
	};
	TwType renderType = TwDefineEnum("Rendering Method", renderEV, RENDERINGMETHOD_N);
	TwAddVarCB(mainBar, "Rendering", renderType, SetRenderingMethodCallback, GetRenderingMethodCallback, this
		, " label='Rendering Method' help='Set Rendering Method' group='Display'");

	ATB_ASSERT(TwAddVarRO(mainBar, "Rendering Progress", TW_TYPE_FLOAT, &m_fRenderingProgress,
		"group='Display'"));

	ATB_ASSERT(TwAddVarRW(mainBar, "Light Ambient Color", TW_TYPE_COLOR3F, &m_dirLight.ambient, " group='Directional Light' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Light Diffuse Color", TW_TYPE_COLOR3F, &m_dirLight.diffuse, " group='Directional Light' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Light Specular Color", TW_TYPE_COLOR3F, &m_dirLight.specular, " group='Directional Light' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Color", TW_TYPE_COLOR3F, &m_v3LightColor, " group='Directional Light' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Intensity Multiplier", TW_TYPE_FLOAT, &m_fLightIntMultiplier,
		" label='Intensity Multiplier' help='Multiply light color' group='Directional Light' step=0.1"));
	//ATB_ASSERT(TwAddVarRW(mainBar, "Direction", TW_TYPE_DIR3F, &m_f3LightDir, " group='Directional Light' "));

	ATB_ASSERT(TwAddVarRW(mainBar, "Bias Min", TW_TYPE_FLOAT, &m_fShadowBiasMin,
		"group='Shadow Mapping'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Bias Max", TW_TYPE_FLOAT, &m_fShadowBiasMax,
		"group='Shadow Mapping'"));

	ATB_ASSERT(TwAddVarRW(mainBar, "Model Ambient Color", TW_TYPE_COLOR3F, &m_modelBlinnPhongMaterial.ambient, " group='Material' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Model Diffuse Color", TW_TYPE_COLOR3F, &m_modelBlinnPhongMaterial.diffuse, " group='Material' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Model Specular Color", TW_TYPE_COLOR3F, &m_modelBlinnPhongMaterial.specular, " group='Material' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Model Shininess", TW_TYPE_FLOAT, &m_modelBlinnPhongMaterial.shininess,
		"group='Material' step=0.1"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Model Forced Tangent", TW_TYPE_DIR3F, &m_v3ForcedTangent,
		"group='Material'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Enable Forced Tangent", TW_TYPE_BOOLCPP, &m_bIsForceTangent,
		"group='Material'"));


	std::string facename[] = {"Right", "Left", "Top", "Bottom", "Back", "Front"};
	for (unsigned int i = 0; i < 6; i++)
	{
		std::string bName = "openenvmap" + i;
		std::string vName = "envmaplabel" + i;
		std::string bPara = "label='Browse " + facename[i] + " Map' group='Environment Map'";
		std::string vPara = "label='" + facename[i] + " Map' group='Environment Map'";
		m_buttonInterfaceCmd[i].side = i;
		m_buttonInterfaceCmd[i].win = this;
		ATB_ASSERT(TwAddButton(mainBar, bName.c_str(), BrowseCubemapButton, m_buttonInterfaceCmd + i, bPara.c_str()));
		ATB_ASSERT(TwAddVarRW(mainBar, vName.c_str(), TW_TYPE_STDSTRING, &m_sEnvMapNames[i], vPara.c_str()));
	}
	ATB_ASSERT(TwAddSeparator(mainBar, "envmapsep", "group='Environment Map'"));
	ATB_ASSERT(TwAddVarRO(mainBar, "isDirty", TW_TYPE_BOOLCPP, &m_bIsEnvMapDirty,
		" label='Not loaded yet' help='Loaded Model' group='Environment Map'"));
	ATB_ASSERT(TwAddButton(mainBar, "loadenvmap"
		, LoadCubemapButton, this, "label='Load Map' group='Environment Map'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Env Multiplier", TW_TYPE_FLOAT, &m_fEnvMapMultiplier,
		" label='Multiplier' help='Multiply Environment map value' group='Environment Map'"));

	ATB_ASSERT(TwAddSeparator(mainBar, "instructionsep", ""));
	ATB_ASSERT(TwAddButton(mainBar, "instruction1", NULL, NULL, "label='LClick+Drag: Rot Light dir'"));
	ATB_ASSERT(TwAddButton(mainBar, "instruction2", NULL, NULL, "label='RClick+Drag: Rot Camera dir'"));
	ATB_ASSERT(TwAddButton(mainBar, "instruction3", NULL, NULL, "label='Scroll: Zoom Camera in/out'"));


	ATB_ASSERT(TwAddButton(mainBar, "Preview", RecordingPreviewButton, this, " group='Recording'"));
	ATB_ASSERT(TwAddButton(mainBar, "Start", RecordingStartButton, this, " group='Recording'"));
	ATB_ASSERT(TwAddButton(mainBar, "Stop", RecordingStopButton, this, " group='Recording'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Frame Per Sec", TW_TYPE_FLOAT, &m_fRecFPS, "group='Recording' step=1"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Circling Time", TW_TYPE_FLOAT, &m_fRecCirSec, "group='Recording' step=0.1"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Store Path", TW_TYPE_STDSTRING, &m_sRecStorePath, "group='Recording'"));

	CHECK_GL_ERROR;
	////////////////////
	// ANT INIT - END //
	////////////////////

	m_AxisLine[0].Init(m_pShareContent);
	m_AxisLine[1].Init(m_pShareContent);
	m_AxisLine[2].Init(m_pShareContent);
	m_InLine.Init(m_pShareContent);
	CHECK_GL_ERROR;
	m_pDiffuseModelEffect = m_pShareContent->GetEffect("DiffuseModelEffect");
	if (!m_pDiffuseModelEffect->GetIsLinked())
	{
		m_pDiffuseModelEffect->initEffect();
		m_pDiffuseModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pDiffuseModelEffect->attachShaderFromFile("..\\shader\\DiffuseModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pDiffuseModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBlinnPhongModelEffect = m_pShareContent->GetEffect("BlinnPhongModelEffect");
	if (!m_pBlinnPhongModelEffect->GetIsLinked())
	{
		m_pBlinnPhongModelEffect->initEffect();
		m_pBlinnPhongModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBlinnPhongModelEffect->attachShaderFromFile("..\\shader\\BlinnPhongModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBlinnPhongModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBlinnPhongNormalModelEffect = m_pShareContent->GetEffect("BlinnPhongNormalModelEffect");
	if (!m_pBlinnPhongNormalModelEffect->GetIsLinked())
	{
		m_pBlinnPhongNormalModelEffect->initEffect();
		m_pBlinnPhongNormalModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBlinnPhongNormalModelEffect->attachShaderFromFile("..\\shader\\BlinnPhongNormalModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBlinnPhongNormalModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pDiffuseNormalModelEffect = m_pShareContent->GetEffect("DiffuseNormalModelEffect");
	if (!m_pDiffuseNormalModelEffect->GetIsLinked())
	{
		m_pDiffuseNormalModelEffect->initEffect();
		m_pDiffuseNormalModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pDiffuseNormalModelEffect->attachShaderFromFile("..\\shader\\DiffuseNormalModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pDiffuseNormalModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBRDFModelEffect = m_pShareContent->GetEffect("BRDFModelEffect");
	if (!m_pBRDFModelEffect->GetIsLinked())
	{
		m_pBRDFModelEffect->initEffect();
		m_pBRDFModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBRDFModelEffect->attachShaderFromFile("..\\shader\\BRDFModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBRDFModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pDiffuseEnvModelEffect = m_pShareContent->GetEffect("DiffuseEnvModelEffect");
	if (!m_pDiffuseEnvModelEffect->GetIsLinked())
	{
		m_pDiffuseEnvModelEffect->initEffect();
		m_pDiffuseEnvModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pDiffuseEnvModelEffect->attachShaderFromFile("..\\shader\\DiffuseEnvModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pDiffuseEnvModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBlinnPhongEnvModelEffect = m_pShareContent->GetEffect("BlinnPhongEnvModelEffect");
	if (!m_pBlinnPhongEnvModelEffect->GetIsLinked())
	{
		m_pBlinnPhongEnvModelEffect->initEffect();
		m_pBlinnPhongEnvModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBlinnPhongEnvModelEffect->attachShaderFromFile("..\\shader\\BlinnPhongEnvModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBlinnPhongEnvModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBRDFEnvModelEffect = m_pShareContent->GetEffect("BRDFEnvModelEffect");
	if (!m_pBRDFEnvModelEffect->GetIsLinked())
	{
		m_pBRDFEnvModelEffect->initEffect();
		m_pBRDFEnvModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBRDFEnvModelEffect->attachShaderFromFile("..\\shader\\BRDFEnvModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBRDFEnvModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pDiffuseEnvSModelEffect = m_pShareContent->GetEffect("DiffuseEnvSModelEffect");
	if (!m_pDiffuseEnvSModelEffect->GetIsLinked())
	{
		m_pDiffuseEnvSModelEffect->initEffect();
		m_pDiffuseEnvSModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pDiffuseEnvSModelEffect->attachShaderFromFile("..\\shader\\DiffuseEnvSModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pDiffuseEnvSModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBlinnPhongEnvSModelEffect = m_pShareContent->GetEffect("BlinnPhongEnvSModelEffect");
	if (!m_pBlinnPhongEnvSModelEffect->GetIsLinked())
	{
		m_pBlinnPhongEnvSModelEffect->initEffect();
		m_pBlinnPhongEnvSModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBlinnPhongEnvSModelEffect->attachShaderFromFile("..\\shader\\BlinnPhongEnvSModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBlinnPhongEnvSModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBRDFEnvSModelEffect = m_pShareContent->GetEffect("BRDFEnvSModelEffect");
	if (!m_pBRDFEnvSModelEffect->GetIsLinked())
	{
		m_pBRDFEnvSModelEffect->initEffect();
		m_pBRDFEnvSModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBRDFEnvSModelEffect->attachShaderFromFile("..\\shader\\BRDFEnvSModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBRDFEnvSModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pBlinnPhongNormalEnvSModelEffect = m_pShareContent->GetEffect("BlinnPhongNormalEnvSModelEffect");
	if (!m_pBlinnPhongNormalEnvSModelEffect->GetIsLinked())
	{
		m_pBlinnPhongNormalEnvSModelEffect->initEffect();
		m_pBlinnPhongNormalEnvSModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pBlinnPhongNormalEnvSModelEffect->attachShaderFromFile("..\\shader\\BlinnPhongNormalEnvSModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pBlinnPhongNormalEnvSModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pDiffuseNormalEnvSModelEffect = m_pShareContent->GetEffect("DiffuseNormalEnvSModelEffect");
	if (!m_pDiffuseNormalEnvSModelEffect->GetIsLinked())
	{
		m_pDiffuseNormalEnvSModelEffect->initEffect();
		m_pDiffuseNormalEnvSModelEffect->attachShaderFromFile("..\\shader\\ModelVS.glsl", GL_VERTEX_SHADER);
		m_pDiffuseNormalEnvSModelEffect->attachShaderFromFile("..\\shader\\DiffuseNormalEnvSModelPS.glsl", GL_FRAGMENT_SHADER);
		m_pDiffuseNormalEnvSModelEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pSkyboxEffect = m_pShareContent->GetEffect("SkyboxEffect");
	if (!m_pSkyboxEffect->GetIsLinked())
	{
		m_pSkyboxEffect->initEffect();
		m_pSkyboxEffect->attachShaderFromFile("..\\shader\\SkyboxVS.glsl", GL_VERTEX_SHADER);
		m_pSkyboxEffect->attachShaderFromFile("..\\shader\\SkyboxPS.glsl", GL_FRAGMENT_SHADER);
		m_pSkyboxEffect->linkEffect();
	}
	CHECK_GL_ERROR;
	m_pDepthEffect = m_pShareContent->GetEffect("DepthEffect");
	if (!m_pDepthEffect->GetIsLinked())
	{
		m_pDepthEffect->initEffect();
		m_pDepthEffect->attachShaderFromFile("..\\shader\\DepthVS.glsl", GL_VERTEX_SHADER);
		m_pDepthEffect->attachShaderFromFile("..\\shader\\DepthPS.glsl", GL_FRAGMENT_SHADER);
		m_pDepthEffect->linkEffect();
	}
	CHECK_GL_ERROR;

	m_skybox.SetGeometry(NPGeoHelper::GetBoxShape(1.f, 1.f, 1.f));
	m_floor.SetGeometry(NPGeoHelper::GetFloorPlaneShape(10.f, 10.f, 10.f), 1);
	//m_floorMaterial.ambient = NPMathHelper::Vec3(1.f, 1.f, 1.f);
	//m_floorMaterial.diffuse = NPMathHelper::Vec3(1.f, 1.f, 1.f);
	//m_floorMaterial.specular = NPMathHelper::Vec3(1.f, 1.f, 1.f);
	m_floorMaterial.shininess = 50.f;
	m_floorSpace.m_fRadius = 10.f;

	m_dirLight.ambient = NPMathHelper::Vec3(0.1f, 0.1f, 0.1f);
	m_dirLight.diffuse = NPMathHelper::Vec3(1.f, 1.f, 1.f);
	m_dirLight.specular = NPMathHelper::Vec3(0.3f, 0.3f, 0.3f);

	m_modelBlinnPhongMaterial.ambient = NPMathHelper::Vec3(0.f, 0.f, 0.f);
	m_modelBlinnPhongMaterial.diffuse = NPMathHelper::Vec3(1.f, 1.f, 1.f);
	m_modelBlinnPhongMaterial.specular = NPMathHelper::Vec3(1.f, 1.f, 1.f);
	m_modelBlinnPhongMaterial.shininess = 25.f;

	NPGLHelper::loadTextureFromFile("..\\texture\\floor.png", m_iFloorTex, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR, false);
	NPGLHelper::loadTextureFromFile("..\\texture\\floor_nmap.png", m_iFloorNormalTex, GL_REPEAT, GL_REPEAT, GL_LINEAR, GL_LINEAR, false);

	CHECK_GL_ERROR;
	glGenFramebuffers(1, &m_uiHDRFBO);
	glGenTextures(1, &m_uiHDRCB);
	glBindTexture(GL_TEXTURE_2D, m_uiHDRCB);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_iSizeW, m_iSizeH, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenRenderbuffers(1, &m_uiHDRDB);
	glBindRenderbuffer(GL_RENDERBUFFER, m_uiHDRDB);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_iSizeW, m_iSizeH);
	glBindFramebuffer(GL_FRAMEBUFFER, m_uiHDRFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_uiHDRCB, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_uiHDRDB);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		DEBUG_COUT("[!!!]FRAMEBUFFER::CREATION_FAILED" << std::endl);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CHECK_GL_ERROR;
	glGenFramebuffers(1, &m_uiRECFBO);
	glGenTextures(1, &m_uiRECCB);
	glBindTexture(GL_TEXTURE_2D, m_uiRECCB);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_iSizeW, m_iSizeH, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glGenRenderbuffers(1, &m_uiRECDB);
	glBindRenderbuffer(GL_RENDERBUFFER, m_uiRECDB);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_iSizeW, m_iSizeH);
	glBindFramebuffer(GL_FRAMEBUFFER, m_uiRECFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_uiRECCB, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_uiRECDB);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		DEBUG_COUT("[!!!]FRAMEBUFFER::CREATION_FAILED" << std::endl);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CHECK_GL_ERROR;

	m_pFinalComposeEffect = m_pShareContent->GetEffect("FinalComposeEffect");
	if (!m_pFinalComposeEffect->GetIsLinked())
	{
		m_pFinalComposeEffect->initEffect();
		m_pFinalComposeEffect->attachShaderFromFile("..\\shader\\FinalComposeVS.glsl", GL_VERTEX_SHADER);
		m_pFinalComposeEffect->attachShaderFromFile("..\\shader\\FinalComposePS.glsl", GL_FRAGMENT_SHADER);
		m_pFinalComposeEffect->linkEffect();
	}

	// Shadow Mapping
	{
		glGenFramebuffers(1, &m_uiDepthMapFBO);
		glGenTextures(1, &m_uiDepthMapTex);
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		GLfloat borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
		glBindFramebuffer(GL_FRAMEBUFFER, m_uiDepthMapFBO);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_uiDepthMapTex, 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}


	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glDepthFunc(GL_LEQUAL);
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

	SetRenderingMethod(RENDERINGMETHOD_BRDFDIRLIGHT);

	return 0;
}

int ModelViewWindow::OnTick(const float deltaTime)
{
	// Camera control - bgn
	glm::vec2 cursorMoved = m_v2CurrentCursorPos - m_v2LastCursorPos;
	if (m_recStatus == REC_NONE)
	{
		if (m_bIsCamRotate)
		{
			m_Cam.AddPitch(-cursorMoved.x * m_fCamSenX);
			m_Cam.AddYaw(cursorMoved.y * m_fCamSenY);
		}
		if (m_bIsInRotate)
		{
			m_fInPitch = m_fInPitch + cursorMoved.x * m_fInSenX;
			m_fInYaw = (m_fInYaw - cursorMoved.y * m_fInSenY);
			if (m_fInYaw < 0) m_fInYaw = 0.f;
			if (m_fInYaw > M_PI * 0.5f) m_fInYaw = M_PI * 0.5f;
			while (m_fInPitch < 0) m_fInPitch = m_fInPitch + M_PI * 2.f;
			while (m_fInPitch > M_PI * 2.f) m_fInPitch -= M_PI * 2.f;
		}
		if (abs(m_fScrollY) > M_EPSILON)
		{
			float curZoom = m_Cam.GetRadius();
			curZoom += m_fScrollY * m_fZoomSen;
			curZoom = (curZoom < m_fZoomMin) ? m_fZoomMin : (curZoom > m_fZoomMax) ? m_fZoomMax : curZoom;
			m_Cam.SetRadius(curZoom);
			m_fScrollY = 0.f;
		}
	}

	m_v2LastCursorPos = m_v2CurrentCursorPos;
	// Camera control - end


	glBindFramebuffer(GL_FRAMEBUFFER, m_uiHDRFBO);
	switch (m_eRenderingMethod)
	{
	case RENDERINGMETHOD_DIFFUSEDIRLIGHT:
		RenderMethod_DiffuseDirLight();
		break;
	case RENDERINGMETHOD_BLINNPHONGDIRLIGHT:
		RenderMethod_BlinnPhongDirLight();
		break;
	case RENDERINGMETHOD_BRDFDIRLIGHT:
		RenderMethod_BRDFDirLight();
		break;
	case RENDERINGMETHOD_DIFFUSEENVMAP:
		RenderMethod_DiffuseEnvMap();
		break;
	case RENDERINGMETHOD_BLINNPHONGENVMAP:
		RenderMethod_BlinnPhongEnvMap();
		break;
	case RENDERINGMETHOD_BRDFENVMAP:
		RenderMethod_BRDFEnvMap();
		break;
	case RENDERINGMETHOD_BRDFENVMAPS:
		RenderMethod_BRDFEnvMapS();
		break;
	case RENDERINGMETHOD_DIFFUSEENVMAPS:
		RenderMethod_DiffuseEnvMapS();
		break;
	case RENDERINGMETHOD_BLINNPHONGENVMAPS:
		RenderMethod_BlinnPhongEnvMapS();
		break;
	}

	// Recording - BGN
	if (m_recStatus != REC_NONE)
	{
		bool isRenderingCompleted = (m_fRenderingProgress >= 100.f 
			|| m_eRenderingMethod == RENDERINGMETHOD_DIFFUSEDIRLIGHT
			|| m_eRenderingMethod == RENDERINGMETHOD_BLINNPHONGDIRLIGHT
			|| m_eRenderingMethod == RENDERINGMETHOD_BRDFDIRLIGHT
			|| m_recStatus == REC_PREVIEW);

		if (isRenderingCompleted)
		{
			if (m_recStatus == REC_RECORDING)
			{

				glBindFramebuffer(GL_FRAMEBUFFER, m_uiRECFBO);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				m_pFinalComposeEffect->activeEffect();
				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, m_uiHDRCB);
				m_pFinalComposeEffect->SetInt("hdrBuffer", 0);
				m_pFinalComposeEffect->SetFloat("exposure", m_fExposure);
				RenderScreenQuad();
				glBindTexture(GL_TEXTURE_2D, 0);
				m_pFinalComposeEffect->deactiveEffect();

				float nextFrameTime = 1.f / m_fRecFPS;
				m_Cam.AddPitch(2.f * M_PI * nextFrameTime / m_fRecCirSec);
				std::stringstream storedPathSS;
				std::stringstream maxNumFrameSS;
				std::string maxNumFrameS;
				std::stringstream curNumFrameSS;
				std::string curNumFrameS;

				int maxFrame = m_fRecCirSec * m_fRecFPS;
				maxNumFrameSS << maxFrame;
				maxNumFrameSS >> maxNumFrameS;
				int maxDigit = maxNumFrameS.length();
				curNumFrameSS << m_uiRecCurFrame;
				curNumFrameSS >> curNumFrameS;
				int curDigit = curNumFrameS.length();
				storedPathSS << m_sRecStorePath;
				for (int i = curDigit; i != maxDigit; i++)
				{
					storedPathSS << "0";
				}
				storedPathSS << m_uiRecCurFrame << ".bmp";
				std::string storedPathS;
				storedPathSS >> storedPathS;

				NPGLHelper::saveScreenShotBMP(storedPathS.c_str(), m_iSizeW, m_iSizeH);
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
			}
			else
			{
				m_Cam.AddPitch(2.f * M_PI * deltaTime / m_fRecCirSec);
			}

			float completion = ((float)m_uiRecCurFrame + 1.f) / (m_fRecCirSec * m_fRecFPS);
			if (completion >= 1)
			{
				SetRecording(REC_NONE);
			}
			else
			{
				m_uiRecCurFrame = (m_recStatus == REC_RECORDING) ? m_uiRecCurFrame + 1 : m_uiRecCurFrame + (deltaTime * m_fRecFPS);
			}
		}
	}
	// Recording - END

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	m_pFinalComposeEffect->activeEffect();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_uiHDRCB);
	m_pFinalComposeEffect->SetInt("hdrBuffer", 0);
	m_pFinalComposeEffect->SetFloat("exposure", m_fExposure);
	RenderScreenQuad();
	glBindTexture(GL_TEXTURE_2D, 0);
	m_pFinalComposeEffect->deactiveEffect();

	glClear(GL_DEPTH_BUFFER_BIT);
	NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);

	if (m_bIsSceneGUI)
	{
		m_AxisLine[0].Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(1.f, 0.f, 0.f), NPMathHelper::Vec3(1.0f, 0.f, 0.f)
			, m_Cam.GetViewMatrix(), myProj.GetDataColumnMajor());
		m_AxisLine[1].Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(0.f, 1.f, 0.f), NPMathHelper::Vec3(0.0f, 1.f, 0.f)
			, m_Cam.GetViewMatrix(), myProj.GetDataColumnMajor());
		m_AxisLine[2].Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(0.f, 0.f, 1.f), NPMathHelper::Vec3(0.0f, 0.f, 1.f)
			, m_Cam.GetViewMatrix(), myProj.GetDataColumnMajor());
	}

	if (m_bIsSceneGUI)
	{
		glm::vec3 InLineEnd;
		InLineEnd.y = sin(m_fInYaw) * 10.f;
		InLineEnd.x = cos(m_fInYaw) * sin(m_fInPitch) * 10.f;
		InLineEnd.z = cos(m_fInYaw) * cos(m_fInPitch) * 10.f;
		m_InLine.Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(InLineEnd.x, InLineEnd.y, InLineEnd.z), NPMathHelper::Vec3(1.0f, 1.f, 1.f)
			, m_Cam.GetViewMatrix(), myProj.GetDataColumnMajor());
	}

	ATB_ASSERT(TwSetCurrentWindow(m_uiID));
	ATB_ASSERT(TwDraw());

	glfwSwapBuffers(GetGLFWWindow());

	return 0;
}

void ModelViewWindow::OnTerminate()
{
	NPTwTerminate(m_uiID);
}

void ModelViewWindow::OnHandleInputMSG(const INPUTMSG &msg)
{
	ATB_ASSERT(TwSetCurrentWindow(m_uiID));
	switch (msg.type)
	{
	case Window::INPUTMSG_KEYBOARDKEY:
		if (TwEventCharGLFW(msg.key, msg.action))
			break;
		if (msg.key == GLFW_KEY_ESCAPE && msg.action == GLFW_PRESS)
			glfwSetWindowShouldClose(m_pWindow, GL_TRUE);
		//if (msg.key == GLFW_KEY_O && msg.action == GLFW_PRESS)
		//	OpenModelData();
		break;
	case Window::INPUTMSG_MOUSEKEY:
		if (TwEventMouseButtonGLFW(msg.key, msg.action))
			break;
		if (msg.key == GLFW_MOUSE_BUTTON_RIGHT)
		{
			m_bIsCamRotate = (msg.action == GLFW_PRESS);
		}
		if (msg.key == GLFW_MOUSE_BUTTON_LEFT)
		{
			m_bIsInRotate = (msg.action == GLFW_PRESS);
		}
		break;
	case Window::INPUTMSG_MOUSECURSOR:
		TwEventMousePosGLFW(msg.xpos, msg.ypos);
		m_v2CurrentCursorPos.x = msg.xpos;
		m_v2CurrentCursorPos.y = msg.ypos;
		break;
	case Window::INPUTMSG_MOUSESCROLL:
		m_iScrollingTemp += msg.yoffset;
		if (TwEventMouseWheelGLFW(m_iScrollingTemp))
			break;
		m_fScrollY = msg.yoffset;
		break;
	}
}

void ModelViewWindow::OpenModelData()
{
	std::string file = NPOSHelper::BrowseFile("All\0*.*\0Text\0*.TXT\0");
	if (file.empty())
		return;

	if (m_pModel)
	{
		delete m_pModel;
		m_pModel = NULL;
	}

	m_pModel = new BRDFModel::Model();
	if (!m_pModel->LoadModel(file.c_str()))
	{
		std::string message = "Cannot load file ";
		message = message + file;
		NPOSHelper::CreateMessageBox(message.c_str(), "Load BRDF Data Failure", NPOSHelper::MSGBOX_OK);
		return;
	}
	m_sModelName = file.c_str();
	m_bIsLoadModel = true;
}

void ModelViewWindow::SetBRDFData(const char* path, unsigned int n_th, unsigned int n_ph)
{
	m_bIsBRDFUpdated = false;
	m_sNewBRDFPath = path;
	m_uiNewTH = n_th;
	m_uiNewPH = n_ph;
}


void ModelViewWindow::SetCubemap(unsigned int side)
{
	assert(side < 6);
	std::string file = NPOSHelper::BrowseFile("All\0*.*\0Text\0*.TXT\0");
	if (file.empty())
		return;
	m_bIsEnvMapDirty = true;
	m_sEnvMapNames[side] = file;
}

void ModelViewWindow::LoadCubemap()
{
	for (auto &facename : m_sEnvMapNames)
		if (facename.size() <= 0)
			return;

	if (m_bIsEnvMapLoaded)
		glDeleteTextures(1, &m_uiEnvMap);

	NPGLHelper::loadCubemapFromFiles(m_sEnvMapNames, m_uiEnvMap);
	m_bIsEnvMapDirty = false;
	m_bIsEnvMapLoaded = true;
}

void ModelViewWindow::SetRenderingMethod(RENDERINGMETHODS method)
{
	switch (m_eRenderingMethod)
	{
	case RENDERINGMETHOD_DIFFUSEDIRLIGHT:
		RenderMethod_DiffuseDirLightQuit();
		break;
	case RENDERINGMETHOD_BLINNPHONGDIRLIGHT:
		RenderMethod_BlinnPhongDirLightQuit();
		break;
	case RENDERINGMETHOD_BRDFDIRLIGHT:
		RenderMethod_BRDFDirLightQuit();
		break;
	case RENDERINGMETHOD_DIFFUSEENVMAP:
		RenderMethod_DiffuseEnvMapQuit();
		break;
	case RENDERINGMETHOD_BLINNPHONGENVMAP:
		RenderMethod_BlinnPhongEnvMapQuit();
		break;
	case RENDERINGMETHOD_BRDFENVMAP:
		RenderMethod_BRDFEnvMapQuit();
		break;
	case RENDERINGMETHOD_BRDFENVMAPS:
		RenderMethod_BRDFEnvMapSQuit();
		break;
	case RENDERINGMETHOD_DIFFUSEENVMAPS:
		RenderMethod_DiffuseEnvMapSQuit();
		break;
	case RENDERINGMETHOD_BLINNPHONGENVMAPS:
		RenderMethod_BlinnPhongEnvMapSQuit();
		break;
	}

	m_eRenderingMethod = method;

	switch (m_eRenderingMethod)
	{
	case RENDERINGMETHOD_DIFFUSEDIRLIGHT:
		RenderMethod_DiffuseDirLightInit();
		break;
	case RENDERINGMETHOD_BLINNPHONGDIRLIGHT:
		RenderMethod_BlinnPhongDirLightInit();
		break;
	case RENDERINGMETHOD_BRDFDIRLIGHT:
		RenderMethod_BRDFDirLightInit();
		break;
	case RENDERINGMETHOD_DIFFUSEENVMAP:
		RenderMethod_DiffuseEnvMapInit();
		break;
	case RENDERINGMETHOD_BLINNPHONGENVMAP:
		RenderMethod_BlinnPhongEnvMapInit();
		break;
	case RENDERINGMETHOD_BRDFENVMAP:
		RenderMethod_BRDFEnvMapInit();
		break;
	case RENDERINGMETHOD_DIFFUSEENVMAPS:
		RenderMethod_DiffuseEnvMapSInit();
		break;
	case RENDERINGMETHOD_BLINNPHONGENVMAPS:
		RenderMethod_BlinnPhongEnvMapSInit();
		break;
	case RENDERINGMETHOD_BRDFENVMAPS:
		RenderMethod_BRDFEnvMapSInit();
		break;
	}
}

void ModelViewWindow::SetRecording(const RECORD_STATUS status)
{
	switch (m_recStatus)
	{
	case REC_NONE:
		break;
	case REC_PREVIEW:
		break;
	case REC_RECORDING:
		break;
	}
	m_uiRecCurFrame = 0;
	m_recStatus = status;
}

void ModelViewWindow::UpdateBRDFData()
{
	if (!m_bIsBRDFUpdated)
	{
		m_bIsBRDFUpdated = true;

		if (m_bIsLoadTexture)
		{
			glDeleteTextures(1, &m_iBRDFEstTex);
			m_bIsLoadTexture = false;
		}

		if (m_sNewBRDFPath.length() <= 0)
		{
			m_bIsLoadTexture = false;
			return;
		}

		int width, height;
		//if (!NPGLHelper::loadTextureFromFile(m_sNewBRDFPath.c_str(), m_iBRDFEstTex, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST, false))
		if (!NPGLHelper::loadHDRTextureFromFile(m_sNewBRDFPath.c_str(), m_iBRDFEstTex, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST))
		{
			std::string message = "Cannot load file ";
			message = message + m_sNewBRDFPath;
			NPOSHelper::CreateMessageBox(message.c_str(), "Load BRDF Data Failure", NPOSHelper::MSGBOX_OK);
			return;
		}
		m_uiNTH = m_uiNewTH;
		m_uiNPH = m_uiNewPH;
		m_bIsLoadTexture = true;
		m_sBRDFTextureName = m_sNewBRDFPath;
	}
}

void ModelViewWindow::RenderMethod_DiffuseDirLight()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Depth rendering
	if (m_pModel)
	{
		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);
		Render_ShadowMap(m_dirLight.dir);
	}

	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	UpdateBRDFData();
	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
			, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
			, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pDiffuseModelEffect->activeEffect();
		m_pDiffuseModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pDiffuseModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pDiffuseModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pDiffuseModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());

		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);

		m_pDiffuseModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient);
		m_pDiffuseModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse);
		m_pDiffuseModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular);
		m_pDiffuseModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess);

		m_pDiffuseModelEffect->SetVec3("light.ambient", m_dirLight.ambient * m_fLightIntMultiplier);
		m_pDiffuseModelEffect->SetVec3("light.diffuse", m_dirLight.diffuse * m_fLightIntMultiplier);
		m_pDiffuseModelEffect->SetVec3("light.specular", m_dirLight.specular * m_fLightIntMultiplier);
		m_pDiffuseModelEffect->SetVec3("light.dir", m_dirLight.dir);

		m_pDiffuseModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		m_pDiffuseModelEffect->SetFloat("biasMin", m_fShadowBiasMin); CHECK_GL_ERROR;
		m_pDiffuseModelEffect->SetFloat("biasMax", m_fShadowBiasMax); CHECK_GL_ERROR;
		m_pDiffuseModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pDiffuseModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		m_pModel->Draw(*m_pDiffuseModelEffect);


		m_pDiffuseModelEffect->deactiveEffect();
	}

	if (m_bIsShowFloor)
	{

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::Identity();
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));

		m_pDiffuseNormalModelEffect->activeEffect();
		m_pDiffuseNormalModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pDiffuseNormalModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pDiffuseNormalModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pDiffuseNormalModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());

		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);

		m_pDiffuseNormalModelEffect->SetVec3("material.ambient", m_floorMaterial.ambient);
		m_pDiffuseNormalModelEffect->SetVec3("material.diffuse", m_floorMaterial.diffuse);
		m_pDiffuseNormalModelEffect->SetVec3("material.specular", m_floorMaterial.specular);
		m_pDiffuseNormalModelEffect->SetFloat("material.shininess", m_floorMaterial.shininess);

		m_pDiffuseNormalModelEffect->SetVec3("light.ambient", m_dirLight.ambient * m_fLightIntMultiplier);
		m_pDiffuseNormalModelEffect->SetVec3("light.diffuse", m_dirLight.diffuse * m_fLightIntMultiplier);
		m_pDiffuseNormalModelEffect->SetVec3("light.specular", m_dirLight.specular * m_fLightIntMultiplier);
		m_pDiffuseNormalModelEffect->SetVec3("light.dir", m_dirLight.dir);

		m_pDiffuseNormalModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		m_pDiffuseNormalModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pDiffuseNormalModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pDiffuseNormalModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pDiffuseNormalModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		glActiveTexture(GL_TEXTURE0); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_iFloorTex); CHECK_GL_ERROR;
		m_pDiffuseNormalModelEffect->SetInt("texture_diffuse1", 0); CHECK_GL_ERROR;

		glActiveTexture(GL_TEXTURE1); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_iFloorNormalTex); CHECK_GL_ERROR;
		m_pDiffuseNormalModelEffect->SetInt("texture_normal1", 1); CHECK_GL_ERROR;

		// Draw Floor Plane
		NPMathHelper::Mat4x4 floorModelMat = NPMathHelper::Mat4x4::Identity();
		NPMathHelper::Mat4x4 floorTranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(floorModelMat));
		m_pDiffuseNormalModelEffect->SetMatrix("model", floorModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pDiffuseNormalModelEffect->SetMatrix("tranInvModel", floorTranInvModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		glBindVertexArray(m_floor.GetVAO()); CHECK_GL_ERROR;
		glDrawElements(GL_TRIANGLES, m_floor.GetIndicesSize(), GL_UNSIGNED_INT, 0); CHECK_GL_ERROR;
		glBindVertexArray(0);

		m_pDiffuseNormalModelEffect->deactiveEffect();
	}

	if (m_bIsWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void ModelViewWindow::RenderMethod_BlinnPhongDirLight()
{
	m_fRenderingProgress = 100.0f;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Depth rendering
	if (m_pModel)
	{
		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);
		Render_ShadowMap(m_dirLight.dir);
	}

	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	UpdateBRDFData();
	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
			, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
			, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pBlinnPhongModelEffect->activeEffect();
		m_pBlinnPhongModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBlinnPhongModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBlinnPhongModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pBlinnPhongModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());

		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);

		m_pBlinnPhongModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient);
		m_pBlinnPhongModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse);
		m_pBlinnPhongModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular);
		m_pBlinnPhongModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess);

		m_pBlinnPhongModelEffect->SetVec3("light.ambient", m_dirLight.ambient * m_fLightIntMultiplier);
		m_pBlinnPhongModelEffect->SetVec3("light.diffuse", m_dirLight.diffuse * m_fLightIntMultiplier);
		m_pBlinnPhongModelEffect->SetVec3("light.specular", m_dirLight.specular * m_fLightIntMultiplier);
		m_pBlinnPhongModelEffect->SetVec3("light.dir", m_dirLight.dir);

		m_pBlinnPhongModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		m_pBlinnPhongModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pBlinnPhongModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pBlinnPhongModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pBlinnPhongModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		m_pModel->Draw(*m_pBlinnPhongModelEffect);


		m_pBlinnPhongModelEffect->deactiveEffect();
	}

	if (m_bIsShowFloor)
	{

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::Identity();
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));

		m_pBlinnPhongNormalModelEffect->activeEffect();
		m_pBlinnPhongNormalModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBlinnPhongNormalModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBlinnPhongNormalModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pBlinnPhongNormalModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());

		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);

		m_pBlinnPhongNormalModelEffect->SetVec3("material.ambient", m_floorMaterial.ambient);
		m_pBlinnPhongNormalModelEffect->SetVec3("material.diffuse", m_floorMaterial.diffuse);
		m_pBlinnPhongNormalModelEffect->SetVec3("material.specular", m_floorMaterial.specular);
		m_pBlinnPhongNormalModelEffect->SetFloat("material.shininess", m_floorMaterial.shininess);

		m_pBlinnPhongNormalModelEffect->SetVec3("light.ambient", m_dirLight.ambient * m_fLightIntMultiplier);
		m_pBlinnPhongNormalModelEffect->SetVec3("light.diffuse", m_dirLight.diffuse * m_fLightIntMultiplier);
		m_pBlinnPhongNormalModelEffect->SetVec3("light.specular", m_dirLight.specular * m_fLightIntMultiplier);
		m_pBlinnPhongNormalModelEffect->SetVec3("light.dir", m_dirLight.dir);

		m_pBlinnPhongNormalModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		m_pBlinnPhongNormalModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pBlinnPhongNormalModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pBlinnPhongNormalModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		glActiveTexture(GL_TEXTURE0); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_iFloorTex); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetInt("texture_diffuse1", 0); CHECK_GL_ERROR;

		glActiveTexture(GL_TEXTURE1); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_iFloorNormalTex); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetInt("texture_normal1", 1); CHECK_GL_ERROR;

		// Draw Floor Plane
		NPMathHelper::Mat4x4 floorModelMat = NPMathHelper::Mat4x4::Identity();
		NPMathHelper::Mat4x4 floorTranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(floorModelMat));
		m_pBlinnPhongNormalModelEffect->SetMatrix("model", floorModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetMatrix("tranInvModel", floorTranInvModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		glBindVertexArray(m_floor.GetVAO()); CHECK_GL_ERROR;
		glDrawElements(GL_TRIANGLES, m_floor.GetIndicesSize(), GL_UNSIGNED_INT, 0); CHECK_GL_ERROR;
		glBindVertexArray(0);

		m_pBlinnPhongNormalModelEffect->deactiveEffect();
	}

	if (m_bIsWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void ModelViewWindow::RenderMethod_BRDFDirLight()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Depth rendering
	if (m_pModel)
	{
		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);
		Render_ShadowMap(m_dirLight.dir);
	}

	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	UpdateBRDFData();
	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
			, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
			, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pBRDFModelEffect->activeEffect();
		m_pBRDFModelEffect->SetInt("n_th", m_uiNTH);
		m_pBRDFModelEffect->SetInt("n_ph", m_uiNPH);
		m_pBRDFModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBRDFModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBRDFModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pBRDFModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());

		m_pBRDFModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient); CHECK_GL_ERROR;
		m_pBRDFModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse); CHECK_GL_ERROR;
		m_pBRDFModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular); CHECK_GL_ERROR;
		m_pBRDFModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess); CHECK_GL_ERROR;

		glm::vec3 lightDir;
		lightDir.y = -sin(m_fInYaw);
		lightDir.x = -cos(m_fInYaw) * sin(m_fInPitch);
		lightDir.z = -cos(m_fInYaw) * cos(m_fInPitch);

		if (m_bIsForceTangent)
		{
			m_pBRDFModelEffect->SetVec3("forced_tangent_w", m_v3ForcedTangent);
		}

		m_pBRDFModelEffect->SetVec3("lightDir", lightDir.x, lightDir.y, lightDir.z);
		m_pBRDFModelEffect->SetVec3("lightColor", m_v3LightColor.x * m_fLightIntMultiplier
			, m_v3LightColor.y * m_fLightIntMultiplier, m_v3LightColor.z * m_fLightIntMultiplier);
		m_pBRDFModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		m_pBRDFModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pBRDFModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pBRDFModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pBRDFModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		if (m_bIsLoadTexture)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m_iBRDFEstTex);
			m_pBRDFModelEffect->SetInt("texture_brdf", 0);
		}

		m_pModel->Draw(*m_pBRDFModelEffect);
		m_pBRDFModelEffect->deactiveEffect();
	}

	if (m_bIsShowFloor)
	{

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::Identity();
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));

		m_pBlinnPhongNormalModelEffect->activeEffect();
		m_pBlinnPhongNormalModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBlinnPhongNormalModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBlinnPhongNormalModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pBlinnPhongNormalModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());

		m_dirLight.dir._y = -sin(m_fInYaw);
		m_dirLight.dir._x = -cos(m_fInYaw) * sin(m_fInPitch);
		m_dirLight.dir._z = -cos(m_fInYaw) * cos(m_fInPitch);

		m_pBlinnPhongNormalModelEffect->SetVec3("material.ambient", m_floorMaterial.ambient);
		m_pBlinnPhongNormalModelEffect->SetVec3("material.diffuse", m_floorMaterial.diffuse);
		m_pBlinnPhongNormalModelEffect->SetVec3("material.specular", m_floorMaterial.specular);
		m_pBlinnPhongNormalModelEffect->SetFloat("material.shininess", m_floorMaterial.shininess);

		m_pBlinnPhongNormalModelEffect->SetVec3("light.ambient", m_dirLight.ambient * m_fLightIntMultiplier);
		m_pBlinnPhongNormalModelEffect->SetVec3("light.diffuse", m_dirLight.diffuse * m_fLightIntMultiplier);
		m_pBlinnPhongNormalModelEffect->SetVec3("light.specular", m_dirLight.specular * m_fLightIntMultiplier);
		m_pBlinnPhongNormalModelEffect->SetVec3("light.dir", m_dirLight.dir);

		m_pBlinnPhongNormalModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		m_pBlinnPhongNormalModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pBlinnPhongNormalModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pBlinnPhongNormalModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		glActiveTexture(GL_TEXTURE0); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_iFloorTex); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetInt("texture_diffuse1", 0); CHECK_GL_ERROR;

		glActiveTexture(GL_TEXTURE1); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_iFloorNormalTex); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetInt("texture_normal1", 1); CHECK_GL_ERROR;

		// Draw Floor Plane
		NPMathHelper::Mat4x4 floorModelMat = NPMathHelper::Mat4x4::Identity();
		NPMathHelper::Mat4x4 floorTranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(floorModelMat));
		m_pBlinnPhongNormalModelEffect->SetMatrix("model", floorModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pBlinnPhongNormalModelEffect->SetMatrix("tranInvModel", floorTranInvModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		glBindVertexArray(m_floor.GetVAO()); CHECK_GL_ERROR;
		glDrawElements(GL_TRIANGLES, m_floor.GetIndicesSize(), GL_UNSIGNED_INT, 0); CHECK_GL_ERROR;
		glBindVertexArray(0);

		m_pBlinnPhongNormalModelEffect->deactiveEffect();
	}

	if (m_bIsWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void ModelViewWindow::RenderMethod_DiffuseEnvMap()
{
	if (NPMathHelper::Mat4x4(m_Cam.GetViewMatrix()) != m_matLastCam)
	{
		m_matLastCam = NPMathHelper::Mat4x4(m_Cam.GetViewMatrix());
		m_uiEnvInitSamp = 0;
	}

	NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
		, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
		, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));

	if (modelMat != m_matLastModel)
	{
		m_matLastModel = modelMat;
		m_uiEnvInitSamp = 0;
	}

	if (m_uiEnvInitSamp + ITR_COUNT > m_uiMaxSampling)
		return;

	if (m_uiEnvInitSamp <= 0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		if (m_uiEnvInitSamp > 0)
		{
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pDiffuseEnvModelEffect->activeEffect(); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetMatrix("view", m_Cam.GetViewMatrix()); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetInt("init_samp", m_uiEnvInitSamp); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetInt("max_samp", m_uiMaxSampling); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier); CHECK_GL_ERROR;

		glm::vec3 lightDir;
		lightDir.y = -sin(m_fInYaw);
		lightDir.x = -cos(m_fInYaw) * sin(m_fInPitch);
		lightDir.z = -cos(m_fInYaw) * cos(m_fInPitch);
		m_pDiffuseEnvModelEffect->SetVec3("viewPos", m_Cam.GetPos()); CHECK_GL_ERROR;

		m_pDiffuseEnvModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular); CHECK_GL_ERROR;
		m_pDiffuseEnvModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess); CHECK_GL_ERROR;

		if (m_bIsEnvMapLoaded)
		{
			glActiveTexture(GL_TEXTURE4); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap); CHECK_GL_ERROR;
			m_pDiffuseEnvModelEffect->SetInt("envmap", 4); CHECK_GL_ERROR;
		}

		m_pModel->Draw(*m_pDiffuseEnvModelEffect); CHECK_GL_ERROR;
		m_uiEnvInitSamp += ITR_COUNT;
		m_fRenderingProgress = (float)m_uiEnvInitSamp / (float)m_uiMaxSampling * 100.f;

		m_pDiffuseEnvModelEffect->deactiveEffect();

		glDisable(GL_BLEND);
	}

	if (m_bIsEnvMapLoaded)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		glCullFace(GL_FRONT);
		NPMathHelper::Mat4x4 noTranCamMath = m_Cam.GetViewMatrix();
		noTranCamMath._03 = noTranCamMath._13 = noTranCamMath._23 = 0.f;
		m_pSkyboxEffect->activeEffect();
		m_pSkyboxEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("view", noTranCamMath.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("model", NPMathHelper::Mat4x4::scaleTransform(1.0f, 1.0f, 1.0f).GetDataColumnMajor());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
		m_pSkyboxEffect->SetInt("envmap", 0);

		glBindVertexArray(m_skybox.GetVAO());
		glDrawElements(GL_TRIANGLES, m_skybox.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		m_pSkyboxEffect->deactiveEffect();
		glCullFace(GL_BACK);
	}

	if (m_bIsWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void ModelViewWindow::RenderMethod_BlinnPhongEnvMap()
{
	if (NPMathHelper::Mat4x4(m_Cam.GetViewMatrix()) != m_matLastCam)
	{
		m_matLastCam = NPMathHelper::Mat4x4(m_Cam.GetViewMatrix());
		m_uiEnvInitSamp = 0;
	}

	NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
		, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
		, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));

	if (modelMat != m_matLastModel)
	{
		m_matLastModel = modelMat;
		m_uiEnvInitSamp = 0;
	}

	if (m_uiEnvInitSamp + ITR_COUNT > m_uiMaxSampling)
		return;

	if (m_uiEnvInitSamp <= 0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		if (m_uiEnvInitSamp > 0)
		{
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pBlinnPhongEnvModelEffect->activeEffect(); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetMatrix("view", m_Cam.GetViewMatrix()); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor()); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetInt("init_samp", m_uiEnvInitSamp); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetInt("max_samp", m_uiMaxSampling); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier); CHECK_GL_ERROR;

		glm::vec3 lightDir;
		lightDir.y = -sin(m_fInYaw);
		lightDir.x = -cos(m_fInYaw) * sin(m_fInPitch);
		lightDir.z = -cos(m_fInYaw) * cos(m_fInPitch);
		m_pBlinnPhongEnvModelEffect->SetVec3("viewPos", m_Cam.GetPos()); CHECK_GL_ERROR;

		m_pBlinnPhongEnvModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular); CHECK_GL_ERROR;
		m_pBlinnPhongEnvModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess); CHECK_GL_ERROR;

		if (m_bIsEnvMapLoaded)
		{
			glActiveTexture(GL_TEXTURE4); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap); CHECK_GL_ERROR;
			m_pBlinnPhongEnvModelEffect->SetInt("envmap", 4); CHECK_GL_ERROR;
		}

		m_pModel->Draw(*m_pBlinnPhongEnvModelEffect); CHECK_GL_ERROR;
		m_uiEnvInitSamp += ITR_COUNT;
		m_fRenderingProgress = (float)m_uiEnvInitSamp / (float)m_uiMaxSampling * 100.f;

		m_pBlinnPhongEnvModelEffect->deactiveEffect();

		glDisable(GL_BLEND);
	}

	if (m_bIsEnvMapLoaded)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		glCullFace(GL_FRONT);
		NPMathHelper::Mat4x4 noTranCamMath = m_Cam.GetViewMatrix();
		noTranCamMath._03 = noTranCamMath._13 = noTranCamMath._23 = 0.f;
		m_pSkyboxEffect->activeEffect();
		m_pSkyboxEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("view", noTranCamMath.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("model", NPMathHelper::Mat4x4::scaleTransform(1.0f, 1.0f, 1.0f).GetDataColumnMajor());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
		m_pSkyboxEffect->SetInt("envmap", 0);

		glBindVertexArray(m_skybox.GetVAO());
		glDrawElements(GL_TRIANGLES, m_skybox.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		m_pSkyboxEffect->deactiveEffect();
		glCullFace(GL_BACK);
	}

	if (m_bIsWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void ModelViewWindow::RenderMethod_BRDFEnvMap()
{
	UpdateBRDFData();

	if (NPMathHelper::Mat4x4(m_Cam.GetViewMatrix()) != m_matLastCam)
	{
		m_matLastCam = NPMathHelper::Mat4x4(m_Cam.GetViewMatrix());
		m_uiEnvInitSamp = 0;
	}

	NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
		, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
		, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));

	if (modelMat != m_matLastModel)
	{
		m_matLastModel = modelMat;
		m_uiEnvInitSamp = 0;
	}

	if (m_uiEnvInitSamp + ITR_COUNT > m_uiNPH * m_uiNTH)
		return;

	if (m_uiEnvInitSamp <= 0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		if (m_uiEnvInitSamp > 0)
		{
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pBRDFEnvModelEffect->activeEffect();
		m_pBRDFEnvModelEffect->SetInt("n_th", m_uiNTH);
		m_pBRDFEnvModelEffect->SetInt("n_ph", m_uiNPH);
		m_pBRDFEnvModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBRDFEnvModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBRDFEnvModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pBRDFEnvModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());
		m_pBRDFEnvModelEffect->SetInt("init_samp", m_uiEnvInitSamp);
		m_pBRDFEnvModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier);

		m_pBRDFEnvModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient); CHECK_GL_ERROR;
		m_pBRDFEnvModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse); CHECK_GL_ERROR;
		m_pBRDFEnvModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular); CHECK_GL_ERROR;
		m_pBRDFEnvModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess); CHECK_GL_ERROR;

		glm::vec3 lightDir;
		lightDir.y = -sin(m_fInYaw);
		lightDir.x = -cos(m_fInYaw) * sin(m_fInPitch);
		lightDir.z = -cos(m_fInYaw) * cos(m_fInPitch);
		m_pBRDFEnvModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		if (m_bIsLoadTexture)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m_iBRDFEstTex);
			m_pBRDFEnvModelEffect->SetInt("texture_brdf", 0);
		}

		if (m_bIsEnvMapLoaded)
		{
			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
			m_pBRDFEnvModelEffect->SetInt("envmap", 4);
		}

		m_pModel->Draw(*m_pBRDFEnvModelEffect);
		m_uiEnvInitSamp += ITR_COUNT;
		m_fRenderingProgress = (float)m_uiEnvInitSamp / (float)(m_uiNPH * m_uiNTH) * 100.f;

		m_pBRDFEnvModelEffect->deactiveEffect();

		glDisable(GL_BLEND);
	}

	if (m_bIsEnvMapLoaded)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		glCullFace(GL_FRONT);
		NPMathHelper::Mat4x4 noTranCamMath = m_Cam.GetViewMatrix();
		noTranCamMath._03 = noTranCamMath._13 = noTranCamMath._23 = 0.f;
		m_pSkyboxEffect->activeEffect();
		m_pSkyboxEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("view", noTranCamMath.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("model", NPMathHelper::Mat4x4::scaleTransform(1.0f, 1.0f, 1.0f).GetDataColumnMajor());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
		m_pSkyboxEffect->SetInt("envmap", 0);

		glBindVertexArray(m_skybox.GetVAO());
		glDrawElements(GL_TRIANGLES, m_skybox.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		m_pSkyboxEffect->deactiveEffect();
		glCullFace(GL_BACK);
	}

	if (m_bIsWireFrame)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void ModelViewWindow::RenderMethod_DiffuseEnvMapS()
{
	UpdateBRDFData();

	if (NPMathHelper::Mat4x4(m_Cam.GetViewMatrix()) != m_matLastCam)
	{
		m_matLastCam = NPMathHelper::Mat4x4(m_Cam.GetViewMatrix());
		m_uiEnvInitSamp = 0;
	}

	NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
		, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
		, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));

	if (modelMat != m_matLastModel)
	{
		m_matLastModel = modelMat;
		m_uiEnvInitSamp = 0;
	}

	if (m_uiEnvInitSamp + ITR_COUNT > m_uiEnvShadowMaxSamp)
		return;

	if (m_uiEnvInitSamp <= 0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		// Samp Dir
		NPMathHelper::Vec2 hemiSpace = NPSamplingHelper::hammersley2d(m_uiEnvInitSamp / 2, m_uiMaxSampling / 2);
		NPMathHelper::Vec3 sampDir = NPSamplingHelper::hemisphereSample_uniform(hemiSpace._x, hemiSpace._y);
		if (m_uiEnvInitSamp % 2 == 1)
			sampDir._y *= -1;

		// Depth rendering
		if (m_pModel)
		{
			Render_ShadowMap(sampDir*-1.f);
		}

		if (m_bIsWireFrame)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		if (m_uiEnvInitSamp > 0)
		{
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		if (m_bIsShowFloor)
		{
			NPMathHelper::Mat4x4 floorModelMat = NPMathHelper::Mat4x4::Identity();
			NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(floorModelMat));
			m_pDiffuseNormalEnvSModelEffect->activeEffect();
			m_pDiffuseNormalEnvSModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
			m_pDiffuseNormalEnvSModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
			m_pDiffuseNormalEnvSModelEffect->SetMatrix("model", floorModelMat.GetDataColumnMajor());
			m_pDiffuseNormalEnvSModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());
			m_pDiffuseNormalEnvSModelEffect->SetInt("init_samp", m_uiEnvInitSamp);
			m_pDiffuseNormalEnvSModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier);
			m_pDiffuseNormalEnvSModelEffect->SetInt("max_samp", m_uiEnvShadowMaxSamp);

			m_pDiffuseNormalEnvSModelEffect->SetVec3("viewPos", m_Cam.GetPos());

			if (m_bIsEnvMapLoaded)
			{
				glActiveTexture(GL_TEXTURE4);
				glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
				m_pDiffuseNormalEnvSModelEffect->SetInt("envmap", 4);
			}

			m_pDiffuseNormalEnvSModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
			m_pDiffuseNormalEnvSModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
			m_pDiffuseNormalEnvSModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
			glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
			m_pDiffuseNormalEnvSModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

			glActiveTexture(GL_TEXTURE0); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_iFloorTex); CHECK_GL_ERROR;
			m_pDiffuseNormalEnvSModelEffect->SetInt("texture_diffuse1", 0); CHECK_GL_ERROR;

			glActiveTexture(GL_TEXTURE1); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_iFloorNormalTex); CHECK_GL_ERROR;
			m_pDiffuseNormalEnvSModelEffect->SetInt("texture_normal1", 1); CHECK_GL_ERROR;

			m_pDiffuseNormalEnvSModelEffect->SetVec3("material.ambient", m_floorMaterial.ambient);
			m_pDiffuseNormalEnvSModelEffect->SetVec3("material.diffuse", m_floorMaterial.diffuse);
			m_pDiffuseNormalEnvSModelEffect->SetVec3("material.specular", m_floorMaterial.specular);
			m_pDiffuseNormalEnvSModelEffect->SetFloat("material.shininess", m_floorMaterial.shininess);

			m_pDiffuseNormalEnvSModelEffect->SetVec3("samp_dir_w", sampDir);
			glBindVertexArray(m_floor.GetVAO()); CHECK_GL_ERROR;
			glDrawElements(GL_TRIANGLES, m_floor.GetIndicesSize(), GL_UNSIGNED_INT, 0); CHECK_GL_ERROR;
			glBindVertexArray(0);
			m_pDiffuseNormalEnvSModelEffect->deactiveEffect();
		}

		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pDiffuseEnvSModelEffect->activeEffect();
		m_pDiffuseEnvSModelEffect->SetInt("n_th", m_uiNTH);
		m_pDiffuseEnvSModelEffect->SetInt("n_ph", m_uiNPH);
		m_pDiffuseEnvSModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pDiffuseEnvSModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pDiffuseEnvSModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pDiffuseEnvSModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());
		m_pDiffuseEnvSModelEffect->SetInt("init_samp", m_uiEnvInitSamp);
		m_pDiffuseEnvSModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier);
		m_pDiffuseEnvSModelEffect->SetInt("max_samp", m_uiEnvShadowMaxSamp);

		m_pDiffuseEnvSModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		if (m_bIsEnvMapLoaded)
		{
			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
			m_pDiffuseEnvSModelEffect->SetInt("envmap", 4);
		}

		m_pDiffuseEnvSModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pDiffuseEnvSModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pDiffuseEnvSModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pDiffuseEnvSModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		m_pDiffuseEnvSModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient);
		m_pDiffuseEnvSModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse);
		m_pDiffuseEnvSModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular);
		m_pDiffuseEnvSModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess);
		m_pDiffuseEnvSModelEffect->SetVec3("samp_dir_w", sampDir);
		m_pModel->Draw(*m_pDiffuseEnvSModelEffect);
		m_uiEnvInitSamp += ITR_COUNT;
		m_fRenderingProgress = (float)m_uiEnvInitSamp / (float)(m_uiEnvShadowMaxSamp)* 100.f;

		m_pDiffuseEnvSModelEffect->deactiveEffect();

		glDisable(GL_BLEND);

		if (m_bIsWireFrame)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
	}

	if (m_bIsEnvMapLoaded)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		glCullFace(GL_FRONT);
		NPMathHelper::Mat4x4 noTranCamMath = m_Cam.GetViewMatrix();
		noTranCamMath._03 = noTranCamMath._13 = noTranCamMath._23 = 0.f;
		m_pSkyboxEffect->activeEffect();
		m_pSkyboxEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("view", noTranCamMath.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("model", NPMathHelper::Mat4x4::scaleTransform(1.0f, 1.0f, 1.0f).GetDataColumnMajor());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
		m_pSkyboxEffect->SetInt("envmap", 0);

		glBindVertexArray(m_skybox.GetVAO());
		glDrawElements(GL_TRIANGLES, m_skybox.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		m_pSkyboxEffect->deactiveEffect();
		glCullFace(GL_BACK);
	}
}

void ModelViewWindow::RenderMethod_BlinnPhongEnvMapS()
{
	UpdateBRDFData();

	if (NPMathHelper::Mat4x4(m_Cam.GetViewMatrix()) != m_matLastCam)
	{
		m_matLastCam = NPMathHelper::Mat4x4(m_Cam.GetViewMatrix());
		m_uiEnvInitSamp = 0;
	}

	NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
		, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
		, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));

	if (modelMat != m_matLastModel)
	{
		m_matLastModel = modelMat;
		m_uiEnvInitSamp = 0;
	}

	if (m_uiEnvInitSamp + ITR_COUNT > m_uiEnvShadowMaxSamp)
		return;

	if (m_uiEnvInitSamp <= 0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		// Samp Dir
		NPMathHelper::Vec2 hemiSpace = NPSamplingHelper::hammersley2d(m_uiEnvInitSamp / 2, m_uiMaxSampling / 2);
		NPMathHelper::Vec3 sampDir = NPSamplingHelper::hemisphereSample_uniform(hemiSpace._x, hemiSpace._y);
		if (m_uiEnvInitSamp % 2 == 1)
			sampDir._y *= -1;

		// Depth rendering
		if (m_pModel)
		{
			Render_ShadowMap(sampDir*-1.f);
		}

		if (m_bIsWireFrame)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		if (m_uiEnvInitSamp > 0)
		{
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		if (m_bIsShowFloor)
		{
			NPMathHelper::Mat4x4 floorModelMat = NPMathHelper::Mat4x4::Identity();
			NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(floorModelMat));
			m_pBlinnPhongNormalEnvSModelEffect->activeEffect();
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("model", floorModelMat.GetDataColumnMajor());
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("init_samp", m_uiEnvInitSamp);
			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier);
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("max_samp", m_uiEnvShadowMaxSamp);

			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("viewPos", m_Cam.GetPos());

			if (m_bIsEnvMapLoaded)
			{
				glActiveTexture(GL_TEXTURE4);
				glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
				m_pBlinnPhongNormalEnvSModelEffect->SetInt("envmap", 4);
			}

			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
			glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

			glActiveTexture(GL_TEXTURE0); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_iFloorTex); CHECK_GL_ERROR;
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("texture_diffuse1", 0); CHECK_GL_ERROR;

			glActiveTexture(GL_TEXTURE1); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_iFloorNormalTex); CHECK_GL_ERROR;
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("texture_normal1", 1); CHECK_GL_ERROR;

			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("material.ambient", m_floorMaterial.ambient);
			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("material.diffuse", m_floorMaterial.diffuse);
			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("material.specular", m_floorMaterial.specular);
			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("material.shininess", m_floorMaterial.shininess);

			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("samp_dir_w", sampDir);
			glBindVertexArray(m_floor.GetVAO()); CHECK_GL_ERROR;
			glDrawElements(GL_TRIANGLES, m_floor.GetIndicesSize(), GL_UNSIGNED_INT, 0); CHECK_GL_ERROR;
			glBindVertexArray(0);
			m_pBlinnPhongNormalEnvSModelEffect->deactiveEffect();
		}

		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pBlinnPhongEnvSModelEffect->activeEffect();
		m_pBlinnPhongEnvSModelEffect->SetInt("n_th", m_uiNTH);
		m_pBlinnPhongEnvSModelEffect->SetInt("n_ph", m_uiNPH);
		m_pBlinnPhongEnvSModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBlinnPhongEnvSModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBlinnPhongEnvSModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pBlinnPhongEnvSModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());
		m_pBlinnPhongEnvSModelEffect->SetInt("init_samp", m_uiEnvInitSamp);
		m_pBlinnPhongEnvSModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier);
		m_pBlinnPhongEnvSModelEffect->SetInt("max_samp", m_uiEnvShadowMaxSamp);

		m_pBlinnPhongEnvSModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		if (m_bIsEnvMapLoaded)
		{
			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
			m_pBlinnPhongEnvSModelEffect->SetInt("envmap", 4);
		}

		m_pBlinnPhongEnvSModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pBlinnPhongEnvSModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pBlinnPhongEnvSModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pBlinnPhongEnvSModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		m_pBlinnPhongEnvSModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient);
		m_pBlinnPhongEnvSModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse);
		m_pBlinnPhongEnvSModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular);
		m_pBlinnPhongEnvSModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess);
		m_pBlinnPhongEnvSModelEffect->SetVec3("samp_dir_w", sampDir);
		m_pModel->Draw(*m_pBlinnPhongEnvSModelEffect);
		m_uiEnvInitSamp += ITR_COUNT;
		m_fRenderingProgress = (float)m_uiEnvInitSamp / (float)(m_uiEnvShadowMaxSamp)* 100.f;

		m_pBlinnPhongEnvSModelEffect->deactiveEffect();

		glDisable(GL_BLEND);

		if (m_bIsWireFrame)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
	}

	if (m_bIsEnvMapLoaded)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		glCullFace(GL_FRONT);
		NPMathHelper::Mat4x4 noTranCamMath = m_Cam.GetViewMatrix();
		noTranCamMath._03 = noTranCamMath._13 = noTranCamMath._23 = 0.f;
		m_pSkyboxEffect->activeEffect();
		m_pSkyboxEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("view", noTranCamMath.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("model", NPMathHelper::Mat4x4::scaleTransform(1.0f, 1.0f, 1.0f).GetDataColumnMajor());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
		m_pSkyboxEffect->SetInt("envmap", 0);

		glBindVertexArray(m_skybox.GetVAO());
		glDrawElements(GL_TRIANGLES, m_skybox.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		m_pSkyboxEffect->deactiveEffect();
		glCullFace(GL_BACK);
	}
}

void ModelViewWindow::RenderMethod_BRDFEnvMapS()
{
	UpdateBRDFData();

	if (NPMathHelper::Mat4x4(m_Cam.GetViewMatrix()) != m_matLastCam)
	{
		m_matLastCam = NPMathHelper::Mat4x4(m_Cam.GetViewMatrix());
		m_uiEnvInitSamp = 0;
	}

	NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
		, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
		, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));

	if (modelMat != m_matLastModel)
	{
		m_matLastModel = modelMat;
		m_uiEnvInitSamp = 0;
	}

	if (m_uiEnvInitSamp + ITR_COUNT > m_uiEnvShadowMaxSamp)
		return;

	if (m_uiEnvInitSamp <= 0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (/*m_bIsLoadTexture &&*/ m_pModel)
	{
		// Samp Dir
		NPMathHelper::Vec2 hemiSpace = NPSamplingHelper::hammersley2d(m_uiEnvInitSamp / 2, m_uiMaxSampling / 2);
		NPMathHelper::Vec3 sampDir = NPSamplingHelper::hemisphereSample_uniform(hemiSpace._x, hemiSpace._y);
		if (m_uiEnvInitSamp % 2 == 1)
			sampDir._y *= -1;

		// Depth rendering
		if (m_pModel)
		{
			Render_ShadowMap(sampDir*-1.f);
		}

		if (m_bIsWireFrame)
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		if (m_uiEnvInitSamp > 0)
		{
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		if (m_bIsShowFloor)
		{
			NPMathHelper::Mat4x4 floorModelMat = NPMathHelper::Mat4x4::Identity();
			NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(floorModelMat));
			m_pBlinnPhongNormalEnvSModelEffect->activeEffect();
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("model", floorModelMat.GetDataColumnMajor());
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("init_samp", m_uiEnvInitSamp);
			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier);
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("max_samp", m_uiEnvShadowMaxSamp);

			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("viewPos", m_Cam.GetPos());

			if (m_bIsEnvMapLoaded)
			{
				glActiveTexture(GL_TEXTURE4);
				glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
				m_pBlinnPhongNormalEnvSModelEffect->SetInt("envmap", 4);
			}

			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
			m_pBlinnPhongNormalEnvSModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
			glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

			glActiveTexture(GL_TEXTURE0); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_iFloorTex); CHECK_GL_ERROR;
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("texture_diffuse1", 0); CHECK_GL_ERROR;

			glActiveTexture(GL_TEXTURE1); CHECK_GL_ERROR;
			glBindTexture(GL_TEXTURE_2D, m_iFloorNormalTex); CHECK_GL_ERROR;
			m_pBlinnPhongNormalEnvSModelEffect->SetInt("texture_normal1", 1); CHECK_GL_ERROR;

			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("material.ambient", m_floorMaterial.ambient);
			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("material.diffuse", m_floorMaterial.diffuse);
			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("material.specular", m_floorMaterial.specular);
			m_pBlinnPhongNormalEnvSModelEffect->SetFloat("material.shininess", m_floorMaterial.shininess);

			m_pBlinnPhongNormalEnvSModelEffect->SetVec3("samp_dir_w", sampDir);
			glBindVertexArray(m_floor.GetVAO()); CHECK_GL_ERROR;
			glDrawElements(GL_TRIANGLES, m_floor.GetIndicesSize(), GL_UNSIGNED_INT, 0); CHECK_GL_ERROR;
			glBindVertexArray(0);
			m_pBlinnPhongNormalEnvSModelEffect->deactiveEffect();
		}

		NPMathHelper::Mat4x4 tranInvModelMat = NPMathHelper::Mat4x4::transpose(NPMathHelper::Mat4x4::inverse(modelMat));
		m_pBRDFEnvSModelEffect->activeEffect();
		m_pBRDFEnvSModelEffect->SetInt("n_th", m_uiNTH);
		m_pBRDFEnvSModelEffect->SetInt("n_ph", m_uiNPH);
		m_pBRDFEnvSModelEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBRDFEnvSModelEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBRDFEnvSModelEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
		m_pBRDFEnvSModelEffect->SetMatrix("tranInvModel", tranInvModelMat.GetDataColumnMajor());
		m_pBRDFEnvSModelEffect->SetInt("init_samp", m_uiEnvInitSamp);
		m_pBRDFEnvSModelEffect->SetFloat("env_multiplier", m_fEnvMapMultiplier);
		m_pBRDFEnvSModelEffect->SetInt("max_samp", m_uiEnvShadowMaxSamp);

		m_pBRDFEnvSModelEffect->SetVec3("material.ambient", m_modelBlinnPhongMaterial.ambient); CHECK_GL_ERROR;
		m_pBRDFEnvSModelEffect->SetVec3("material.diffuse", m_modelBlinnPhongMaterial.diffuse); CHECK_GL_ERROR;
		m_pBRDFEnvSModelEffect->SetVec3("material.specular", m_modelBlinnPhongMaterial.specular); CHECK_GL_ERROR;
		m_pBRDFEnvSModelEffect->SetFloat("material.shininess", m_modelBlinnPhongMaterial.shininess); CHECK_GL_ERROR;

		m_pBRDFEnvSModelEffect->SetVec3("viewPos", m_Cam.GetPos());

		if (m_bIsForceTangent)
		{
			m_pBRDFEnvSModelEffect->SetVec3("forced_tangent_w", m_v3ForcedTangent);
		}

		if (m_bIsLoadTexture)
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, m_iBRDFEstTex);
			m_pBRDFEnvSModelEffect->SetInt("texture_brdf", 0);
		}

		if (m_bIsEnvMapLoaded)
		{
			glActiveTexture(GL_TEXTURE4);
			glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
			m_pBRDFEnvSModelEffect->SetInt("envmap", 4);
		}

		m_pBRDFEnvSModelEffect->SetFloat("biasMin", m_fShadowBiasMin);
		m_pBRDFEnvSModelEffect->SetFloat("biasMax", m_fShadowBiasMax);
		m_pBRDFEnvSModelEffect->SetMatrix("shadowMap", m_matShadowMapMat.GetDataColumnMajor());
		glActiveTexture(GL_TEXTURE5); CHECK_GL_ERROR;
		glBindTexture(GL_TEXTURE_2D, m_uiDepthMapTex); CHECK_GL_ERROR;
		m_pBRDFEnvSModelEffect->SetInt("texture_shadow", 5); CHECK_GL_ERROR;

		m_pBRDFEnvSModelEffect->SetVec3("samp_dir_w", sampDir);
		m_pModel->Draw(*m_pBRDFEnvSModelEffect);
		m_uiEnvInitSamp += ITR_COUNT;
		m_fRenderingProgress = (float)m_uiEnvInitSamp / (float)(m_uiEnvShadowMaxSamp)* 100.f;

		m_pBRDFEnvSModelEffect->deactiveEffect();

		glDisable(GL_BLEND);

		if (m_bIsWireFrame)
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
	}

	if (m_bIsEnvMapLoaded)
	{
		NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
		glCullFace(GL_FRONT);
		NPMathHelper::Mat4x4 noTranCamMath = m_Cam.GetViewMatrix();
		noTranCamMath._03 = noTranCamMath._13 = noTranCamMath._23 = 0.f;
		m_pSkyboxEffect->activeEffect();
		m_pSkyboxEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("view", noTranCamMath.GetDataColumnMajor());
		m_pSkyboxEffect->SetMatrix("model", NPMathHelper::Mat4x4::scaleTransform(1.0f, 1.0f, 1.0f).GetDataColumnMajor());

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_CUBE_MAP, m_uiEnvMap);
		m_pSkyboxEffect->SetInt("envmap", 0);

		glBindVertexArray(m_skybox.GetVAO());
		glDrawElements(GL_TRIANGLES, m_skybox.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		m_pSkyboxEffect->deactiveEffect();
		glCullFace(GL_BACK);
	}
}


void ModelViewWindow::RenderMethod_DiffuseDirLightInit()
{

}

void ModelViewWindow::RenderMethod_BlinnPhongDirLightInit()
{

}

void ModelViewWindow::RenderMethod_BRDFDirLightInit()
{

}

void ModelViewWindow::RenderMethod_DiffuseEnvMapInit()
{
	m_uiEnvInitSamp = 0;
	m_matLastCam = NPMathHelper::Mat4x4::Identity();
	m_matLastModel = NPMathHelper::Mat4x4::Identity();
}

void ModelViewWindow::RenderMethod_BlinnPhongEnvMapInit()
{
	m_uiEnvInitSamp = 0;
	m_matLastCam = NPMathHelper::Mat4x4::Identity();
	m_matLastModel = NPMathHelper::Mat4x4::Identity();
}

void ModelViewWindow::RenderMethod_BRDFEnvMapInit()
{
	m_uiEnvInitSamp = 0;
	m_matLastCam = NPMathHelper::Mat4x4::Identity();
	m_matLastModel = NPMathHelper::Mat4x4::Identity();
}

void ModelViewWindow::RenderMethod_DiffuseEnvMapSInit()
{
	m_uiEnvInitSamp = 0;
	m_matLastCam = NPMathHelper::Mat4x4::Identity();
	m_matLastModel = NPMathHelper::Mat4x4::Identity();
}

void ModelViewWindow::RenderMethod_BlinnPhongEnvMapSInit()
{
	m_uiEnvInitSamp = 0;
	m_matLastCam = NPMathHelper::Mat4x4::Identity();
	m_matLastModel = NPMathHelper::Mat4x4::Identity();
}

void ModelViewWindow::RenderMethod_BRDFEnvMapSInit()
{
	m_uiEnvInitSamp = 0;
	m_matLastCam = NPMathHelper::Mat4x4::Identity();
	m_matLastModel = NPMathHelper::Mat4x4::Identity();
}


void ModelViewWindow::RenderMethod_DiffuseDirLightQuit()
{

}

void ModelViewWindow::RenderMethod_BlinnPhongDirLightQuit()
{

}

void ModelViewWindow::RenderMethod_BRDFDirLightQuit()
{

}

void ModelViewWindow::RenderMethod_DiffuseEnvMapQuit()
{

}

void ModelViewWindow::RenderMethod_BlinnPhongEnvMapQuit()
{

}

void ModelViewWindow::RenderMethod_BRDFEnvMapQuit()
{

}

void ModelViewWindow::RenderMethod_DiffuseEnvMapSQuit()
{

}

void ModelViewWindow::RenderMethod_BlinnPhongEnvMapSQuit()
{

}

void ModelViewWindow::RenderMethod_BRDFEnvMapSQuit()
{

}


void ModelViewWindow::Render_ShadowMap(const NPMathHelper::Vec3 lightDir)
{
	NPMathHelper::Mat4x4 modelMat = NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::translation(m_v3ModelPos)
		, NPMathHelper::Mat4x4::mul(NPMathHelper::Mat4x4::rotationTransform(m_v3ModelRot)
		, NPMathHelper::Mat4x4::scaleTransform(m_fModelScale, m_fModelScale, m_fModelScale)));

	BRDFModel::SphericalSpace space = m_pModel->GetSphericalSpace();
	space.m_v3Center = NPMathHelper::Vec3::transform(modelMat, space.m_v3Center, true);
	space.m_fRadius = m_fModelScale * space.m_fRadius;
	space = space.Merge(m_floorSpace);

	NPMathHelper::Mat4x4 lightProj = NPMathHelper::Mat4x4::orthogonalProjection(2.0f*space.m_fRadius, 2.0f*space.m_fRadius, 1.f, 2.f*space.m_fRadius);
	NPMathHelper::Mat4x4 lightView = NPMathHelper::Mat4x4::lookAt(space.m_v3Center - lightDir * (space.m_fRadius + 1.5f)
		, space.m_v3Center, NPMathHelper::Vec3(0.f, 1.f, 0.f));
	m_matShadowMapMat = NPMathHelper::Mat4x4::mul(lightProj, lightView);

	glDisable(GL_CULL_FACE);
	glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
	glBindFramebuffer(GL_FRAMEBUFFER, m_uiDepthMapFBO);

	glClear(GL_DEPTH_BUFFER_BIT);
	m_pDepthEffect->activeEffect();
	m_pDepthEffect->SetMatrix("lightMat", m_matShadowMapMat.GetDataColumnMajor());
	m_pDepthEffect->SetMatrix("model", modelMat.GetDataColumnMajor());
	m_pModel->Draw(*m_pDepthEffect);
	if (m_bIsShowFloor)
	{
		m_pDepthEffect->SetMatrix("model", NPMathHelper::Mat4x4::Identity().GetDataColumnMajor());
		glBindVertexArray(m_floor.GetVAO());
		glDrawElements(GL_TRIANGLES, m_floor.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}
	m_pDepthEffect->deactiveEffect();
	glBindFramebuffer(GL_FRAMEBUFFER, m_uiHDRFBO);
	glViewport(0, 0, m_iSizeW, m_iSizeH);
	glEnable(GL_CULL_FACE);
}


void ModelViewWindow::RenderScreenQuad()
{
	if (m_uiVAOQuad == 0)
	{
		GLfloat quadVertices[] = {
			-1.f, 1.f, 0.f, 0.f, 1.f,
			-1.f, -1.f, 0.f, 0.f, 0.f,
			1.f, 1.f, 0.f, 1.f, 1.f,
			1.f, -1.f, 0.f, 1.f, 0.f
		};
		glGenVertexArrays(1, &m_uiVAOQuad);
		glGenBuffers(1, &m_uiVBOQuad);
		glBindVertexArray(m_uiVAOQuad);
		glBindBuffer(GL_ARRAY_BUFFER, m_uiVBOQuad);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)( 3 * sizeof(GLfloat)));
	}
	glBindVertexArray(m_uiVAOQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}



