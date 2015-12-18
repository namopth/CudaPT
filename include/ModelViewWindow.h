#ifndef MODELVIEWWINDOW_H
#define MODELVIEWWINDOW_H

#include "glhelper.h"
#include "camhelper.h"
#include "mathhelper.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace BRDFModel
{
	struct SphericalSpace {
		NPMathHelper::Vec3 m_v3Center;
		float m_fRadius;

		SphericalSpace() : m_v3Center(0.f, 0.f, 0.f), m_fRadius(0.f)
		{ }

		void Expand(NPMathHelper::Vec3 &point)
		{
			NPMathHelper::Vec3 cenToExpand = point - m_v3Center;
			float cenToExpandLength = NPMathHelper::Vec3::length(cenToExpand);
			if (cenToExpandLength > m_fRadius)
			{
				NPMathHelper::Vec3 otherSideEdge = m_v3Center - NPMathHelper::Vec3::normalize(cenToExpand) * m_fRadius;
				m_v3Center = (otherSideEdge + point) * 0.5f;
				m_fRadius = NPMathHelper::Vec3::length(m_v3Center - otherSideEdge);
			}
		}

		SphericalSpace Merge(const SphericalSpace &other)
		{
			SphericalSpace result;
			NPMathHelper::Vec3 toOther = other.m_v3Center - m_v3Center;
			float lenToOther = NPMathHelper::Vec3::length(toOther);
			float maxForward = glm::max(m_fRadius, other.m_fRadius + lenToOther);
			float maxBackward = glm::max(m_fRadius, other.m_fRadius - lenToOther);
			NPMathHelper::Vec3 toOtherDir = NPMathHelper::Vec3::normalize(toOther);
			NPMathHelper::Vec3 forwPoint = m_v3Center + toOtherDir * maxForward;
			NPMathHelper::Vec3 backPoint = m_v3Center - toOtherDir * maxBackward;
			result.m_fRadius = (maxForward + maxBackward) * 0.5f;
			result.m_v3Center = (forwPoint + backPoint) * 0.5f;
			return result;
		}

		static SphericalSpace CalcSpaceFromPoints(std::vector<NPMathHelper::Vec3> &points)
		{
			SphericalSpace result;
			if (points.size() <= 0)
				return result;
			result.m_v3Center = points[0];
			for (auto it = points.begin() + 1; it != points.end(); it++)
			{
				result.Expand(*it);
			}

			return result;
		}
	};
	struct Vertex {
		NPMathHelper::Vec3 position;
		NPMathHelper::Vec3 normal;
		NPMathHelper::Vec3 tangent;
		NPMathHelper::Vec2 texCoords;
	};

	struct Texture {
		GLuint id;
		std::string name;
		unsigned int type;
	};

	class Mesh {
	public:
		std::vector<Vertex> m_vertices;
		std::vector<GLuint> m_indices;
		std::vector<Texture> m_textures;

		SphericalSpace m_space;

		Mesh(std::vector<Vertex> vertices, std::vector<GLuint> indices, std::vector<Texture> textures)
			: m_vertices(vertices)
			, m_indices(indices)
			, m_textures(textures)
		{
			SetupMesh();
		}

		~Mesh()
		{
			for (auto &texture : m_textures)
			{
				glDeleteTextures(1, &texture.id);
			}
			glDeleteVertexArrays(1,&m_iVAO);
			glDeleteBuffers(1,&m_iVBO);
			glDeleteBuffers(1,&m_iEBO);
		}

		void Draw(NPGLHelper::Effect &effect);

	protected:
		GLuint m_iVAO;
		GLuint m_iVBO;
		GLuint m_iEBO;

		void SetupMesh();
	};

	class Model {
	public:
		Model();
		~Model() 
		{
			for (auto &mesh : m_meshes)
			{
				if (mesh)
					delete mesh;
			}
			m_meshes.clear(); 
		}

		void Draw(NPGLHelper::Effect &effect);
		bool LoadModel(const char* path);
		inline SphericalSpace GetSphericalSpace() { return m_space; }

	protected:
		void ProcessNode(aiNode* node, const aiScene* scene);
		Mesh* ProcessMesh(aiMesh* mesh, const aiScene* scene);
		std::vector<Texture> LoadMaterialTextures(aiMaterial* mat, aiTextureType type, const char* name, const unsigned int typeId);

		std::vector<Mesh*> m_meshes;
		std::string m_sDirectory;

		SphericalSpace m_space;
	};
}

class ModelViewWindow : public NPGLHelper::Window
{
public:
	struct CUBEMAPLOADCMD{
		ModelViewWindow* win;
		unsigned int side;
		CUBEMAPLOADCMD() {}
		CUBEMAPLOADCMD(ModelViewWindow* w, unsigned int s) : win(w), side(s) {}
	};

	ModelViewWindow(const char* name, const int sizeW = 800, const int sizeH = 600);
	virtual ~ModelViewWindow();

	virtual int OnInit();
	virtual int OnTick(const float deltaTime);
	virtual void OnTerminate();
	virtual void OnHandleInputMSG(const INPUTMSG &msg);

	void OpenModelData();
	void SetBRDFData(const char* path, unsigned int n_th, unsigned int n_ph);

	void SetCubemap(unsigned int side);
	void LoadCubemap();

	enum RENDERINGMETHODS
	{
		RENDERINGMETHOD_DIFFUSEDIRLIGHT,
		RENDERINGMETHOD_BLINNPHONGDIRLIGHT,
		RENDERINGMETHOD_BRDFDIRLIGHT,
		RENDERINGMETHOD_DIFFUSEENVMAP,
		RENDERINGMETHOD_BLINNPHONGENVMAP,
		RENDERINGMETHOD_BRDFENVMAP,
		RENDERINGMETHOD_DIFFUSEENVMAPS,
		RENDERINGMETHOD_BLINNPHONGENVMAPS,
		RENDERINGMETHOD_BRDFENVMAPS,
		RENDERINGMETHOD_N,
		RENDERINGMETHOD_NONE,
	};
	RENDERINGMETHODS GetRenderingMethod() { return m_eRenderingMethod; }
	void SetRenderingMethod(RENDERINGMETHODS method);


	enum RECORD_STATUS
	{
		REC_NONE,
		REC_PREVIEW,
		REC_RECORDING
	};
	void SetRecording(const RECORD_STATUS status);

protected:
	void UpdateBRDFData();

	// Rendering methods
	RENDERINGMETHODS m_eRenderingMethod;

	void RenderMethod_DiffuseDirLight();
	void RenderMethod_BlinnPhongDirLight();
	void RenderMethod_BRDFDirLight();
	void RenderMethod_DiffuseEnvMap();
	void RenderMethod_BlinnPhongEnvMap();
	void RenderMethod_BRDFEnvMap();
	void RenderMethod_DiffuseEnvMapS();
	void RenderMethod_BlinnPhongEnvMapS();
	void RenderMethod_BRDFEnvMapS();

	void RenderMethod_DiffuseDirLightInit();
	void RenderMethod_BlinnPhongDirLightInit();
	void RenderMethod_BRDFDirLightInit();
	void RenderMethod_DiffuseEnvMapInit();
	void RenderMethod_BlinnPhongEnvMapInit();
	void RenderMethod_BRDFEnvMapInit();
	void RenderMethod_DiffuseEnvMapSInit();
	void RenderMethod_BlinnPhongEnvMapSInit();
	void RenderMethod_BRDFEnvMapSInit();

	void RenderMethod_DiffuseDirLightQuit();
	void RenderMethod_BlinnPhongDirLightQuit();
	void RenderMethod_BRDFDirLightQuit();
	void RenderMethod_DiffuseEnvMapQuit();
	void RenderMethod_BlinnPhongEnvMapQuit();
	void RenderMethod_BRDFEnvMapQuit();
	void RenderMethod_BRDFEnvMapSQuit();
	void RenderMethod_DiffuseEnvMapSQuit();
	void RenderMethod_BlinnPhongEnvMapSQuit();

	void Render_ShadowMap(const NPMathHelper::Vec3 lightDir);

	// Render Quad
	void RenderScreenQuad();
	GLuint m_uiVBOQuad;
	GLuint m_uiVAOQuad;

	// Main Render Target (HDR)
	GLuint m_uiHDRFBO;
	GLuint m_uiHDRCB;
	GLuint m_uiHDRDB;
	NPGLHelper::Effect* m_pFinalComposeEffect;
	float m_fExposure;
	NPCamHelper::RotateCamera m_Cam;

	NPGLHelper::Effect* m_pDiffuseModelEffect;
	NPGLHelper::Effect* m_pBlinnPhongModelEffect;
	NPGLHelper::Effect* m_pBRDFModelEffect;
	NPGLHelper::Effect* m_pDiffuseEnvModelEffect;
	NPGLHelper::Effect* m_pBlinnPhongEnvModelEffect;
	NPGLHelper::Effect* m_pBRDFEnvModelEffect;
	NPGLHelper::Effect* m_pDiffuseEnvSModelEffect;
	NPGLHelper::Effect* m_pBlinnPhongEnvSModelEffect;
	NPGLHelper::Effect* m_pBRDFEnvSModelEffect;
	unsigned int m_uiEnvInitSamp;
	unsigned int m_uiEnvShadowMaxSamp;
	NPMathHelper::Mat4x4 m_matLastCam;
	NPMathHelper::Mat4x4 m_matLastModel;
	float m_fRenderingProgress;

	// BRDF
	bool m_bIsBRDFUpdated;
	std::string m_sNewBRDFPath;
	unsigned int m_uiNewTH;
	unsigned int m_uiNewPH;
	unsigned int m_uiNTH, m_uiNPH;
	bool m_bIsLoadTexture;
	GLuint m_iBRDFEstTex;
	std::string m_sBRDFTextureName;
	NPMathHelper::Vec3 m_v3ForcedTangent;
	bool m_bIsForceTangent;

	// Blinn-phong
	unsigned int m_uiMaxSampling;

	struct Material {
		NPMathHelper::Vec3 ambient;
		NPMathHelper::Vec3 diffuse;
		NPMathHelper::Vec3 specular;
		float shininess;
		Material() : shininess(0.5f), ambient(0.1, 0.1f, 0.1f)
			, diffuse(0.3f, 0.3f, 0.3f), specular(0.3f, 0.3f, 0.3) {}
	};

	struct DirLight {
		NPMathHelper::Vec3 dir;
		NPMathHelper::Vec3 ambient;
		NPMathHelper::Vec3 diffuse;
		NPMathHelper::Vec3 specular;
		DirLight() : dir(0.f, 1.f, 0.f), ambient(0.1, 0.1f, 0.1f)
			, diffuse(0.3f, 0.3f, 0.3f), specular(0.3f, 0.3f, 0.3) {}

	};

	// Model
	bool m_bIsLoadModel;
	BRDFModel::Model* m_pModel;
	std::string m_sModelName;
	Material m_modelBlinnPhongMaterial;

	// Display Options
	bool m_bIsWireFrame;
	bool m_bIsSceneGUI;
	NPGLHelper::DebugLine m_InLine;
	NPGLHelper::DebugLine m_AxisLine[3];

	// Light
	DirLight m_dirLight;
	glm::vec3 m_v3LightColor;
	float m_fLightIntMultiplier;
	float m_fInSenX, m_fInSenY;
	float m_fInPitch, m_fInYaw;

	// Shadow Map
	float m_fShadowBiasMin;
	float m_fShadowBiasMax;
	NPGLHelper::Effect* m_pDepthEffect;
	GLuint m_uiDepthMapFBO;
	GLuint m_uiDepthMapTex;
	NPMathHelper::Mat4x4 m_matShadowMapMat;
	static const unsigned int SHADOW_WIDTH;
	static const unsigned int SHADOW_HEIGHT;

	// Env Map
	float m_fEnvMapMultiplier;
	bool m_bIsEnvMapLoaded;
	bool m_bIsEnvMapDirty;
	std::string m_sEnvMapNames[6];
	GLuint m_uiEnvMap;
	CUBEMAPLOADCMD m_buttonInterfaceCmd[6];

	// Camera Control
	bool m_bIsCamRotate, m_bIsInRotate;
	float m_fCamSenX, m_fCamSenY;
	float m_fZoomSen, m_fScrollY;
	float m_fZoomMin, m_fZoomMax;
	glm::vec2 m_v2LastCursorPos;
	glm::vec2 m_v2CurrentCursorPos;

	//Model
	NPMathHelper::Vec3 m_v3ModelPos;
	float m_fModelScale;
	NPMathHelper::Quat m_v3ModelRot;

	//AntTweakBar
	int m_iScrollingTemp;

	//Other object
	//Sky
	NPGLHelper::Effect* m_pSkyboxEffect;
	NPGLHelper::RenderObject m_skybox;

	//Floor
	bool m_bIsShowFloor;
	NPGLHelper::Effect* m_pBlinnPhongNormalModelEffect;
	NPGLHelper::Effect* m_pDiffuseNormalModelEffect;
	NPGLHelper::Effect* m_pBlinnPhongNormalEnvSModelEffect;
	NPGLHelper::Effect* m_pDiffuseNormalEnvSModelEffect;
	NPGLHelper::RenderObject m_floor;
	GLuint m_iFloorTex;
	GLuint m_iFloorNormalTex;
	Material m_floorMaterial;
	BRDFModel::SphericalSpace m_floorSpace;

	//Recording
	GLuint m_uiRECFBO;
	GLuint m_uiRECCB;
	GLuint m_uiRECDB;
	RECORD_STATUS m_recStatus;
	float m_fRecFPS;
	float m_fRecCirSec;
	float m_uiRecCurFrame;
	std::string m_sRecStorePath;
};

#endif