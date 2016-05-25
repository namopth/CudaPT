#ifndef CUDAPT_H
#define CUDAPT_H

#define APP_NAME "Cuda Ray Tracing Lab"
#define APP_AUTH "Namo Podee"
#define APP_DESC "Ray Tracing Program for Experiment"
#define APP_VERS "1.0.0"
#define APP_DATE __DATE__

#include "glhelper.h"
#include "camhelper.h"
#include "mathhelper.h"
#include "macrohelper.h"
#include "raytracer.h"

#define ADAPCHEAT

class CUDAPTWindow : public NPGLHelper::Window
{
public:
	CUDAPTWindow(const char* name, const int sizeW = 800, const int sizeH = 600);
	virtual ~CUDAPTWindow();

	virtual int OnInit();
	virtual int OnTick(const float deltaTime);
	virtual void OnTerminate();
	virtual bool ShouldTerminateProgramOnTerminate() { return true; }
	virtual void OnHandleInputMSG(const INPUTMSG &msg);

	void BrowseModel();
	void BrowseAndSaveResult();

	void BrowseEnvSetting();
	void BrowseAndSaveEnvSetting();

	void ChooseResultAsConvergedResult();
	bool SetConvergedResult(const float* data);
	void ClearConvergedResult();
	void ExportConvergedResult();

	void ToggleCollectRMSE();
	void CalculateRMSE();

	inline RTRenderer* GetRenderer() { return &m_raytracer; }

protected:
	std::vector<float> m_uiPTResultData;
	GLuint m_uiPTResultTex;

	NPGLHelper::Effect* m_pFinalComposeEffect;
	float m_fExposure;

	NPCamHelper::FlyCamera m_cam;
	bool m_bIsMLBClicked;
	NPMathHelper::Vec2 m_v2ClickedPos;
	bool m_bIsMRBHeld;
	bool m_bIsCamRotate;
	NPMathHelper::Vec3 m_v3CamMoveDir;
	float m_fCamSenX, m_fCamSenY;
	float m_fCamMoveSpeed;
	RTScene m_scene;
	RTRenderer m_raytracer;

	uint32 m_uDeltaTimeSec;
	float m_fFPS;
	bool m_bIsTracing;
	bool m_bIsWireFrame;
	bool m_bIsSceneGUI;

	float m_fScrollY;
	NPMathHelper::Vec2 m_v2LastCursorPos;
	NPMathHelper::Vec2 m_v2CurrentCursorPos;

	// RMSE Comparison
	float m_fRMSECaptureLimit;
	uint32 m_uiRMSECaptureSPP;
	uint32 m_uiRMSECurSPP;
	bool m_bIsRMSECapturing;
	float m_fRMSECaptureSecTime;
	float m_fRMSECaptureElapSecTime;
	float* m_pCapturedConvergedResult;
	bool m_bIsShowCapturedConvergedResult;
	bool m_bIsCapturedConvergedResultValid;
	float m_fRMSEResult;

	void RenderScreenQuad();
	GLuint m_uiVBOQuad;
	GLuint m_uiVAOQuad;
};

#endif