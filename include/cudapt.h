#ifndef CUDAPT_H
#define CUDAPT_H

#include "glhelper.h"
#include "camhelper.h"
#include "mathhelper.h"
#include "macrohelper.h"
#include "raytracer.h"

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

protected:
	std::vector<float> m_uiPTResultData;
	GLuint m_uiPTResultTex;

	NPGLHelper::Effect* m_pFinalComposeEffect;
	float m_fExposure;

	NPCamHelper::FlyCamera m_cam;
	bool m_bIsCamRotate;
	NPMathHelper::Vec3 m_v3CamMoveDir;
	float m_fCamSenX, m_fCamSenY;
	float m_fCamMoveSpeed;
	RTScene m_scene;
	RTRenderer m_raytracer;

	bool m_bIsTracing;
	bool m_bIsWireFrame;
	bool m_bIsSceneGUI;

	float m_fScrollY;
	NPMathHelper::Vec2 m_v2LastCursorPos;
	NPMathHelper::Vec2 m_v2CurrentCursorPos;

	void RenderScreenQuad();
	GLuint m_uiVBOQuad;
	GLuint m_uiVAOQuad;
};

#endif