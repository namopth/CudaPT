#ifndef BRDFVISUALIZER_H
#define BRDFVISUALIZER_H

#include "glhelper.h"
#include "camhelper.h"
#include "mathhelper.h"
#include "macrohelper.h"

class BRDFVisualizer : public NPGLHelper::Window
{
public:
	BRDFVisualizer(const char* name, const int sizeW = 800, const int sizeH = 600);
	virtual ~BRDFVisualizer();

	virtual int OnInit();
	virtual int OnTick(const float deltaTime);
	virtual void OnTerminate(); 
	virtual bool ShouldTerminateProgramOnTerminate() { return true; }
	virtual void OnHandleInputMSG(const INPUTMSG &msg);

	void OpenBRDFData();
	void OpenModelWindow();

protected:
	GLuint m_iBRDFEstTex;
	bool m_bIsLoadTexture;
	std::string m_sBRDFTextureName;
	NPGLHelper::Effect* m_pBRDFVisEffect;
	NPGLHelper::RenderObject testObject;
	NPCamHelper::RotateCamera m_Cam;

	bool m_bIsWireFrame;
	bool m_bIsSceneGUI;
	bool m_bIsCamRotate, m_bIsInRotate;
	float m_fCamSenX, m_fCamSenY;
	float m_fInSenX, m_fInSenY;
	float m_fInPitch, m_fInYaw;
	float m_fZoomSen, m_fScrollY;
	float m_fZoomMin, m_fZoomMax;
	glm::vec2 m_v2LastCursorPos;
	glm::vec2 m_v2CurrentCursorPos;
	NPGLHelper::DebugLine m_InLine;
	NPGLHelper::DebugLine m_AxisLine[3];
	std::string m_sBRDFFilePath;
	unsigned int m_uiNPH;
	unsigned int m_uiNTH;
	unsigned int m_uiModelWindowWSize;
	unsigned int m_uiModelWindowHSize;

	unsigned int m_uiModelWindowID;
};

#endif