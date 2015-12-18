#include "brdfvisualizer.h"

#include <iostream>
#include <string>

#include "geohelper.h"
#include "oshelper.h"
#include "atbhelper.h"

#include "ModelViewWindow.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600


int main()
{
	NPGLHelper::App mainApp;
	return mainApp.Run(new BRDFVisualizer("BRDF Visualizer", WINDOW_WIDTH, WINDOW_HEIGHT));
}

void TW_CALL BRDFButton(void * window)
{
	BRDFVisualizer* appWin = (BRDFVisualizer*)window;
	if (appWin)
		appWin->OpenBRDFData();
}

void TW_CALL ModelButton(void * window)
{
	BRDFVisualizer* appWin = (BRDFVisualizer*)window;
	if (appWin)
		appWin->OpenModelWindow();
}


BRDFVisualizer::BRDFVisualizer(const char* name, const int sizeW, const int sizeH)
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
	, m_fZoomMax(4.0f)
	, m_fZoomSen(0.1f)
	, m_pBRDFVisEffect(nullptr)
	, m_bIsLoadTexture(false)
	, m_uiModelWindowID(0)
	, m_sBRDFFilePath("")
	, m_sBRDFTextureName("None")
	, m_uiNPH(64)
	, m_uiNTH(16)
	, m_bIsWireFrame(true)
	, m_bIsSceneGUI(true)
	, m_uiModelWindowWSize(1600)
	, m_uiModelWindowHSize(900)
{
}

BRDFVisualizer::~BRDFVisualizer()
{
}

int BRDFVisualizer::OnInit()
{
	// AntTweakBar Init
	ATB_ASSERT(NPTwInit(m_uiID, TW_OPENGL_CORE, nullptr));
	ATB_ASSERT(TwSetCurrentWindow(m_uiID));
	ATB_ASSERT(TwWindowSize(m_iSizeW, m_iSizeH));
	TwBar* mainBar = TwNewBar("BRDFVisualizer");
	ATB_ASSERT(TwDefine(" BRDFVisualizer help='These properties defines the application behavior' "));
	ATB_ASSERT(TwAddButton(mainBar, "openbrdf", BRDFButton, this, "label='Browse File' group='BRDF File'"));
	ATB_ASSERT(TwAddVarRO(mainBar, "brdfname", TW_TYPE_STDSTRING, &m_sBRDFTextureName,
		" label='Loaded BRDF' help='Loaded BRDF' group='BRDF File'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "n_ph", TW_TYPE_UINT32, &m_uiNPH,
		" label='BRDF File N_PH' step=1 keyIncr=s keyDecr=S help='N_PH' group='BRDF File'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "n_th", TW_TYPE_UINT32, &m_uiNTH,
		" label='BRDF File N_TH' step=1 keyIncr=d keyDecr=D help='N_TH' group='BRDF File'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "wireframe", TW_TYPE_BOOLCPP, &m_bIsWireFrame,
		" label='Wireframe' help='Show Wireframe' group='Display'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "scenegui", TW_TYPE_BOOLCPP, &m_bIsSceneGUI,
		" label='Scene GUI' help='Show Scene GUI' group='Display'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Window Width", TW_TYPE_UINT32, &m_uiModelWindowWSize,
		"step=1 keyIncr=s keyDecr=S help='N_PH' group='3D Model'"));
	ATB_ASSERT(TwAddVarRW(mainBar, "Window Height", TW_TYPE_UINT32, &m_uiModelWindowHSize,
		"step=1 keyIncr=s keyDecr=S help='N_PH' group='3D Model'"));
	ATB_ASSERT(TwAddButton(mainBar, "showmodelview", ModelButton, this, "label='Show View' group='3D Model'"));
	ATB_ASSERT(TwAddSeparator(mainBar, "modelviewsep", ""));
	ATB_ASSERT(TwAddButton(mainBar, "instruction1", NULL, NULL, "label='LClick+Drag: Rot Light dir'"));
	ATB_ASSERT(TwAddButton(mainBar, "instruction2", NULL, NULL, "label='RClick+Drag: Rot Camera dir'"));
	ATB_ASSERT(TwAddButton(mainBar, "instruction3", NULL, NULL, "label='Scroll: Zoom Camera in/out'"));

	m_AxisLine[0].Init(m_pShareContent);
	m_AxisLine[1].Init(m_pShareContent);
	m_AxisLine[2].Init(m_pShareContent);
	m_InLine.Init(m_pShareContent);

	m_pBRDFVisEffect = m_pShareContent->GetEffect("BRDFVisEffect");
	if (!m_pBRDFVisEffect->GetIsLinked())
	{
		m_pBRDFVisEffect->initEffect();
		m_pBRDFVisEffect->attachShaderFromFile("../shader/BRDFVisualizeVS.glsl", GL_VERTEX_SHADER);
		m_pBRDFVisEffect->attachShaderFromFile("../shader/BRDFVisualizePS.glsl", GL_FRAGMENT_SHADER);
		m_pBRDFVisEffect->linkEffect();
	}

	testObject.SetGeometry(NPGeoHelper::GetSlicedHemisphereShape(1.f, 64, 64));

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	return 0;
}

int BRDFVisualizer::OnTick(const float deltaTime)
{
	//DEBUG_COUT("BRDFVisualizer::OnTick BGN");
	// Camera control - bgn
	glm::vec2 cursorMoved = m_v2CurrentCursorPos - m_v2LastCursorPos;
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
	if (abs(m_fScrollY) > 1E-9)
	{
		float curZoom = m_Cam.GetRadius();
		curZoom += m_fScrollY * m_fZoomSen;
		curZoom = (curZoom < m_fZoomMin) ? m_fZoomMin : (curZoom > m_fZoomMax) ? m_fZoomMax : curZoom;
		m_Cam.SetRadius(curZoom);
		m_fScrollY = 0.f;
	}
	m_v2LastCursorPos = m_v2CurrentCursorPos;
	// Camera control - end

	glm::mat4 proj, view, model;
	proj = glm::perspective(45.0f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
	view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
	model = glm::rotate(model, 0.0f, glm::vec3(0.0f, 1.0f, 0.0f));
	NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if (m_bIsLoadTexture)
	{
		m_pBRDFVisEffect->activeEffect();
		m_pBRDFVisEffect->SetInt("n_th", m_uiNTH);
		m_pBRDFVisEffect->SetInt("n_ph", m_uiNPH);
		m_pBRDFVisEffect->SetFloat("i_yaw", m_fInYaw);
		m_pBRDFVisEffect->SetFloat("i_pitch", m_fInPitch);
		m_pBRDFVisEffect->SetMatrix("projection", myProj.GetDataColumnMajor());
		m_pBRDFVisEffect->SetMatrix("view", m_Cam.GetViewMatrix());
		m_pBRDFVisEffect->SetMatrix("model", glm::value_ptr(model));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_iBRDFEstTex);
		m_pBRDFVisEffect->SetInt("brdfTexture", 0);

		glBindVertexArray(testObject.GetVAO());
		glDrawElements(GL_TRIANGLES, testObject.GetIndicesSize(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
		glActiveTexture(GL_TEXTURE0);
		m_pBRDFVisEffect->deactiveEffect();
	}

	if (m_bIsSceneGUI)
	{
		m_AxisLine[0].Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(1.f, 0.f, 0.f), NPMathHelper::Vec3(1.0f, 0.f, 0.f)
			, m_Cam.GetViewMatrix(), glm::value_ptr(proj));
		m_AxisLine[1].Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(0.f, 1.f, 0.f), NPMathHelper::Vec3(0.0f, 1.f, 0.f)
			, m_Cam.GetViewMatrix(), glm::value_ptr(proj));
		m_AxisLine[2].Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(0.f, 0.f, 1.f), NPMathHelper::Vec3(0.0f, 0.f, 1.f)
			, m_Cam.GetViewMatrix(), glm::value_ptr(proj));
	}

	if (m_bIsSceneGUI)
	{
		glm::vec3 InLineEnd;
		InLineEnd.y = sin(m_fInYaw) * 10.f;
		InLineEnd.x = cos(m_fInYaw) * sin(m_fInPitch) * 10.f;
		InLineEnd.z = cos(m_fInYaw) * cos(m_fInPitch) * 10.f;
		m_InLine.Draw(NPMathHelper::Vec3(), NPMathHelper::Vec3(InLineEnd.x, InLineEnd.y, InLineEnd.z), NPMathHelper::Vec3(1.0f, 1.f, 1.f)
			, m_Cam.GetViewMatrix(), glm::value_ptr(proj));
	}

	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		ATB_ASSERT(TwSetCurrentWindow(m_uiID));
		ATB_ASSERT(TwDraw());
	}

	glfwSwapBuffers(GetGLFWWindow());

	//DEBUG_COUT("BRDFVisualizer::OnTick END");
	return 0;
}

void BRDFVisualizer::OnTerminate()
{
	testObject.ClearGeometry();

	NPTwTerminate(m_uiID);
}

void BRDFVisualizer::OnHandleInputMSG(const INPUTMSG &msg)
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
		//	OpenBRDFData();
		//if (msg.key == GLFW_KEY_M && msg.action == GLFW_PRESS)
		//	OpenModelWindow();
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
		m_fScrollY = msg.yoffset;
		break;
	}
}

void BRDFVisualizer::OpenBRDFData()
{
	std::string file = NPOSHelper::BrowseFile("All\0*.*\0Text\0*.TXT\0");
	if (file.empty())
		return;

	int width, height;
	//if (!NPGLHelper::loadTextureFromFile(file.c_str(), m_iBRDFEstTex, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST, false))
	if (!NPGLHelper::loadHDRTextureFromFile(file.c_str(), m_iBRDFEstTex, GL_REPEAT, GL_REPEAT, GL_NEAREST, GL_NEAREST))
	{
		std::string message = "Cannot load file ";
		message = message + file;
		NPOSHelper::CreateMessageBox(message.c_str(), "Load BRDF Data Failure", NPOSHelper::MSGBOX_OK);
		return;
	}

	m_sBRDFFilePath = m_sBRDFTextureName = file;
	m_bIsLoadTexture = true;

	ModelViewWindow* modelViewWindow = (ModelViewWindow*)GetOwner()->GetWindow(m_uiModelWindowID);
	if (modelViewWindow)
	{
		modelViewWindow->SetBRDFData(m_sBRDFFilePath.c_str(), m_uiNTH, m_uiNPH);
	}
}

void BRDFVisualizer::OpenModelWindow()
{
	if (!(m_uiModelWindowID > 0 && GetOwner() && GetOwner()->GetIsWindowActive(m_uiModelWindowID)))
	{
		m_uiModelWindowID = GetOwner()->AttachWindow(new ModelViewWindow("Model View", m_uiModelWindowWSize, m_uiModelWindowHSize));
	}

	ModelViewWindow* modelViewWindow = (ModelViewWindow*)GetOwner()->GetWindow(m_uiModelWindowID);
	if (modelViewWindow)
	{
		modelViewWindow->SetBRDFData(m_sBRDFFilePath.c_str(), m_uiNTH, m_uiNPH);
	}
}