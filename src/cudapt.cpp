#include "cudapt.h"

#include <iostream>
#include <string>

#include "atbhelper.h"
#include "oshelper.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

int main()
{
	NPGLHelper::App mainApp;
	return mainApp.Run(new CUDAPTWindow("Cuda PT", WINDOW_WIDTH, WINDOW_HEIGHT));
}

void TW_CALL TWBrowseModel(void * window)
{
	CUDAPTWindow* appWin = (CUDAPTWindow*)window;
	if (appWin)
		appWin->BrowseModel();
}

void TW_CALL SetRenderingMethodCallback(const void *value, void *clientData)
{
	CUDAPTWindow* appWin = (CUDAPTWindow*)clientData;
	if (appWin)
	{
		RTRenderer::RENDERER_MODE method = *((RTRenderer::RENDERER_MODE*) value);
		appWin->GetRenderer()->SetRendererMode(method);
	}
}
void TW_CALL GetRenderingMethodCallback(void *value, void *clientData)
{
	CUDAPTWindow* appWin = (CUDAPTWindow*)clientData;
	if (appWin)
	{
		*(RTRenderer::RENDERER_MODE*)value = appWin->GetRenderer()->GetRendererMode();
	}
}

CUDAPTWindow::CUDAPTWindow(const char* name, const int sizeW, const int sizeH)
	: Window(name, sizeW, sizeH)
	, m_pFinalComposeEffect(0)
	, m_fExposure(1.0f)
	, m_bIsTracing(true)
	, m_bIsWireFrame(false)
	, m_bIsSceneGUI(true)
	, m_uiVBOQuad(0)
	, m_uiVAOQuad(0)
	, m_cam()
	, m_bIsCamRotate(false)
	, m_fCamSenX(0.01f)
	, m_fCamSenY(.005f)
	, m_fCamMoveSpeed(20.0f)
	, m_uDeltaTimeSec(0)
	, m_fFPS(0.f)
{

}

CUDAPTWindow::~CUDAPTWindow()
{

}

int CUDAPTWindow::OnInit()
{

	m_raytracer.Init(m_iSizeW, m_iSizeH);

	m_uiPTResultData.resize(m_iSizeW * m_iSizeH * 3);
	for (int i = 0; i < m_iSizeW; i++)
	{
		for (int j = 0; j < m_iSizeH; j++)
		{
			int ind = (j * m_iSizeW + i) * 3;
			m_uiPTResultData[ind] = (float)i / (float)m_iSizeW;
			m_uiPTResultData[ind + 1] = (float)j / (float)m_iSizeH;
			m_uiPTResultData[ind + 2] = 1.f;
		}
	}

	// AntTweakBar Init
	ATB_ASSERT(NPTwInit(m_uiID, TW_OPENGL_CORE, nullptr));
	ATB_ASSERT(TwSetCurrentWindow(m_uiID));
	ATB_ASSERT(TwWindowSize(m_iSizeW, m_iSizeH));
	TwBar* mainBar = TwNewBar("CUDAPT");
	ATB_ASSERT(TwDefine(" CUDAPT help='These properties defines the application behavior' "));
	ATB_ASSERT(TwAddVarRW(mainBar, "Tracer_Enabled", TW_TYPE_BOOLCPP, &m_bIsTracing, "group='Tracer' label='Enable'"));

	TwEnumVal renderEV[] = {
		{ RTRenderer::RENDERER_MODE_CPU_DEBUG, "CPU DEBUG" },
		{ RTRenderer::RENDERER_MODE_CUDA_DEBUG, "CUDA DEBUG" },
		{ RTRenderer::RENDERER_MODE_CUDA_DEBUG_BVH, "CUDA DEBUG BVH" },
		{ RTRenderer::RENDERER_MODE_CUDA_PT, "CUDA PT" }
	};
	TwType renderType = TwDefineEnum("Rendering Method", renderEV, RTRenderer::RENDERER_MODE_N);
	TwAddVarCB(mainBar, "Rendering", renderType, SetRenderingMethodCallback, GetRenderingMethodCallback, this
		, " label='Rendering Method' help='Set Rendering Method' group='Tracer'");

	ATB_ASSERT(TwAddButton(mainBar, "addmodel", TWBrowseModel, this, "label='Add Model' group='Scene'"));
	ATB_ASSERT(TwAddVarRO(mainBar, "Frame Time", TW_TYPE_UINT32, &m_uDeltaTimeSec,
		" label='Time(ms)' group='Render Info'"));
	ATB_ASSERT(TwAddVarRO(mainBar, "Frame FPS", TW_TYPE_FLOAT, &m_fFPS,
		" label='FPS' group='Render Info'"));

	m_pFinalComposeEffect = m_pShareContent->GetEffect("FinalComposeEffect");
	if (!m_pFinalComposeEffect->GetIsLinked())
	{
		m_pFinalComposeEffect->initEffect();
		m_pFinalComposeEffect->attachShaderFromFile("..\\shader\\FinalComposeVS.glsl", GL_VERTEX_SHADER);
		m_pFinalComposeEffect->attachShaderFromFile("..\\shader\\FinalComposePS.glsl", GL_FRAGMENT_SHADER);
		m_pFinalComposeEffect->linkEffect();
	}

	glGenTextures(1, &m_uiPTResultTex);
	glBindTexture(GL_TEXTURE_2D, m_uiPTResultTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, m_iSizeW, m_iSizeH, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	return 0;
}

int CUDAPTWindow::OnTick(const float deltaTime)
{
	//DEBUG_COUT("BRDFVisualizer::OnTick BGN");
	m_uDeltaTimeSec = (uint32)(deltaTime * 1000.f);
	m_fFPS = (deltaTime > M_EPSILON) ? 1.f / deltaTime : 0.f;

	// Camera control - bgn
	NPMathHelper::Vec2 cursorMoved = m_v2CurrentCursorPos - m_v2LastCursorPos;
	if (m_bIsCamRotate && m_bIsMRBHeld)
	{
		m_cam.AddPitch(-cursorMoved._x * m_fCamSenX);
		m_cam.AddYaw(-cursorMoved._y * m_fCamSenY);
	}
	m_bIsCamRotate = m_bIsMRBHeld;
	m_v2LastCursorPos = m_v2CurrentCursorPos;
	// Camera control - end

	if (m_v3CamMoveDir.length() > M_EPSILON)
	{
		m_cam.AddPosForward(m_v3CamMoveDir.normalize() * m_fCamMoveSpeed * deltaTime);
	}
	m_cam.UpdateViewMatrix();

	glm::mat4 proj, view, model;
	proj = glm::perspective(45.0f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);
	view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
	model = glm::rotate(model, 0.0f, glm::vec3(0.0f, 1.0f, 0.0f));
	NPMathHelper::Mat4x4 myProj = NPMathHelper::Mat4x4::perspectiveProjection(M_PI * 0.5f, (float)m_iSizeW / (float)m_iSizeH, 0.1f, 100.0f);

	// Rendering - bgn
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (m_bIsWireFrame)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	const float* traceResult = nullptr;
	if (m_bIsTracing)
	{
		m_raytracer.Render(m_cam.GetPos(), m_cam.GetDir()
			, m_cam.GetUp(), M_PI_2 * 0.5f, m_scene);
		traceResult = m_raytracer.GetResult();
	}

	m_pFinalComposeEffect->activeEffect();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_uiPTResultTex);
	if (traceResult)
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_iSizeW, m_iSizeH, GL_RGB, GL_FLOAT, traceResult);
	m_pFinalComposeEffect->SetInt("hdrBuffer", 0);
	m_pFinalComposeEffect->SetFloat("exposure", m_fExposure);
	RenderScreenQuad();
	glBindTexture(GL_TEXTURE_2D, 0);
	m_pFinalComposeEffect->deactiveEffect();

	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		ATB_ASSERT(TwSetCurrentWindow(m_uiID));
		ATB_ASSERT(TwDraw());
	}

	glfwSwapBuffers(GetGLFWWindow());

	// Rendering - end
	//DEBUG_COUT("BRDFVisualizer::OnTick END");
	return 0;
}

void CUDAPTWindow::OnTerminate()
{
	NPTwTerminate(m_uiID);
}

void CUDAPTWindow::OnHandleInputMSG(const INPUTMSG &msg)
{
	ATB_ASSERT(TwSetCurrentWindow(m_uiID));
	switch (msg.type)
	{
	case Window::INPUTMSG_KEYBOARDKEY:
		if (TwEventCharGLFW(msg.key, msg.action))
			break;
		if (msg.key == GLFW_KEY_ESCAPE && msg.action == GLFW_PRESS)
			glfwSetWindowShouldClose(m_pWindow, GL_TRUE);
		else if (msg.key == GLFW_KEY_W)
			m_v3CamMoveDir._z -= (msg.action == GLFW_PRESS) ? 1.f : (msg.action == GLFW_RELEASE) ? - 1.f : 0.f;
		else if (msg.key == GLFW_KEY_S)
			m_v3CamMoveDir._z += (msg.action == GLFW_PRESS) ? 1.f : (msg.action == GLFW_RELEASE) ? -1.f : 0.f;
		else if (msg.key == GLFW_KEY_A)
			m_v3CamMoveDir._x -= (msg.action == GLFW_PRESS) ? 1.f : (msg.action == GLFW_RELEASE) ? -1.f : 0.f;
		else if (msg.key == GLFW_KEY_D)
			m_v3CamMoveDir._x += (msg.action == GLFW_PRESS) ? 1.f : (msg.action == GLFW_RELEASE) ? -1.f : 0.f;
		break;
	case Window::INPUTMSG_MOUSEKEY:
		if (TwEventMouseButtonGLFW(msg.key, msg.action))
		{
			if (msg.key != GLFW_MOUSE_BUTTON_RIGHT || !m_bIsMRBHeld) break;
		}
		if (msg.key == GLFW_MOUSE_BUTTON_RIGHT)
			m_bIsMRBHeld = (msg.action == GLFW_PRESS);
		break;
	case Window::INPUTMSG_MOUSECURSOR:
		TwEventMousePosGLFW(msg.xpos, msg.ypos);
		m_v2CurrentCursorPos._x = msg.xpos;
		m_v2CurrentCursorPos._y = msg.ypos;
		break;
	case Window::INPUTMSG_MOUSESCROLL:
		m_fScrollY = msg.yoffset;
		break;
	}
}

void CUDAPTWindow::BrowseModel()
{
	std::string file = NPOSHelper::BrowseFile("All\0*.*\0Text\0*.TXT\0");
	if (file.empty())
		return;
	if (!m_scene.AddModel(file.c_str())){
		std::string message = "Cannot load file ";
		message = message + file;
		NPOSHelper::CreateMessageBox(message.c_str(), "Load Model Data Failure", NPOSHelper::MSGBOX_OK);
		return;
	}
}

void CUDAPTWindow::RenderScreenQuad()
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
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	}
	glBindVertexArray(m_uiVAOQuad);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}