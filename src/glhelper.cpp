#include "glhelper.h"
#include "macrohelper.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <SOIL.h>

#include "hdrhelper.h"

GLEWContext* glewGetContext()
{
	if (NPGLHelper::App::g_pMainApp)
	{
		NPGLHelper::Window* curWin = NPGLHelper::App::g_pMainApp->GetCurrentWindow();
		return (curWin) ? curWin->GetGLEWContext() : nullptr;
	}
	return nullptr;
}

namespace NPGLHelper
{
	bool loadASCIIFromFile(std::string file, std::string &content)
	{
		std::ifstream t(file, std::ios::in);
		if (!t.good())
			return false;

		std::stringstream buffer;
		buffer << t.rdbuf();
		content = buffer.str();
		t.close();
		t.clear();
		return true;
	}

	bool createShaderFromFile(std::string file, GLuint type, GLuint &result)
	{
		std::string shaderSource;
		if (loadASCIIFromFile(file, shaderSource))
		{
			//std::cout << "Loaded shader" << std::endl << shaderSource << std::endl;
			GLuint shader;
			shader = glCreateShader(type);
			const char *shaderSource_cstr = shaderSource.c_str();
			glShaderSource(shader, 1, &shaderSource_cstr, NULL);
			glCompileShader(shader);
			result = shader;

			std::string info;
			if (!checkShaderError(shader, GL_COMPILE_STATUS,info))
			{
				DEBUG_COUT("[!!!]SHADER::COMPILATION_FAILED " << file << std::endl << info);
				return false;
			}
			else
			{
				DEBUG_COUT("SHADER:COMPILATION_SUCCEED " << file);
			}
			return true;
		}
		return false;
	}

	bool checkShaderError(GLuint shader, GLuint checking, std::string &info)
	{
		GLint success;
		GLchar infoLog[512];
		glGetShaderiv(shader, checking, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 512, NULL, infoLog);
			info = infoLog;
		}
		return success != 0;
	}

	bool checkProgramError(GLuint program, GLuint checking, std::string &info)
	{
		GLint success;
		GLchar infoLog[512];
		glGetProgramiv(program, checking, &success);
		if (!success)
		{
			glGetProgramInfoLog(program, 512, NULL, infoLog);
			info = infoLog;
		}
		return success != 0;
	}

	bool loadHDRTextureFromFile(const char* path, GLuint &id, GLint warpS, GLint warpT, GLint minFil, GLint maxFil)
	{
		int width, height;
		FILE* f;
		fopen_s(&f, path, "rb");
		NPHDRHelper::RGBE_ReadHeader(f, &width, &height, NULL);
		float *image = (float *)malloc(sizeof(float) * 3 * width*height);
		NPHDRHelper::RGBE_ReadPixels_RLE(f, image, width, height);
		fclose(f);

		glGenTextures(1, &id);
		glBindTexture(GL_TEXTURE_2D, id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, image);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
		return true;
	}

	bool loadTextureFromFile(const char* path, GLuint &id, GLint warpS, GLint warpT, GLint minFil, GLint maxFil, bool sRGB)
	{
		int width, height;
		unsigned char* image = SOIL_load_image(path, &width, &height, 0, SOIL_LOAD_RGB);
		if (!image)
		{
			DEBUG_COUT(SOIL_last_result());
			return false;
		}

		glGenTextures(1, &id);
		glBindTexture(GL_TEXTURE_2D, id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, (sRGB) ? GL_SRGB : GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
		glGenerateMipmap(GL_TEXTURE_2D);
		SOIL_free_image_data(image);
		glBindTexture(GL_TEXTURE_2D, 0);
		return true;
	}

	bool loadCubemapFromFiles(std::string faces[6], GLuint &id, bool sRGB)
	{
		glGenTextures(1, &id); CHECK_GL_ERROR;
		glActiveTexture(GL_TEXTURE0); CHECK_GL_ERROR;

		int width, height;
		unsigned char* image;

		glBindTexture(GL_TEXTURE_CUBE_MAP, id); CHECK_GL_ERROR;
		for (unsigned int i = 0; i < 6; i++)
		{
			image = SOIL_load_image(faces[i].c_str(), &width, &height, 0, SOIL_LOAD_RGB); 
			if (!image)
			{
				return false;
			}
			glTexImage2D(
				GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
				(sRGB) ? GL_SRGB : GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); CHECK_GL_ERROR;
			SOIL_free_image_data(image);
		}
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR); CHECK_GL_ERROR;
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR); CHECK_GL_ERROR;
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); CHECK_GL_ERROR;
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); CHECK_GL_ERROR;
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); CHECK_GL_ERROR;

		glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
		return true;
	}


	bool saveScreenShotBMP(const char* filename, const float width, const float height)
	{
		int save_result = SOIL_save_screenshot(filename, SOIL_SAVE_TYPE_BMP, 0, 0, width, height);
		if (!save_result)
		{
			DEBUG_COUT(SOIL_last_result());
		}
		return save_result;
	}

	RenderObject::RenderObject()
		: m_iVAO(-1)
		, m_iVBO(-1)
		, m_iEBO(-1)
		, m_iIndicesSize(0)
	{

	}

	RenderObject::~RenderObject()
	{

	}

	void RenderObject::SetGeometry(const NPGeoHelper::Geometry& geo, unsigned int type)
	{
		if (type == 0)
		{
			std::vector<GLfloat> vertices;
			for (auto it = geo.vertices.begin(); it != geo.vertices.end(); it++)
			{
				vertices.push_back(it->pos._x);
				vertices.push_back(it->pos._y);
				vertices.push_back(it->pos._z);
				vertices.push_back(it->tex._x);
				vertices.push_back(it->tex._y);
			}

			m_iIndicesSize = geo.indices.size();
			std::vector<GLuint> indices;
			for (auto it = geo.indices.begin(); it != geo.indices.end(); it++)
			{
				indices.push_back(*it);
			}


			if (m_iVAO >= 0)
			{
				ClearGeometry();
			}

			glGenBuffers(1, &m_iVBO);
			glGenBuffers(1, &m_iEBO);
			glGenVertexArrays(1, &m_iVAO);
			glBindVertexArray(m_iVAO);
			glBindBuffer(GL_ARRAY_BUFFER, m_iVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iEBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glBindVertexArray(0);
		}
		else if (type == 1)
		{
			std::vector<GLfloat> vertices;
			for (auto it = geo.vertices.begin(); it != geo.vertices.end(); it++)
			{
				vertices.push_back(it->pos._x);
				vertices.push_back(it->pos._y);
				vertices.push_back(it->pos._z);
				vertices.push_back(it->norm._x);
				vertices.push_back(it->norm._y);
				vertices.push_back(it->norm._z);
				vertices.push_back(it->tan._x);
				vertices.push_back(it->tan._y);
				vertices.push_back(it->tan._z);
				vertices.push_back(it->tex._x);
				vertices.push_back(it->tex._y);
			}

			m_iIndicesSize = geo.indices.size();
			std::vector<GLuint> indices;
			for (auto it = geo.indices.begin(); it != geo.indices.end(); it++)
			{
				indices.push_back(*it);
			}


			if (m_iVAO >= 0)
			{
				ClearGeometry();
			}

			glGenBuffers(1, &m_iVBO);
			glGenBuffers(1, &m_iEBO);
			glGenVertexArrays(1, &m_iVAO);
			glBindVertexArray(m_iVAO);
			glBindBuffer(GL_ARRAY_BUFFER, m_iVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iEBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
			glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 11 * sizeof(GLfloat), (GLvoid*)(9 * sizeof(GLfloat)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);
			glEnableVertexAttribArray(2);
			glEnableVertexAttribArray(3);
			glBindVertexArray(0);
		}
	}

	void RenderObject::ClearGeometry()
	{
		if (m_iVAO >= 0)
		{
			glDeleteVertexArrays(1, &m_iVAO);
		}
		m_iVAO = -1;

		if (m_iVBO >= 0)
		{
			glDeleteBuffers(1, &m_iVBO);
		}
		m_iVBO = -1;

		if (m_iEBO >= 0)
		{
			glDeleteBuffers(1, &m_iEBO);
		}
		m_iEBO = -1;
	}

	Effect::Effect()
		: m_bIsLinked(false)
		, m_iProgram(0)
	{

	}

	Effect::~Effect()
	{
		deleteAttachedShaders();

		if (m_iProgram >= 0)
		{
			glDeleteProgram(m_iProgram);
			m_iProgram = -1;
		}
	}

	void Effect::initEffect()
	{
		if (m_iProgram == 0)
			m_iProgram = glCreateProgram();
	}

	void Effect::attachShaderFromFile(const char* filename, const GLuint type)
	{
		assert(m_iProgram >= 0);
		GLuint shader;
		if (NPGLHelper::createShaderFromFile(filename, type, shader))
		{
			glAttachShader(m_iProgram, shader);
			m_vAttachedShader.push_back(shader);
		}
	}

	void Effect::deleteAttachedShaders()
	{
		for (auto it = m_vAttachedShader.begin(); it != m_vAttachedShader.end(); it++)
		{
			glDeleteShader(*it);
		}
		m_vAttachedShader.clear();
	}

	bool Effect::linkEffect()
	{
		assert(m_iProgram >= 0);
		glLinkProgram(m_iProgram);
		deleteAttachedShaders();

		std::string pLinkInfo;
		if (!NPGLHelper::checkProgramError(m_iProgram, GL_LINK_STATUS, pLinkInfo))
		{
			DEBUG_COUT("[!!!]SHADER::LINK_FAILED" << std::endl << pLinkInfo);
			return false;
		}
		else
		{
			DEBUG_COUT("SHADER::LINK_SUCCEED");
		}

		m_bIsLinked = true;

		return true;
	}

	bool Effect::activeEffect()
	{
		assert(m_iProgram >= 0);
		glUseProgram(m_iProgram);

		return true;
	}

	bool Effect::deactiveEffect()
	{
		glUseProgram(0);
		return true;
	}

	void Effect::SetMatrix(const char* var, const float* mat)
	{
		assert(m_iProgram >= 0);
		GLuint matLoc = glGetUniformLocation(m_iProgram, var);
		glUniformMatrix4fv(matLoc, 1, GL_FALSE, mat);
	}

	void Effect::SetInt(const char* var, const int value)
	{
		assert(m_iProgram >= 0);
		GLuint valueLoc = glGetUniformLocation(m_iProgram, var);
		glUniform1i(valueLoc, value);
	}

	void Effect::SetFloat(const char* var, const float value)
	{
		assert(m_iProgram >= 0);
		GLuint valueLoc = glGetUniformLocation(m_iProgram, var);
		glUniform1f(valueLoc, value);
	}

	void Effect::SetVec3(const char* var, const float x, const float y, const float z)
	{
		assert(m_iProgram >= 0);
		GLuint valueLoc = glGetUniformLocation(m_iProgram, var);
		glUniform3f(valueLoc, x, y, z);
	}

	void Effect::SetVec3(const char* var, const NPMathHelper::Vec3 &value)
	{
		assert(m_iProgram >= 0);
		GLuint valueLoc = glGetUniformLocation(m_iProgram, var);
		glUniform3f(valueLoc, value._x, value._y, value._z);
	}

	Window::Window(const char* name, const int sizeW, const int sizeH)
		: m_sName(name)
		, m_iSizeW(sizeW)
		, m_iSizeH(sizeH)
		, m_bIsInit(false)
		, m_pWindow(NULL)
		, m_pGLEWContext(NULL)
		, m_uiID(0)
		, m_pOwnerApp(NULL)
	{

	}

	Window::~Window()
	{
		if (m_pShareContent && m_pShareContent->DeRef() <= 0)
		{
			delete m_pShareContent;
			m_pShareContent = 0;
		}
	}

	void Window::AddInputMSG(INPUTMSG msg)
	{
		m_queueInputMSG.push(msg);
	}

	void Window::ProcessInputMSGQueue()
	{
		while (!m_queueInputMSG.empty())
		{
			INPUTMSG msg = m_queueInputMSG.front();
			m_queueInputMSG.pop();
			OnHandleInputMSG(msg);
		}
	}

	App::App(const int sizeW, const int sizeH)
		: m_iSizeW(sizeW)
		, m_iSizeH(sizeH)
		, m_pWindow(nullptr)
		, m_bIsInit(false)
		, m_fDeltaTime(0.f)
		, m_uiCurrentWindowID(0)
		, m_uiCurrentMaxID(0)
		, m_bForceShutdown(false)
	{
		g_pMainApp = this;
	}

	App::~App()
	{
		g_pMainApp = NULL;
	}

	int App::Run(Window* initWindow)
	{
		if (GLInit() < 0)
			return -1;
		AttachWindow(initWindow);

		while (WindowsUpdate())
		{
			glfwPollEvents();
			float currentTime = glfwGetTime();

			if (m_fLastTime > 0.f)
				m_fDeltaTime = currentTime - m_fLastTime;

			for (auto it = m_mapWindows.begin(); it != m_mapWindows.end(); it++)
			{
				SetCurrentWindow(it->first);
				it->second->ProcessInputMSGQueue();
				it->second->OnTick(GetDeltaTime());
			}

			m_fLastTime = currentTime;
		}
		glfwTerminate();
		return 0;
	}

	void App::Shutdown()
	{
		m_bForceShutdown = true;
	}

	void App::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
	{
		for (auto& mapWin : m_mapWindows)
		{
			if (mapWin.second->GetGLFWWindow() == window)
			{
				Window::INPUTMSG msg;
				msg.type = Window::INPUTMSG_KEYBOARDKEY;
				msg.key = key;
				msg.scancode = scancode;
				msg.action = action;
				msg.mode = mode;
				msg.timestamp = glfwGetTime();
				mapWin.second->AddInputMSG(msg);
				break;
			}
		}
	}

	void App::MouseKeyCallback(GLFWwindow* window, int key, int action, int mode)
	{
		for (auto& mapWin : m_mapWindows)
		{
			if (mapWin.second->GetGLFWWindow() == window)
			{
				Window::INPUTMSG msg;
				msg.type = Window::INPUTMSG_MOUSEKEY;
				msg.key = key;
				msg.action = action;
				msg.mode = mode;
				msg.timestamp = glfwGetTime();
				mapWin.second->AddInputMSG(msg);
				break;
			}
		}
	}

	void App::MouseCursorCallback(GLFWwindow* window, double xpos, double ypos)
	{
		for (auto& mapWin : m_mapWindows)
		{
			if (mapWin.second->GetGLFWWindow() == window)
			{
				Window::INPUTMSG msg;
				msg.type = Window::INPUTMSG_MOUSECURSOR;
				msg.xpos = xpos;
				msg.ypos = ypos;
				msg.timestamp = glfwGetTime();
				mapWin.second->AddInputMSG(msg);
				break;
			}
		}
	}

	void App::MouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		for (auto& mapWin : m_mapWindows)
		{
			if (mapWin.second->GetGLFWWindow() == window)
			{
				Window::INPUTMSG msg;
				msg.type = Window::INPUTMSG_MOUSESCROLL;
				msg.xoffset = xoffset;
				msg.yoffset = yoffset;
				msg.timestamp = glfwGetTime();
				mapWin.second->AddInputMSG(msg);
				break;
			}
		}
	}

	App* App::g_pMainApp = nullptr;
	void App::GlobalKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
	{
		if (g_pMainApp)
			g_pMainApp->KeyCallback(window, key, scancode, action, mode);
	}

	void App::GlobalMouseKeyCallback(GLFWwindow *window, int key, int action, int mode)
	{
		if (g_pMainApp)
			g_pMainApp->MouseKeyCallback(window, key, action, mode);
	}

	void App::GlobalMouseCursorCallback(GLFWwindow* window, double xpos, double ypos)
	{
		if (g_pMainApp)
			g_pMainApp->MouseCursorCallback(window, xpos, ypos);
	}

	void App::GlobalMouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		if (g_pMainApp)
			g_pMainApp->MouseScrollCallback(window, xoffset, yoffset);
		//std::cout << "Scrolled " << xoffset << ", " << yoffset << std::endl;
	}

	int App::GLInit()
	{
		if (m_bIsInit || m_pWindow)
			return 0;

		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

		m_bIsInit = true;
		return 0;
	}

	bool App::WindowsUpdate()
	{
		TerminateShouldQuitWindows();

		if (m_bForceShutdown && m_mapWindows.size() > 0)
		{
			TerminateShouldQuitWindows();
		}

		return (m_mapWindows.size() > 0);
	}

	void App::TerminateShouldQuitWindows()
	{
		std::vector<unsigned int> removeList;
		for (auto it = m_mapWindows.begin(); it != m_mapWindows.end(); it++)
		{
			if (glfwWindowShouldClose(it->second->GetGLFWWindow()) || m_bForceShutdown)
			{
				removeList.push_back(it->first);
			}
		}

		for (auto it = removeList.begin(); it != removeList.end(); it++)
		{
			Window* closeWin = m_mapWindows[*it];
			if (closeWin->ShouldTerminateProgramOnTerminate())
				m_bForceShutdown = true;
			SetCurrentWindow(*it);
			closeWin->OnTerminate();
			GLEWContext* glewCont = closeWin->GetGLEWContext();
			GLFWwindow* glfwWin = closeWin->GetGLFWWindow();
			if (closeWin)
			{
				delete closeWin;
				closeWin = NULL;
			}
			if (glewCont)
			{
				delete glewCont;
				glewCont = NULL;
			}
			if (glfwWin)
			{
				glfwDestroyWindow(glfwWin);
				glfwWin = NULL;
			}
			m_mapWindows.erase(*it);
		}
	}

	unsigned int App::AttachWindow(Window* window, Window* sharedGLWindow)
	{
		if (!window)
			return 0;

		unsigned int prevWinId = m_uiCurrentWindowID;
		window->m_pWindow = glfwCreateWindow(window->m_iSizeW, window->m_iSizeH, window->m_sName.c_str(), nullptr
			, (sharedGLWindow) ? sharedGLWindow->GetGLFWWindow() : nullptr);
		if (sharedGLWindow)
		{
			window->ShareContentWithOther(sharedGLWindow);
			window->GetShareContent()->AddRef();
		}
		else
		{
			window->m_pShareContent = new ShareContent();
		}
		window->m_uiID = ++m_uiCurrentMaxID;
		m_mapWindows[window->m_uiID] = window;

		if (!window->m_pWindow)
		{
			std::cout << "Failed to create GLFW for window " << window->m_sName << std::endl;
			m_mapWindows.erase(window->m_uiID);
			m_uiCurrentMaxID--;
			return 0;
		}
		window->m_pGLEWContext = new GLEWContext();
		if (!window->m_pGLEWContext)
		{
			std::cout << "Failed to create GLEW Context for window " << window->m_sName << std::endl;
			m_mapWindows.erase(window->m_uiID);
			m_uiCurrentMaxID--;
			return 0;
		}
		SetCurrentWindow(window->m_uiID);

		glfwSetKeyCallback(window->m_pWindow, GlobalKeyCallback);
		glfwSetMouseButtonCallback(window->m_pWindow, GlobalMouseKeyCallback);
		glfwSetCursorPosCallback(window->m_pWindow, GlobalMouseCursorCallback);
		glfwSetScrollCallback(window->m_pWindow, GlobalMouseScrollCallback);

		glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK)
		{
			std::cout << "Failed to initialize GLEW for window " << window->m_sName << std::endl;
			if (prevWinId > 0)
				SetCurrentWindow(prevWinId);
			m_mapWindows.erase(window->m_uiID);
			m_uiCurrentMaxID--;
			return 0;
		}
		CHECK_GL_ERROR;
		glViewport(0, 0, window->m_iSizeW, window->m_iSizeH);
		window->SetOwner(this);
		window->OnInit();

		if (prevWinId > 0)
			SetCurrentWindow(prevWinId);
		return window->m_uiID;
	}

	bool App::SetCurrentWindow(const unsigned int id)
	{
		m_uiCurrentWindowID = id;
		Window* curWin = GetCurrentWindow();
		if (curWin)
			glfwMakeContextCurrent(GetCurrentWindow()->m_pWindow);
		return curWin != nullptr;
	}

	Window* App::GetCurrentWindow()
	{
		if (!m_uiCurrentWindowID)
			return nullptr;
		return m_mapWindows[m_uiCurrentWindowID];
	}

	DebugLine::DebugLine()
		: m_v3Start()
		, m_v3End()
		, m_v3Color(1.f,0.f,0.f)
		, m_iVAO(-1)
		, m_iVBO(-1)
		, m_pEffect(nullptr)
	{

	}

	DebugLine::~DebugLine()
	{

	}

	void DebugLine::Init(ShareContent* content)
	{
		CHECK_GL_ERROR;
		m_pEffect = content->GetEffect("DebugLineEffect");
		if (!m_pEffect->GetIsLinked())
		{
			m_pEffect->initEffect();
			m_pEffect->attachShaderFromFile("../shader/debugLineVS.glsl", GL_VERTEX_SHADER);
			m_pEffect->attachShaderFromFile("../shader/debugLinePS.glsl", GL_FRAGMENT_SHADER);
			m_pEffect->linkEffect();
		}
		glGenBuffers(1, &m_iVBO);
		UpdateBuffer();
		glGenVertexArrays(1, &m_iVAO);
		glBindVertexArray(m_iVAO);
		glBindBuffer(GL_ARRAY_BUFFER, m_iVBO);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glBindVertexArray(0);
		CHECK_GL_ERROR;
	}

	void DebugLine::Draw(const NPMathHelper::Vec3& start, const NPMathHelper::Vec3& end, const NPMathHelper::Vec3& color
		, const float* viewMat, const float* projMat)
	{
		CHECK_GL_ERROR;
		if (start != m_v3Start || end != m_v3End || color != m_v3Color)
		{
			m_v3Start = start;
			m_v3End = end;
			m_v3Color = color;
			UpdateBuffer();
		}

		if (m_v3Start == m_v3End)
			return;

		m_pEffect->activeEffect();
		m_pEffect->SetMatrix("projection", projMat);
		m_pEffect->SetMatrix("view", viewMat);

		glBindVertexArray(m_iVAO);
		glDrawArrays(GL_LINE_STRIP, 0, 2);
		glBindVertexArray(0);

		m_pEffect->deactiveEffect();
		CHECK_GL_ERROR;
	}

	void DebugLine::UpdateBuffer()
	{
		CHECK_GL_ERROR;
		std::vector<GLfloat> vertices;
		vertices.push_back(m_v3Start._x);
		vertices.push_back(m_v3Start._y);
		vertices.push_back(m_v3Start._z);
		vertices.push_back(m_v3Color._x);
		vertices.push_back(m_v3Color._y);
		vertices.push_back(m_v3Color._z);
		vertices.push_back(m_v3End._x);
		vertices.push_back(m_v3End._y);
		vertices.push_back(m_v3End._z);
		vertices.push_back(m_v3Color._x);
		vertices.push_back(m_v3Color._y);
		vertices.push_back(m_v3Color._z);

		glBindBuffer(GL_ARRAY_BUFFER, m_iVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); 
		CHECK_GL_ERROR;
	}
}