#ifndef GLHELPER_H
#define GLHELPER_H

#define GLEW_STATIC
#define GLEW_MX
#include <GL\glew.h>

#include <GLFW\glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include <string>
#include <vector>
#include <queue>
#include <map>

#include "geohelper.h"
#include "mathhelper.h"

#ifdef _DEBUG
static void CheckGLCoreError(const char *file, int line, const char *func)
{
	int err = 0;
	char msg[256];
	while ((err = glGetError()) != 0)
	{
		sprintf_s(msg, "%s(%d) : [%s] GL_CORE_ERROR=0x%x\n", file, line, func, err);
#ifdef ANT_WINDOWS
		OutputDebugString(msg);
#endif
		fprintf(stderr, msg);
	}
}
#   ifdef __FUNCTION__
#       define CHECK_GL_ERROR CheckGLCoreError(__FILE__, __LINE__, __FUNCTION__)
#   else
#       define CHECK_GL_ERROR CheckGLCoreError(__FILE__, __LINE__, "")
#   endif
#else
#   define CHECK_GL_ERROR ((void)(0))
#endif

GLEWContext* glewGetContext();

namespace NPGLHelper
{
	bool loadASCIIFromFile(std::string file, std::string &content);
	bool createShaderFromFile(std::string file, GLuint type, GLuint &result);

	bool checkShaderError(GLuint shader, GLuint checking, std::string &info);
	bool checkProgramError(GLuint program, GLuint checking, std::string &info);

	bool loadHDRTextureFromFile(const char* path, GLuint &id, GLint warpS, GLint warpT, GLint minFil, GLint maxFil);
	bool loadTextureFromFile(const char* path, GLuint &id, GLint warpS, GLint warpT, GLint minFil, GLint maxFil, bool sRGB = true);
	bool loadCubemapFromFiles(std::string faces[6], GLuint &id, bool sRGB = true);

	bool saveScreenShotBMP(const char* filename, const float width, const float height);

	class RenderObject
	{
	public:
		RenderObject();
		~RenderObject();

		void SetGeometry(const NPGeoHelper::Geometry& geo, unsigned int type = 0);
		void ClearGeometry();
		GLuint GetVAO() { return m_iVAO; }
		GLsizei GetIndicesSize() { return m_iIndicesSize; }

	protected:
		GLuint m_iVAO;
		GLuint m_iVBO;
		GLuint m_iEBO;
		GLsizei m_iIndicesSize;
	};

	class Effect
	{
	public:
		Effect();
		~Effect();
		void initEffect();
		void attachShaderFromFile(const char* filename, const GLuint type);
		void deleteAttachedShaders();
		bool linkEffect();
		bool activeEffect();
		bool deactiveEffect();
		
		void SetMatrix(const char* var, const float* mat);
		void SetInt(const char* var, const int value);
		void SetFloat(const char* var, const float value);
		void SetVec3(const char* var, const float x, const float y, const float z);
		void SetVec3(const char* var, const NPMathHelper::Vec3 &value);

		inline const bool GetIsLinked() { return m_bIsLinked; }

	protected:
		GLuint m_iProgram;
		bool m_bIsLinked;
		std::vector<GLuint> m_vAttachedShader;
	};

	class ShareContent
	{
	public:
		ShareContent()
			:m_uiRefCount(1)
		{}
		virtual ~ShareContent()
		{
			for (auto& effect : m_mapEffects)
			{
				delete effect.second;
			}
			m_mapEffects.clear();
		}

		inline Effect* GetEffect(const char* name)
		{ 
			Effect* storedEffect = m_mapEffects[name];
			if (!storedEffect)
				m_mapEffects[name] = new Effect();
			return m_mapEffects[name];
		}

		inline void RemoveEffect(const char* name)
		{
			Effect* storedEffect = m_mapEffects[name];
			if (storedEffect)
			{
				delete storedEffect;
				storedEffect = nullptr;
				m_mapEffects.erase(name);
			}
		}

		inline unsigned int DeRef()
		{
			return --m_uiRefCount;
		}

		inline int AddRef()
		{
			return ++m_uiRefCount;
		}

	protected:
		std::map<std::string, Effect*> m_mapEffects;
		unsigned int m_uiRefCount;
	};

	class App;
	class Window
	{
	public:
		enum INPUTMSG_TYPE
		{
			INPUTMSG_KEYBOARDKEY,
			INPUTMSG_MOUSEKEY,
			INPUTMSG_MOUSECURSOR,
			INPUTMSG_MOUSESCROLL
		};
		enum INPUTMSG_ACTION
		{
			PRESS,
			RELEASE
		};
		struct INPUTMSG
		{
			INPUTMSG_TYPE type;
			float timestamp;
			union
			{
				struct
				{
					int key;
					int scancode;
					int action;
					int mode;
				};
				struct
				{
					double xpos;
					double ypos;
				};
				struct
				{
					double xoffset;
					double yoffset;
				};
			};
		};

		friend class App;
		Window(const char* name, const int sizeW = 800, const int sizeH = 600);
		virtual ~Window();

		virtual int OnInit() = 0;
		virtual int OnTick(const float deltaTime) = 0;
		virtual void OnTerminate() = 0;
		virtual bool ShouldTerminateProgramOnTerminate() { return false; }
		virtual void OnHandleInputMSG(const INPUTMSG &msg) = 0;

		void AddInputMSG(INPUTMSG msg);
		void ProcessInputMSGQueue();


		inline GLEWContext* GetGLEWContext() { return m_pGLEWContext; }
		inline GLFWwindow* GetGLFWWindow() { return m_pWindow; }
		inline ShareContent* GetShareContent() { return m_pShareContent; }
		inline void ShareContentWithOther(Window* other)
		{
			if (m_pShareContent)
				delete m_pShareContent;
			m_pShareContent = other->m_pShareContent;
		}

		inline void SetOwner(App* owner) { m_pOwnerApp = owner; }
		inline App* GetOwner() { return m_pOwnerApp; }

	protected:
		bool m_bIsInit;
		std::string m_sName;
		int m_iSizeW, m_iSizeH;
		GLFWwindow* m_pWindow;
		GLEWContext* m_pGLEWContext;
		unsigned int m_uiID;

		ShareContent* m_pShareContent;
		std::queue<INPUTMSG> m_queueInputMSG;

		App* m_pOwnerApp;
	};

	class App
	{
	public:
		App(const int sizeW = 800, const int sizeH = 600);
		virtual ~App();

		int Run(Window* initWindow);
		void Shutdown();
		virtual void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
		virtual void MouseKeyCallback(GLFWwindow* window, int key, int action, int mode);
		virtual void MouseCursorCallback(GLFWwindow* window, double xpos, double ypos);
		virtual void MouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

		static App* g_pMainApp;
		static void GlobalKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
		static void GlobalMouseKeyCallback(GLFWwindow *window, int key, int action, int mode);
		static void GlobalMouseCursorCallback(GLFWwindow* window, double xpos, double ypos);
		static void GlobalMouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

		inline const float GetDeltaTime() { return m_fDeltaTime; }

		unsigned int AttachWindow(Window* window, Window* sharedGLWindow = nullptr);
		bool SetCurrentWindow(const unsigned int id);
		Window* GetCurrentWindow();
		inline Window* GetWindow(const unsigned int id) { return (GetIsWindowActive(id)) ? m_mapWindows[id] : nullptr; }
		inline bool GetIsWindowActive(const unsigned int id) { return (m_mapWindows.find(id) != m_mapWindows.end()); }

	protected:
		int GLInit();
		bool WindowsUpdate();
		void TerminateShouldQuitWindows();

		bool m_bIsInit;
		int m_iSizeW, m_iSizeH;
		GLFWwindow* m_pWindow;

		std::map<unsigned int, Window*> m_mapWindows;
		unsigned int m_uiCurrentWindowID;
		unsigned int m_uiCurrentMaxID;
		bool m_bForceShutdown;

	private:
		float m_fDeltaTime;
		float m_fLastTime;
	};


	class DebugLine
	{
	public:
		DebugLine();
		~DebugLine();
		void Init(ShareContent* content);
		void Draw(const NPMathHelper::Vec3& start, const NPMathHelper::Vec3& end, const NPMathHelper::Vec3& color
			, const float* viewMat, const float* projMat);
	protected:
		void UpdateBuffer();

		NPMathHelper::Vec3 m_v3Start;
		NPMathHelper::Vec3 m_v3End;
		NPMathHelper::Vec3 m_v3Color;
		Effect* m_pEffect;

		GLuint m_iVAO;
		GLuint m_iVBO;
	};
}

#endif