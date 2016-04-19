#include "oshelper.h"

#include <Windows.h>
#include <Commdlg.h>


namespace NPOSHelper
{
#ifdef WINOS
	std::string BrowseOpenFile(const char* filter)
	{
		OPENFILENAME ofn;
		char szFile[512];

		ZeroMemory(&ofn, sizeof(ofn));
		ofn.lStructSize = sizeof(ofn);
		ofn.hwndOwner = NULL;
		ofn.lpstrFile = szFile;
		ofn.lpstrFile[0] = '\0';
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrFilter = filter;
		ofn.nFilterIndex = 1;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.lpstrInitialDir = NULL;
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

		std::string curDir = NPOSHelper::GetOSCurrentDirectory();
		GetOpenFileName(&ofn);
		NPOSHelper::SetOSCurrentDirectory(curDir);

		return szFile;
	}

	std::string BrowseSaveFile(const char* filter, const char* ext)
	{
		OPENFILENAME ofn;
		char szFile[512];

		ZeroMemory(&ofn, sizeof(ofn));
		ofn.lStructSize = sizeof(ofn);
		ofn.hwndOwner = NULL;
		ofn.lpstrFile = szFile;
		ofn.lpstrFile[0] = '\0';
		ofn.nMaxFile = sizeof(szFile);
		ofn.lpstrDefExt = ext;
		ofn.lpstrFilter = filter;
		ofn.nFilterIndex = 1;
		ofn.lpstrFileTitle = NULL;
		ofn.nMaxFileTitle = 0;
		ofn.lpstrInitialDir = NULL;
		ofn.Flags = OFN_PATHMUSTEXIST;

		std::string curDir = NPOSHelper::GetOSCurrentDirectory();
		GetSaveFileName(&ofn);
		NPOSHelper::SetOSCurrentDirectory(curDir);

		return szFile;
	}

	int CreateMessageBox(const char* text, const char* title, const unsigned int type)
	{
		return MessageBox(NULL, text, title, type);
	}

	std::string GetOSCurrentDirectory()
	{
		char dir[1024];
		GetCurrentDirectory(1024, dir);
		std::string result = dir;
		return result;
	}

	void SetOSCurrentDirectory(std::string &dir)
	{
		SetCurrentDirectory(dir.c_str());
	}
#endif
}