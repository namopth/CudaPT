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

	std::string GetRelPathFromFull(const std::string& mainPath, const std::string& convPath)
	{
		//std::cout << "GetRelPathFromFull\t" << mainPath << "\t" << convPath << "\n";
		std::string relPath;

		// remove similar path
		size_t folEnd = 0;
		size_t matchedEnd = 0;
		while (folEnd < mainPath.size())
		{
			folEnd = mainPath.find("\\", matchedEnd);
			folEnd = (folEnd == std::string::npos) ? mainPath.size() : folEnd - 1;
			if (!mainPath.substr(matchedEnd, folEnd - matchedEnd + 1).compare(convPath.substr(matchedEnd, folEnd - matchedEnd + 1)))
			{
				matchedEnd = folEnd + 2;
			}
			else
			{
				break;
			}
			//std::cout << "Matched Part Processing:\t" << mainPath.substr(0, matchedEnd-1) << "\n";
		}
		//std::cout << "Matched Part\t" << mainPath.substr(0,matchedEnd-1) << "\n";

		if (!matchedEnd)
		{
			return convPath;
		}

		size_t folBeg = matchedEnd;
		while (folBeg < mainPath.size())
		{
			folBeg = mainPath.find("\\", folBeg);
			folBeg = (folBeg == std::string::npos) ? mainPath.size() : folBeg + 1;
			relPath += "..\\";
		}
		relPath += convPath.substr(matchedEnd);
		//std::cout << "RelPath\t" << relPath << "\n";

		return relPath;
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

	float GetOSTimeInSec()
	{
		SYSTEMTIME time;
		GetSystemTime(&time);
		float time_sec = time.wSecond + (float)time.wMilliseconds*0.001f;
		return time_sec;
	}

	long GetOSTimeInMSec()
	{
		SYSTEMTIME time;
		GetSystemTime(&time);
		long time_ms = (time.wSecond * 1000) + time.wMilliseconds;
		return time_ms;
	}

#endif
}