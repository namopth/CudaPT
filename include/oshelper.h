#ifndef OSHELPER_H
#define OSHELPER_H

#ifdef _WIN32
#ifdef _WIN64
#define WINOS
#endif
#endif

#include <string>

namespace NPOSHelper
{
	std::string BrowseFile(const char* filter);

	enum MSGBOX_TYPE
	{
		MSGBOX_OK				= 0x00000000L,
		MSGBOX_OKCANCEL			= 0x00000001L,
		MSGBOX_ABORTRETRYIGNORE = 0x00000002L,
		MSGBOX_YESNOCANCEL		= 0x00000003L,
		MSGBOX_YESNO			= 0x00000004L,
		MSGBOX_RETRYCANCEL		= 0x00000005L
	};

	//NPOSHelper::CreateMessageBox(NPOSHelper::BrowseFile("All\0*.*\0Text\0*.TXT\0").c_str(), "", NPOSHelper::MSGBOX_OK);
	int CreateMessageBox(const char* text, const char* title, const unsigned int type);

	std::string GetOSCurrentDirectory();
	void SetOSCurrentDirectory(std::string &dir);
}

#endif