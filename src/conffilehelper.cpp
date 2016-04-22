#include "conffilehelper.h"

#include "oshelper.h"
#include "macrohelper.h"

namespace NPConfFileHelper
{
	txtConfFile::txtConfFile(const std::string& path)
		: m_bIsValid(false)
		, m_sPath(path)
	{
		NPOSHelper::ifstr fileStream(m_sPath);
		if (fileStream.is_open())
		{
			m_bIsValid = true;
			fileStream.seekg(0, std::ios::end);
			size_t size = fileStream.tellg();
			fileStream.seekg(0, std::ios::beg);
			char* rawData = new char[size];
			fileStream.read(rawData, size);
			m_sUnreadDataStream << rawData;
			DELETE(rawData);
		}

	}

	txtConfFile::~txtConfFile()
	{
		SyncDataToFile();
	}

	bool txtConfFile::ReadNextVar(std::string& name)
	{
		std::string findName;
		m_sUnreadDataStream >> findName;
		bool isInComment = false;
		while (!m_sUnreadDataStream.eof())
		{
			std::size_t commentT = findName.find("//");
			if (commentT != std::string::npos)
			{
				findName = findName.substr(0, commentT);
				std::string line;
				std::getline(m_sUnreadDataStream, line);
			}

			commentT = findName.find("/*");
			if (commentT != std::string::npos)
			{
				std::size_t commentT2 = findName.find("*/");
				if (commentT2 != std::string::npos && commentT2 > commentT)
				{
					findName = findName.substr(0, commentT) + findName.substr(commentT2, findName.size());
				}
				else if (commentT2 != std::string::npos)
				{
					findName = findName.substr(0, commentT);
				}
				else
				{
					isInComment = true;
					findName = findName.substr(0, commentT);
				}
			}

			if (isInComment)
			{
				commentT = findName.find("*/");
				if (commentT != std::string::npos)
				{
					isInComment = false;
					findName = findName.substr(commentT, findName.size());
				}
			}

			if (!isInComment && findName.size() > 0)
			{
				name = findName;
				return true;
			}
			m_sUnreadDataStream >> findName;
		}
		return false;
	}

	void txtConfFile::WriteVar(const std::string& name)
	{
		if (m_sUnwrittenDataStream.str().size() != 0)
			m_sUnwrittenDataStream << "\n";
		m_sUnwrittenDataStream << name << "\t";
	}

	template<typename T> T txtConfFile::Read()
	{
		T value;
		m_sUnreadDataStream >> value;
		m_sReadDataStream << value;
		return value;
	}

	template<typename T> void txtConfFile::Write(const T& data)
	{
		m_sUnwrittenDataStream << data << "\t";
	}

	void txtConfFile::ClearData()
	{
		m_sUnwrittenDataStream.clear();
		m_sUnreadDataStream.clear();
		m_sReadDataStream.clear();
	}

	bool txtConfFile::SyncDataToFile()
	{
		if (m_sUnwrittenDataStream.str().size() > 0)
		{
			NPOSHelper::ofstr fileStream;
			fileStream.open(m_sPath, std::ios::out, std::ios::trunc);
			if (fileStream.fail())
			{
				return false;
			}
			fileStream << m_sUnwrittenDataStream.str();
			fileStream.close();
			m_sUnwrittenDataStream.clear();
		}
	}
}