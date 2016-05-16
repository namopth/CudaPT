#ifndef CONFFILEHELPER_H
#define CONFFILEHELPER_H

#include <string>
#include <sstream>

#include "mathhelper.h"

namespace NPConfFileHelper
{
	class txtConfFile
	{
	public:
		txtConfFile(const std::string& path = "");
		~txtConfFile();

		bool ReadNextVar(std::string& name);
		void WriteVar(const std::string& name);

		template<typename T> void Read(T& value)
		{
			m_sUnreadDataStream >> value;
			m_sReadDataStream << value;
		}

		void Read(NPMathHelper::Vec3& value)
		{
			m_sUnreadDataStream >> value._x >> value._y >> value._z;
			m_sReadDataStream << value._x << value._y << value._z;
		}

		template<typename T> void Write(const T& data)
		{
			m_sUnwrittenDataStream << data << "\t";
		}

		template<typename T> void WriteRaw(const T& data)
		{
			m_sUnwrittenDataStream << data;
		}

		void Write(const NPMathHelper::Vec3& data)
		{
			m_sUnwrittenDataStream << data._x << "\t" << data._y << "\t" << data._z << "\t";
		}

		void ClearData();
		inline const bool isValid() const { return m_bIsValid; }

		bool SyncDataToFile();
	private:

		std::stringstream m_sUnwrittenDataStream;
		std::stringstream m_sUnreadDataStream;
		std::stringstream m_sReadDataStream;
		bool m_bIsValid;
		std::string m_sPath;
	};
}
#endif