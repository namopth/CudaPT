#ifndef CONFFILEHELPER_H
#define CONFFILEHELPER_H

#include <string>
#include <sstream>

namespace NPConfFileHelper
{
	class txtConfFile
	{
	public:
		txtConfFile(const std::string& path = "");
		~txtConfFile();

		bool ReadNextVar(std::string& name);
		void WriteVar(const std::string& name);

		template<typename T> T Read();
		template<typename T> void Write(const T& data);

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