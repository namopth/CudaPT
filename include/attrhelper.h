#ifndef ATTRHELPER_H
#define ATTRHELPER_H

#include "macrohelper.h"

namespace NPAttrHelper
{
	enum ATTR_TYPE
	{
		ATTR_UINT,
		ATTR_INT,
		ATTR_BOOL,
		ATTR_FLOAT,
		ATTR_STR,
		ATTR_ENUM,
	};
	struct Attrib
	{
		union
		{
			unsigned __int32 attrUInt;
			__int32 attrInt;
			bool attrBool;
			float attrFloat;
		};

		ATTR_TYPE attrType;
		const char* attrName;
		unsigned __int32 attrLength;
		const char** attrEnumName;

		Attrib(const char* _name, unsigned __int32 _var) : attrName(_name), attrUInt(_var), attrType(ATTR_UINT), attrLength(1){}
		Attrib(const char* _name, __int32 _var) : attrName(_name), attrInt(_var), attrType(ATTR_INT), attrLength(1) {}
		Attrib(const char* _name, bool _var) : attrName(_name), attrBool(_var), attrType(ATTR_BOOL), attrLength(1) {}
		Attrib(const char* _name, float _var) : attrName(_name), attrFloat(_var), attrType(ATTR_FLOAT), attrLength(1) {}
		Attrib(const char* _name, const char** _enumName, unsigned __int32 _enumLength, unsigned __int32 _var) 
			: attrName(_name), attrUInt(_var), attrType(ATTR_ENUM), attrEnumName(_enumName), attrLength(_enumLength) {}

		inline const ATTR_TYPE GetType() const { return attrType; }
		inline unsigned __int32* GetUint() { return &attrUInt; }
		inline __int32* GetInt() { return &attrInt; }
		inline bool* GetBool() { return &attrBool; }
		inline float* GetFloat() { return &attrFloat; }

		virtual const int GetLength() const { return attrLength; }
		virtual const char* GetString() const { return nullptr; }
		virtual const char* GetEnumString(const unsigned __int32 n) { return nullptr; }

	};
}

#endif