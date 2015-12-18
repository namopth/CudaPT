#ifndef ATBHELPER_H
#define ATBHELPER_H

#include "macrohelper.h"

#include <AntTweakBar.h>
#define ATB_ASSERT(FUNC) \
	if(!FUNC) \
		{ \
	DEBUG_COUT(TwGetLastError()); \
		}

#endif