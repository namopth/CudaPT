#include "cudaspechelper.h"

namespace NPCudaSpecHelper
{
	__host__ void InitBaseSpectrum()
	{
		if (IsBaseSpectrumValid())
			return;

	}

	__host__ bool IsBaseSpectrumValid()
	{
		return g_fBaseSpecIntY > 0 && g_pDevBaseSpec[0] && g_pDevBaseSpec[1] && g_pDevBaseSpec[2];
	}

	__hd__ Spectrum::Spectrum()
	{
	}

}