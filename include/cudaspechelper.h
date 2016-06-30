#ifndef SPECTRALHELPER_H
#define SPECTRALHELPER_H

#include "mathhelper.h"
#include "cudahelper.h"

namespace NPCudaSpecHelper
{
	static const uint32 c_u32LamdaStart = 400;
	static const uint32 c_u32LamdaEnd = 700;
	static const uint32 c_u32SampleN = 10;

	class Spectrum
	{
	public:
		__hd__ Spectrum();
		__hd__ ~Spectrum(){}

		__hd__ void ClearData() { for (uint32 i = 0; i < c_u32SampleN; i++) m_fSamples[i] = 0.f; }
		__hd__ inline void SetData(const uint32 slot, const float data){ m_fSamples[slot] = data; }
		__hd__ inline float GetData(const uint32 slot) { return m_fSamples[slot]; }

	private:
#pragma pack(push, 1)
		float m_fSamples[c_u32SampleN];
#pragma pack(pop)
	};

	static Spectrum* g_pDevBaseSpec[3] = { nullptr, nullptr, nullptr };
	static float g_fBaseSpecIntY = 0.f;
	__host__ void InitBaseSpectrum();
	__host__ bool IsBaseSpectrumValid();
}

#endif