#ifndef SPECTRALHELPER_H
#define SPECTRALHELPER_H

#include "mathhelper.h"
#include "cudahelper.h"

namespace NPCudaSpecHelper
{
	static const uint32 c_u32LamdaStart = 400;
	static const uint32 c_u32LamdaEnd = 700;
	static const uint32 c_u32SampleN = 30;

	class Spectrum
	{
	public:
		__hd__ Spectrum();
		__hd__ Spectrum(float* samples);
		__hd__ ~Spectrum(){}

		__hd__ void ClearData() { for (uint32 i = 0; i < c_u32SampleN; i++) m_fSamples[i] = 0.f; }
		__hd__ inline void SetData(const uint32 slot, const float data){ m_fSamples[slot] = data; }
		__hd__ inline float GetData(const uint32 slot) const { return m_fSamples[slot]; }

		__hd__ void GetXYZ(float& x, float& y, float& z
			, const Spectrum* baseSpec, const float baseSpecIntY) const;
		__hd__ void GetRGB(float& r, float& g, float& b
			, const Spectrum* baseSpec, const float baseSpecIntY) const;
	private:
#pragma pack(push, 1)
		float m_fSamples[c_u32SampleN];
#pragma pack(pop)
	};

	static Spectrum g_baseSpec[3];
	static Spectrum* g_pDevBaseSpec;
	static float g_fBaseSpecIntY = 0.f;
	__host__ void InitBaseSpectrum();
	__host__ void ClearBaseSpectrum();
	__host__ bool IsBaseSpectrumValid();
}

#endif