#ifndef SPECTRALHELPER_H
#define SPECTRALHELPER_H

#include <vector>

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
		__hd__ inline const float* GetData() const { return m_fSamples; }
		__hd__ inline float* GetData() { return m_fSamples; }

		__hd__ void GetXYZ(float& x, float& y, float& z
			, const Spectrum* baseSpec, const float baseSpecIntY) const;
		__hd__ void GetRGB(float& r, float& g, float& b
			, const Spectrum* baseSpec, const float baseSpecIntY) const;
	private:
#pragma pack(push, 1)
		float m_fSamples[c_u32SampleN];
#pragma pack(pop)
	};

	extern Spectrum g_baseSpec[3];
	extern Spectrum* g_pDevBaseSpec;
	extern float g_fBaseSpecIntY;
	__host__ float AverageSpectrum(const std::vector<float>& specWavelength, const std::vector<float>& specPower
		,const float lambdaStart, const float lambdaEnd);
	__host__ void InitBaseSpectrum();
	__host__ void ClearBaseSpectrum();
	__host__ bool IsBaseSpectrumValid();

	__hd__ void XYZToRGB(const float x, const float y, const float z
		, float& r, float& g, float& b);
	__hd__ void RGBToXYZ(const float r, const float g, const float b
		, float& x, float& y, float& z);
	__hd__ void SPDToXYZ(const float spd[c_u32SampleN], float& x, float& y, float& z
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ
		, const float baseSpecIntY);
	__hd__ void SPDToRGB(const float spd[c_u32SampleN], float& r, float& g, float& b
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ
		, const float baseSpecIntY);
	__hd__ float XYZToSPDAtInd(const float x, const float y, const float z, const uint32 spdInd
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ
		, const float baseSpecIntY);
	__hd__ float RGBToSPDAtInd(const float r, const float g, const float b, const uint32 spdInd
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ
		, const float baseSpecIntY);
}

#endif