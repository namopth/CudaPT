#include "cudaspechelper.h"
#include "oshelper.h"

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

namespace NPCudaSpecHelper
{
	Spectrum g_baseSpec[3];
	Spectrum* g_pDevBaseSpec = nullptr;
	float g_fBaseSpecIntY = 0.f;

#define LERP(__INT__,__BGN__,__END__) __BGN__ + (__END__ - __BGN__) * __INT__
	__host__ float AverageSpectrum(const std::vector<float>& specWavelength, const std::vector<float>& specPower,
		const float lambdaStart, const float lambdaEnd)
	{
		if (specPower.size() <= 0) return 0.f;
		if (specPower.size() == 1) return specPower[0];
		if (lambdaEnd <= specWavelength[0]) return specPower[0];
		if (lambdaStart >= specWavelength[specPower.size() - 1]) return specPower[specPower.size() - 1];
		float sum = 0.f;
		if (lambdaStart < specWavelength[0]) 
			sum += specPower[0] * (specWavelength[0] - lambdaStart);
		if (lambdaEnd > specWavelength[specPower.size() - 1])
			sum += specPower[specPower.size() - 1] * (lambdaEnd - specWavelength[specPower.size() - 1]);
		uint32 i = 0;
		while (specWavelength[i + 1] < lambdaStart) i++;
		for (; i + 1 < specPower.size() && specWavelength[i] <= lambdaEnd; i++)
		{
			float segStart = fmaxf(specWavelength[i], lambdaStart);
			float segEnd = fminf(specWavelength[i + 1], lambdaEnd);
			sum += 0.5f * (LERP((segStart - specWavelength[i]) / (specWavelength[i + 1] - specWavelength[i])
				, specPower[i], specPower[i + 1]) 
				+ LERP((segEnd - specWavelength[i]) / (specWavelength[i + 1] - specWavelength[i])
				, specPower[i], specPower[i + 1])) * (segEnd - segStart);
		}

		return sum / (lambdaEnd - lambdaStart);
	}
#undef LERP

	__host__ void InitBaseSpectrum()
	{
		if (IsBaseSpectrumValid())
			return;

		std::vector<float> specPower[3];
		std::vector<float> specWavelength;
		std::string cieFilePath = NPOSHelper::GetOSCurrentDirectory() + "\\..\\asset\\constdata\\cie.csv";
		std::ifstream cieFile(cieFilePath);
		if (cieFile.is_open())
		{
			uint32 fileType = 0;
			while (cieFile.good())
			{
				std::string line;
				if (!std::getline(cieFile, line))
					break;
				std::stringstream sline(line);
				std::string token;
				while (std::getline(sline, token, ','))
				{
					std::stringstream tempStream(token);
					float powerData;
					float wavelengthData;
					switch (fileType)
					{
					case 0:
						tempStream >> wavelengthData;
						specWavelength.push_back(wavelengthData);
						break;
					case 1:
						tempStream >> powerData;
						specPower[0].push_back(powerData);
						break;
					case 2:
						tempStream >> powerData;
						specPower[1].push_back(powerData);
						break;
					case 3:
						tempStream >> powerData;
						specPower[2].push_back(powerData);
						break;
					}
					fileType = (fileType + 1) % 4;
				}
			}
			cieFile.close();
		}
		else
		{
			NPOSHelper::CreateMessageBox("FATAL ERROR! Cannot open CIE XYZ file.", "CIE XYZ Loading"
				, NPOSHelper::MSGBOX_OK);
		}

		g_fBaseSpecIntY = 0.f;
		for (uint32 i = 0; i < c_u32SampleN; i++)
		{
			float lambdaInterval = (float)(c_u32LamdaEnd - c_u32LamdaStart)/(float)c_u32SampleN;
			float xInRange = AverageSpectrum(specWavelength, specPower[0], c_u32LamdaStart + lambdaInterval * i
				, c_u32LamdaStart + lambdaInterval * (i + 1));
			float yInRange = AverageSpectrum(specWavelength, specPower[1], c_u32LamdaStart + lambdaInterval * i
				, c_u32LamdaStart + lambdaInterval * (i + 1));
			float zInRange = AverageSpectrum(specWavelength, specPower[2], c_u32LamdaStart + lambdaInterval * i
				, c_u32LamdaStart + lambdaInterval * (i + 1));
			g_baseSpec[0].SetData(i, xInRange);
			g_baseSpec[1].SetData(i, yInRange);
			g_baseSpec[2].SetData(i, zInRange);
			g_fBaseSpecIntY += yInRange;
		}
		CUFREE(g_pDevBaseSpec);

		HANDLE_ERROR(cudaMalloc((void**)&g_pDevBaseSpec, sizeof(Spectrum) * 3));

		HANDLE_ERROR(cudaMemcpy(g_pDevBaseSpec, g_baseSpec
			, sizeof(Spectrum) * 3, cudaMemcpyHostToDevice));
	}

	__host__ void ClearBaseSpectrum()
	{
		CUFREE(g_pDevBaseSpec);
	}

	__host__ bool IsBaseSpectrumValid()
	{
		return g_fBaseSpecIntY > 0 && g_pDevBaseSpec;
	}

	__hd__ void XYZToRGB(const float x, const float y, const float z
		, float& r, float& g, float& b)
	{
		r = 3.240479f*x - 1.537150f*y - 0.498535f*z;
		g = -0.969256f*x + 1.875991f*y + 0.041556f*z;
		b = 0.055648f*x - 0.204043f*y + 1.057311f*z;
	}

	__hd__ void RGBToXYZ(const float r, const float g, const float b
		, float& x, float& y, float& z)
	{
		x = 0.412453f*r + 0.357580f*g + 0.180423f*b;
		y = 0.212671f*r + 0.715160f*g + 0.072169f*b;
		z = 0.019334f*r + 0.119193f*g + 0.950227f*b;
	}

	__hd__ void SPDToXYZ(const float spd[c_u32SampleN], float& x, float& y, float& z
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ, const float baseSpecIntY)
	{
		x = y = z = 0.f;
		for (uint32 i = 0; i < c_u32SampleN; i++)
		{
			x += baseSpecX[i] * spd[i];
			y += baseSpecY[i] * spd[i];
			z += baseSpecZ[i] * spd[i];
		}
		x /= baseSpecIntY;
		y /= baseSpecIntY;
		z /= baseSpecIntY;
	}

	__hd__ void SPDToRGB(const float spd[c_u32SampleN], float& r, float& g, float& b
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ, const float baseSpecIntY)
	{
		float x, y, z;
		SPDToXYZ(spd, x, y, z, baseSpecX, baseSpecY, baseSpecZ, baseSpecIntY);
		XYZToRGB(x, y, z, r, g, b);
	}

	__hd__ float XYZToSPDAtInd(const float x, const float y, const float z, const uint32 spdInd
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ
		, const float baseSpecIntY)
	{
		return (x * baseSpecX[spdInd] + y * baseSpecY[spdInd] + z * baseSpecZ[spdInd]);
	}

	__hd__ float RGBToSPDAtInd(const float r, const float g, const float b, const uint32 spdInd
		, const float* baseSpecX, const float* baseSpecY, const float* baseSpecZ
		, const float baseSpecIntY)
	{
		float x, y, z;
		RGBToXYZ(r, g, b, x, y, z);
		return XYZToSPDAtInd(x, y, z, spdInd, baseSpecX, baseSpecY, baseSpecZ, baseSpecIntY);
	}

	__hd__ Spectrum::Spectrum()
	{
		ClearData();
	}

	__hd__ Spectrum::Spectrum(float* samples)
	{
		for (uint32 i = 0; i < c_u32SampleN; i++)
		{
			m_fSamples[i] = samples[i];
		}
	}

	__hd__ void Spectrum::GetXYZ(float& x, float& y, float& z
		, const Spectrum* baseSpec, const float baseSpecIntY) const
	{
		SPDToXYZ(m_fSamples, x, y, z, baseSpec[0].GetData(), baseSpec[1].GetData()
			, baseSpec[2].GetData(), baseSpecIntY);
	}

	__hd__ void Spectrum::GetRGB(float& r, float& g, float& b
		, const Spectrum* baseSpec, const float baseSpecIntY) const
	{
		SPDToRGB(m_fSamples, r, g, b, baseSpec[0].GetData(), baseSpec[1].GetData()
			, baseSpec[2].GetData(), baseSpecIntY);
	}

}