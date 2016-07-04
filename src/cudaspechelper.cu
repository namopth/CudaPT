#include "cudaspechelper.h"
#include "oshelper.h"

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

namespace NPCudaSpecHelper
{
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
		CUFREE(g_pDevBaseSpec[0]);
		CUFREE(g_pDevBaseSpec[1]);
		CUFREE(g_pDevBaseSpec[2]);

		HANDLE_ERROR(cudaMalloc((void**)&g_pDevBaseSpec[0], sizeof(Spectrum)));
		HANDLE_ERROR(cudaMalloc((void**)&g_pDevBaseSpec[1], sizeof(Spectrum)));
		HANDLE_ERROR(cudaMalloc((void**)&g_pDevBaseSpec[2], sizeof(Spectrum)));

		HANDLE_ERROR(cudaMemcpy(g_pDevBaseSpec[0], &g_baseSpec[0]
			, sizeof(Spectrum), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(g_pDevBaseSpec[1], &g_baseSpec[1]
			, sizeof(Spectrum), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(g_pDevBaseSpec[2], &g_baseSpec[2]
			, sizeof(Spectrum), cudaMemcpyHostToDevice));
	}

	__host__ void ClearBaseSpectrum()
	{
		CUFREE(g_pDevBaseSpec[0]);
		CUFREE(g_pDevBaseSpec[1]);
		CUFREE(g_pDevBaseSpec[2]);
	}

	__host__ bool IsBaseSpectrumValid()
	{
		return g_fBaseSpecIntY > 0 && g_pDevBaseSpec[0] && g_pDevBaseSpec[1] && g_pDevBaseSpec[2];
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
		, const Spectrum* baseSpec[3], const float baseSpecIntY) const
	{
		x = y = z = 0.f;
		for (uint32 i = 0; i < c_u32SampleN; i++)
		{
			x += baseSpec[0]->GetData(i) * GetData(i);
			y += baseSpec[1]->GetData(i) * GetData(i);
			z += baseSpec[2]->GetData(i) * GetData(i);
		}
		x /= baseSpecIntY;
		y /= baseSpecIntY;
		z /= baseSpecIntY;
	}

	__hd__ void Spectrum::GetRGB(float& r, float& g, float& b
		, const Spectrum* baseSpec[3], const float baseSpecIntY) const
	{
		//HD Monitor
		float xyz[3];
		GetXYZ(xyz[0], xyz[1], xyz[2], baseSpec, baseSpecIntY);
		r = 3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
		g = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
		b = 0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];
	}

}