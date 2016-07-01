#include "cudaspechelper.h"
#include "oshelper.h"

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

namespace NPCudaSpecHelper
{
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
			sum += () * (segEnd - segStart);
		}

		return sum;
	}

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
	}

}