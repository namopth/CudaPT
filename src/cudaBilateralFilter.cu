#include "cudaBilateralFilter.h"

namespace cudaBilateralFilter
{
	void updateGaussian(float* deviceGaussianConst, float delta, uint radius)
	{
		if (!deviceGaussianConst)
		{
			std::cout << "No allocation" << std::endl;
			return;
		}

		float fGaussian[GAUSSIANCOST_N];
		for (uint i = 0; i < 2 * radius + 1; i++)
		{
			int x = (int)i - (int)radius;
			fGaussian[i] = expf(-(x * x) / (2 * delta * delta));
			//std::cout << "(" << x << ", " << delta << ") ";
			//std::cout << fGaussian[i] << ", ";
		}
		//std::cout << std::endl;

		HANDLE_ERROR(cudaMemcpy(deviceGaussianConst, fGaussian, sizeof(float) * GAUSSIANCOST_N, cudaMemcpyHostToDevice));
	}

	__device__ float euclideanLength(float3 a, float3 b, float d)
	{
		float mod = (b.x - a.x) * (b.x - a.x) +
			(b.y - a.y) * (b.y - a.y) +
			(b.z - a.z) * (b.z - a.z);

		return expf(-mod / (2.f * d * d));
	}

	__global__ void bilaterial_kernel(float* input, uint width, uint height, float delta, uint radius, float* gaussianConst, float* output)
	{
		uint x = blockIdx.x*blockDim.x + threadIdx.x;
		uint y = blockIdx.y*blockDim.y + threadIdx.y;
		if (x >= width || y >= height)
		{
			return;
		}
		uint ind = x + y * width;

		float sum = 0.f;
		float3 t = { 0.f, 0.f, 0.f };
		float3 center = { input[ind * 3], input[ind * 3 + 1], input[ind * 3 + 2] };
		for (int i = -(int)radius; i <= (int)radius; i++)
		{
			for (int j = -(int)radius; j <= (int)radius; j++)
			{
				uint curPixInd = min((int)width - 1, max(0, (x + j))) + min((int)height - 1, max(0, (y + i))) * width;
				float3 curPix = { input[curPixInd * 3], input[curPixInd * 3 + 1], input[curPixInd * 3 + 2] };
				float factor = gaussianConst[i + radius] * gaussianConst[j + radius] *
					euclideanLength(curPix, center, delta);
				t = t + factor * curPix;
				sum = sum + factor;
			}
		}
		t = t / sum;
		output[ind * 3] = t.x;
		output[ind * 3 + 1] = t.y;
		output[ind * 3 + 2] = t.z;
	}
}
