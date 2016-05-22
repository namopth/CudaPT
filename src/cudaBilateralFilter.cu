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

	__global__ void bilaterial_posnorm_kernel(float* color, float* pos, float* norm, uint width, uint height, float colorEuD, float posEuD, float normEuD, uint radius, float* gaussianConst, float* output)
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
		float3 cenColor = { color[ind * 3], color[ind * 3 + 1], color[ind * 3 + 2] };
		float3 cenPos = { pos[ind * 3], pos[ind * 3 + 1], pos[ind * 3 + 2] };
		float3 cenNorm = { norm[ind * 3], norm[ind * 3 + 1], norm[ind * 3 + 2] };
		for (int i = -(int)radius; i <= (int)radius; i++)
		{
			for (int j = -(int)radius; j <= (int)radius; j++)
			{
				uint curPixInd = min((int)width - 1, max(0, (x + j))) + min((int)height - 1, max(0, (y + i))) * width;
				float3 curPix = { color[curPixInd * 3], color[curPixInd * 3 + 1], color[curPixInd * 3 + 2] };
				float3 curPos = { pos[curPixInd * 3], pos[curPixInd * 3 + 1], pos[curPixInd * 3 + 2] };
				float3 curNorm = { norm[curPixInd * 3], norm[curPixInd * 3 + 1], norm[curPixInd * 3 + 2] };
				float factor = gaussianConst[i + radius] * gaussianConst[j + radius] *
					euclideanLength(curPix, cenColor, colorEuD)*
					euclideanLength(curPos, cenPos, posEuD)*
					euclideanLength(curNorm, cenNorm, normEuD);
				t = t + factor * curPix;
				sum = sum + factor;
			}
		}
		t = t / sum;
		output[ind * 3] = t.x;
		output[ind * 3 + 1] = t.y;
		output[ind * 3 + 2] = t.z;
	}

	__global__ void bilaterial_posnormemit_kernel(float* color, float* pos, float* norm, float* emit, uint width, uint height
		, float colorEuD, float posEuD, float normEuD, float emitEuD, uint radius, float* gaussianConst, float* output)
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
		float3 cenColor = { color[ind * 3], color[ind * 3 + 1], color[ind * 3 + 2] };
		float3 cenPos = { pos[ind * 3], pos[ind * 3 + 1], pos[ind * 3 + 2] };
		float3 cenNorm = { norm[ind * 3], norm[ind * 3 + 1], norm[ind * 3 + 2] };
		float3 cenEmit = { emit[ind * 3], emit[ind * 3 + 1], emit[ind * 3 + 2] };
		for (int i = -(int)radius; i <= (int)radius; i++)
		{
			for (int j = -(int)radius; j <= (int)radius; j++)
			{
				uint curPixInd = min((int)width - 1, max(0, (x + j))) + min((int)height - 1, max(0, (y + i))) * width;
				float3 curPix = { color[curPixInd * 3], color[curPixInd * 3 + 1], color[curPixInd * 3 + 2] };
				float3 curPos = { pos[curPixInd * 3], pos[curPixInd * 3 + 1], pos[curPixInd * 3 + 2] };
				float3 curNorm = { norm[curPixInd * 3], norm[curPixInd * 3 + 1], norm[curPixInd * 3 + 2] };
				float3 curEmit = { emit[curPixInd * 3], emit[curPixInd * 3 + 1], emit[curPixInd * 3 + 2] };
				float factor = gaussianConst[i + radius] * gaussianConst[j + radius] *
					euclideanLength(curPix, cenColor, colorEuD)*
					euclideanLength(curPos, cenPos, posEuD)*
					euclideanLength(curNorm, cenNorm, normEuD)*
					euclideanLength(curEmit, cenEmit, emitEuD);
				t = t + factor * curPix;
				sum = sum + factor;
			}
		}
		t = t / sum;
		output[ind * 3] = t.x;
		output[ind * 3 + 1] = t.y;
		output[ind * 3 + 2] = t.z;
	}

	__global__ void bilaterial_posnormemitdiff_kernel(float* color, float* pos, float* norm, float* emit, float* diff, uint width, uint height
		, float colorEuD, float posEuD, float normEuD, float emitEuD, float diffEuD, uint radius, float* gaussianConst, float* output)
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
		float3 cenColor = { color[ind * 3], color[ind * 3 + 1], color[ind * 3 + 2] };
		float3 cenPos = { pos[ind * 3], pos[ind * 3 + 1], pos[ind * 3 + 2] };
		float3 cenNorm = { norm[ind * 3], norm[ind * 3 + 1], norm[ind * 3 + 2] };
		float3 cenEmit = { emit[ind * 3], emit[ind * 3 + 1], emit[ind * 3 + 2] };
		float3 cenDiff = { diff[ind * 3], diff[ind * 3 + 1], diff[ind * 3 + 2] };
		for (int i = -(int)radius; i <= (int)radius; i++)
		{
			for (int j = -(int)radius; j <= (int)radius; j++)
			{
				uint curPixInd = min((int)width - 1, max(0, (x + j))) + min((int)height - 1, max(0, (y + i))) * width;
				float3 curPix = { color[curPixInd * 3], color[curPixInd * 3 + 1], color[curPixInd * 3 + 2] };
				float3 curPos = { pos[curPixInd * 3], pos[curPixInd * 3 + 1], pos[curPixInd * 3 + 2] };
				float3 curNorm = { norm[curPixInd * 3], norm[curPixInd * 3 + 1], norm[curPixInd * 3 + 2] };
				float3 curEmit = { emit[curPixInd * 3], emit[curPixInd * 3 + 1], emit[curPixInd * 3 + 2] };
				float3 curDiff = { diff[curPixInd * 3], diff[curPixInd * 3 + 1], diff[curPixInd * 3 + 2] };
				float factor = gaussianConst[i + radius] * gaussianConst[j + radius] *
					euclideanLength(curPix, cenColor, colorEuD)*
					euclideanLength(curPos, cenPos, posEuD)*
					euclideanLength(curNorm, cenNorm, normEuD)*
					euclideanLength(curEmit, cenEmit, emitEuD)*
					euclideanLength(curDiff, cenDiff, diffEuD);
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
