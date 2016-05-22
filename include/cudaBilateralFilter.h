#include "cudahelper.h"

#define GAUSSIANCOST_N 64

namespace cudaBilateralFilter
{
	void updateGaussian(float* deviceGaussianConst, float delta, uint radius);
	__global__ void bilaterial_kernel(float* input, uint width, uint height, float delta, uint radius, float* gaussianConst, float* output);
	__global__ void bilaterial_posnorm_kernel(float* color, float* pos, float* norm, uint width, uint height
		, float colorEuD, float posEuD, float normEuD, uint radius, float* gaussianConst, float* output);
	__global__ void bilaterial_posnormemit_kernel(float* color, float* pos, float* norm, float* emit, uint width, uint height
		, float colorEuD, float posEuD, float normEuD, float emitEuD, uint radius, float* gaussianConst, float* output);
	__global__ void bilaterial_posnormemitdiff_kernel(float* color, float* pos, float* norm, float* emit, float* diff, uint width, uint height
		, float colorEuD, float posEuD, float normEuD, float emitEuD, float diffEuD, uint radius, float* gaussianConst, float* output);
}