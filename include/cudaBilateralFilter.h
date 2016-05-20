#include "cudahelper.h"

#define GAUSSIANCOST_N 64

namespace cudaBilateralFilter
{
	void updateGaussian(float* deviceGaussianConst, float delta, uint radius);
	__global__ void bilaterial_kernel(float* input, uint width, uint height, float delta, uint radius, float* gaussianConst, float* output);
}