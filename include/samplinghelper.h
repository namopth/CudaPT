#ifndef SAMPLINGHELPER_H
#define SAMPLINGHELPER_H

#include "mathhelper.h"

namespace NPSamplingHelper
{
	// ==================
	// Hammersley - Begin
	// ==================

	float radicalInverse_VdC(unsigned int bits) {
		bits = (bits << 16u) | (bits >> 16u);
		bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
		bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
		bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
		bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
		return float(bits) * 2.3283064365386963e-10; // / 0x100000000
	}

	NPMathHelper::Vec2 hammersley2d(unsigned int i, unsigned int N) {
		return NPMathHelper::Vec2(float(i) / float(N), radicalInverse_VdC(i));
	}

	NPMathHelper::Vec3 hemisphereSample_uniform(float u, float v) {
		float phi = v * 2.0 * M_PI;
		float cosTheta = 1.0 - u;
		float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
		return NPMathHelper::Vec3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
	}

	NPMathHelper::Vec3 hemisphereSample_cos(float u, float v) {
		float phi = v * 2.0 * M_PI;
		float cosTheta = sqrt(1.0 - u);
		float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
		return NPMathHelper::Vec3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
	}

	// ================
	// Hammersley - End
	// ================
};
#endif