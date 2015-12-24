#ifndef CUDAMATHHELPER_H
#define CUDAMATHHELPER_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define __hd__ __host__ __device__
#define M_EPSILON	1E-9
#define M_INF		1E20

#ifndef __CUDACC__
#include <math.h>

inline float fminf(float a, float b)
{
	return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
	return a > b ? a : b;
}

inline int max(int a, int b)
{
	return a > b ? a : b;
}

inline int min(int a, int b)
{
	return a < b ? a : b;
}

inline float rsqrtf(float x)
{
	return 1.0f / sqrtf(x);
}

#endif
//#include <math_functions.hpp>

typedef unsigned int uint;
typedef unsigned short ushort;

// float3 - bgn
inline __hd__ float3 vecCross(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.y*rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x);
}

inline __hd__ float vecDot(const float3& lhs, const float3 &rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline __hd__ float3 operator+(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

inline __hd__ float3 operator-(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

inline __hd__ float3 operator*(const float3& lhs, const float& rhs)
{
	return make_float3(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
}

inline __hd__ float3 operator*(const float& lhs, const float3& rhs)
{
	return rhs*lhs;
}

inline __hd__ float3 operator/(const float3& lhs, const float3& rhs)
{
	return make_float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
}

inline __hd__ float3 operator/(const float3& lhs, const float& rhs)
{
	return make_float3(lhs.x/rhs, lhs.y/rhs, lhs.z/rhs);
}

inline __hd__ float3 operator/(const float& lhs, const float3& rhs)
{
	return make_float3(lhs/rhs.x, lhs/rhs.x, lhs/rhs.x);
}

inline __hd__ float operator*(const float3& lhs, const float3& rhs)
{
	return vecDot(lhs,rhs);
}

inline __hd__ float3 operator%(const float3& lhs, const float3& rhs)
{
	return vecCross(lhs, rhs);
}

inline __hd__ float length(const float3& a)
{
	return a.x * a.x + a.y * a.y + a.z * a.z;
}

inline __hd__ float3 normalize(const float3& a)
{
	return a / length(a);
}
// float3 - end

#endif