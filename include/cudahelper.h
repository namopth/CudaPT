#ifndef CUDAMATHHELPER_H
#define CUDAMATHHELPER_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#ifdef __CUDACC__
#define __hd__ __host__ __device__
#else
#define __hd__
#endif

#define M_E         2.71828182845904523536028747135266250   /* e */
#define M_LOG2E     1.44269504088896340735992468100189214   /* log 2e */
#define M_LOG10E    0.434294481903251827651128918916605082  /* log 10e */
#define M_LN2       0.693147180559945309417232121458176568  /* log e2 */
#define M_LN10      2.30258509299404568401799145468436421   /* log e10 */
#define M_PI        3.14159265358979323846264338327950288   /* pi */
#define M_PI_2      1.57079632679489661923132169163975144   /* pi/2 */
#define M_PI_4      0.785398163397448309615660845819875721  /* pi/4 */
#define M_1_PI      0.318309886183790671537767526745028724  /* 1/pi */
#define M_2_PI      0.636619772367581343075535053490057448  /* 2/pi */
#define M_2_SQRTPI  1.12837916709551257389615890312154517   /* 2/sqrt(pi) */
#define M_SQRT2     1.41421356237309504880168872420969808   /* sqrt(2) */
#define M_SQRT1_2   0.707106781186547524400844362104849039  /* 1/sqrt(2) */
#define M_EPSILON	1E-9
#define M_FLT_EPSILON 1E-5
#define M_FLT_BIAS_EPSILON 1E-4
#define M_INF		1E20
#define M_MIN_INF	-1E20

#define USE_CUDA_INTRINSIC

#if defined(_DEBUG)
#   define NEW  new//new(_NORMAL_BLOCK,__FILE__, __LINE__)
#else
#	define NEW  new
#endif

#if !defined(DEL)
#define DEL(x) if(x) delete x; x=NULL;
#endif

#if !defined(DEL_SIZE)
#define DEL_SIZE(x,y) if(x) delete(x,y); x=NULL;
#endif

#if !defined(DEL_ARRAY)
#define DEL_ARRAY(x) if (x) delete [] x; x=NULL; 
#endif

#if !defined(REL)
#define REL(x) if(x) x->Release(); x=NULL;
#endif

#if !defined(CUFREE)
#define CUFREE(x) if(x) cudaFree(x); x = NULL;
#endif

#if !defined(CUFREEARRAY)
#define CUFREEARRAY(x) if(x) cudaFreeArray(x);
#endif

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		//exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define HANDLE_KERNEL_ERROR() (HandleError( cudaPeekAtLastError(), __FILE__, __LINE__ ))

#ifndef __CUDA_ARCH__
#include <vector>
#endif

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

inline __hd__ float saturate(float x)
{
	return (x < 0.f) ? 0.f : (x > 1.f) ? 1.f : x;
}

#endif

typedef unsigned int uint;
typedef unsigned short ushort;

inline __hd__ float escapeZero(const float value, const float epsilon)
{
	float result = value;
	if (fabs(result) < epsilon)
		result = (result > 0) ? result + epsilon : result - epsilon;
	return result;
}

inline __hd__ float rcpf(float x)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return __frcp_rn(x);
#else
	return 1.f / x;
#endif
}

inline __hd__ float lerp(float x, float y, float d)
{
	return x + (y - x) * d;
}

// float3 - bgn
inline __hd__ float3 vecMin(const float3& lhs, const float3& rhs)
{
	return make_float3(fminf(lhs.x, rhs.x), fminf(lhs.y, rhs.y), fminf(lhs.z, rhs.z));
}

inline __hd__ float3 vecMax(const float3& lhs, const float3& rhs)
{
	return make_float3(fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y), fmaxf(lhs.z, rhs.z));
}

inline __hd__ float3 vecMul(const float3& lhs, const float3& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fmul_rn(lhs.x, rhs.x), __fmul_rn(lhs.y, rhs.y), __fmul_rn(lhs.z, rhs.z));
#else
	return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
#endif
}

inline __hd__ float4 vecMul(const float4& lhs, const float4& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float4(__fmul_rn(lhs.x, rhs.x), __fmul_rn(lhs.y, rhs.y), __fmul_rn(lhs.z, rhs.z), __fmul_rn(lhs.w, rhs.w));
#else
	return make_float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
#endif
}

inline __hd__ float3 vecRcp(const float3& lhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__frcp_rn(lhs.x), __frcp_rn(lhs.y), __frcp_rn(lhs.z));
#else
	return make_float3(1.f / lhs.x, 1.f / lhs.y, 1.f / lhs.z);
#endif
}

inline __hd__ float3 vecCross(const float3& lhs, const float3& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fmaf_rn(lhs.y, rhs.z, -__fmul_rn(lhs.z, rhs.y))
		, __fmaf_rn(lhs.z, rhs.x, -__fmul_rn(lhs.x, rhs.z))
		, __fmaf_rn(lhs.x, rhs.y, -__fmul_rn(lhs.y, rhs.x)));
#else
	return make_float3(lhs.y*rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x);
#endif
}

inline __hd__ float vecDot(const float3& lhs, const float3 &rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return __fmaf_rn(lhs.x,rhs.x,__fmaf_rn(lhs.y,rhs.y,__fmul_rn(lhs.z,rhs.z)));
#else
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
#endif
}

inline __hd__ float3 operator+(const float3& lhs, const float3& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fadd_rn(lhs.x,rhs.x), __fadd_rn(lhs.y,rhs.y), __fadd_rn(lhs.z,rhs.z));
#else
	return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
#endif
}

inline __hd__ float4 operator+(const float4& lhs, const float4& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float4(__fadd_rn(lhs.x,rhs.x), __fadd_rn(lhs.y,rhs.y), __fadd_rn(lhs.z,rhs.z), __fadd_rn(lhs.w,rhs.w));
#else
	return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
#endif
}

inline __hd__ float3 operator-(const float3& lhs, const float3& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fsub_rn(lhs.x,rhs.x), __fsub_rn(lhs.y,rhs.y), __fsub_rn(lhs.z,rhs.z));
#else
	return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
#endif
}

inline __hd__ float3 operator*(const float3& lhs, const float& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fmul_rn(lhs.x, rhs), __fmul_rn(lhs.y, rhs), __fmul_rn(lhs.z, rhs));
#else
	return make_float3(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
#endif
}

inline __hd__ float3 operator*(const float& lhs, const float3& rhs)
{
	return rhs*lhs;
}

inline __hd__ float4 operator*(const float4& lhs, const float& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float4(__fmul_rn(lhs.x, rhs), __fmul_rn(lhs.y, rhs), __fmul_rn(lhs.z, rhs), __fmul_rn(lhs.w, rhs));
#else
	return make_float4(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs, lhs.w*rhs);
#endif
}

inline __hd__ float4 operator*(const float& lhs, const float4& rhs)
{
	return rhs*lhs;
}

inline __hd__ float3 operator/(const float3& lhs, const float3& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fdiv_rn(lhs.x, rhs.x), __fdiv_rn(lhs.y, rhs.y), __fdiv_rn(lhs.z, rhs.z));
#else
	return make_float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z);
#endif
}

inline __hd__ float3 operator/(const float3& lhs, const float& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fdiv_rn(lhs.x, rhs), __fdiv_rn(lhs.y, rhs), __fdiv_rn(lhs.z, rhs));
#else
	return make_float3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
#endif
}

inline __hd__ float3 operator/(const float& lhs, const float3& rhs)
{
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return make_float3(__fdiv_rn(lhs,rhs.x), __fdiv_rn(lhs,rhs.y), __fdiv_rn(lhs,rhs.z));
#else
	return make_float3(lhs/rhs.x, lhs/rhs.y, lhs/rhs.z);
#endif
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
#ifdef __CUDA_ARCH__ && USE_CUDA_INTRINSIC
	return __fsqrt_rd(a.x * a.x + a.y * a.y + a.z * a.z);
#else
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
#endif
}

inline __hd__ float3 normalize(const float3& a)
{
	return a / length(a);
}

inline __hd__ float3 vecLerp(const float3& x, const float3& y, const float d)
{
	return x + (y - x) * d;
}

// float3 - end

// float2 - bgn
inline __hd__ float vecDot(const float2& lhs, const float2 &rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y;
}

inline __hd__ float2 operator+(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

inline __hd__ float2 operator-(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

inline __hd__ float2 operator*(const float2& lhs, const float& rhs)
{
	return make_float2(lhs.x*rhs, lhs.y*rhs);
}

inline __hd__ float2 operator*(const float& lhs, const float2& rhs)
{
	return rhs*lhs;
}

inline __hd__ float2 operator/(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x / rhs.x, lhs.y / rhs.y);
}

inline __hd__ float2 operator/(const float2& lhs, const float& rhs)
{
	return make_float2(lhs.x / rhs, lhs.y / rhs);
}

inline __hd__ float2 operator/(const float& lhs, const float2& rhs)
{
	return make_float2(lhs / rhs.x, lhs / rhs.y);
}

inline __hd__ float operator*(const float2& lhs, const float2& rhs)
{
	return vecDot(lhs, rhs);
}

inline __hd__ float length(const float2& a)
{
	return sqrt(a.x * a.x + a.y * a.y);
}

inline __hd__ float2 normalize(const float2& a)
{
	return a / length(a);
}
// float2 - end

template<typename T>
class CudaVector
{
public:

#ifdef __CUDA_ARCH__
	__device__ CudaVector() : m_pDevData(0), m_pDevLength(0), m_pDevIsDevDirty(0), m_pHostVector(0){}
#else
	__host__ CudaVector() : m_pDevData(0), m_pDevLength(0), m_uLastDevLength(0)
		, m_bIsHostDirty(true), m_bIsDevDirty(false), m_pDevIsDevDirty(0){
		m_pHostVector = new std::vector<T>();
	}
	//__host__ CudaVector(const CudaVector &obj)
	//{
	//	m_pHostVector = new std::vector<T>();
	//	m_pHostVector = obj.m_pHostVector;
	//}
#endif

	__hd__ ~CudaVector() 
	{
		if (m_pDevData)		cudaFree(m_pDevData);
		if (m_pDevLength)	cudaFree(m_pDevLength);
		if (m_pDevIsDevDirty)	cudaFree(m_pDevIsDevDirty);
		if (m_pHostVector) delete m_pHostVector;
	}

#ifdef __CUDA_ARCH__
	__device__ bool GetIsCurrentSideDirty() const { return *m_pDevIsDevDirty; }
	__device__ const T& Get(uint index) const { return m_pDevData[index];}
	__device__ uint Length() const { return *m_pDevLength;}
	__device__ void Pushback(T element) {}
	__device__ T& GetEditableData(uint index) { *m_pDevIsDevDirty = true; return m_pDevData[index]; }
	__device__ void GenerateDeviceData() {}
#else
	__host__ bool GetIsCurrentSideDirty() const { return m_bIsHostDirty; }
	__host__ const T& Get(uint index) const { return m_pHostVector->at(index); }
	__host__ T& GetEditableData(uint index) { m_bIsHostDirty = true; return m_pHostVector->at(index); }
	__host__ uint Length() const { return m_pHostVector->size(); }
	__host__ void Pushback(T element)
	{ 
		m_pHostVector->push_back(element); 
		m_bIsHostDirty = true; 
	}
	__host__ void GenerateDeviceData()
	{
		if (!m_pDevLength) cudaMalloc((void**)&m_pDevLength, sizeof(uint));
		if (!m_pDevIsDevDirty) cudaMalloc((void**)&m_pDevIsDevDirty, sizeof(bool));

		if (m_bIsHostDirty)
		{
			if (m_uLastDevLength != m_pHostVector->size())
			{
				m_uLastDevLength = m_pHostVector->size();
				if (m_pDevData) cudaFree(m_pDevData);
				cudaMalloc((void**)&m_pDevData, m_pHostVector->size() * sizeof(T));
				cudaMemcpy(m_pDevLength, &m_uLastDevLength, sizeof(uint), cudaMemcpyHostToDevice);
			}
			T* tempData = new T[m_pHostVector->size()];
			for (uint i = 0; i < m_pHostVector->size(); i++)
				tempData[i] = m_pHostVector->at(i);
			cudaMemcpy(m_pDevData, tempData, m_pHostVector->size() * sizeof(T), cudaMemcpyHostToDevice);
			delete[] tempData;
			m_bIsHostDirty = false;
		}
	}
#endif

protected:
#ifndef __CUDA_ARCH__
	std::vector<T>* m_pHostVector;
#else
	void* m_pHostVector;
#endif
	uint m_uLastDevLength;
	bool m_bIsHostDirty;
	bool m_bIsDevDirty;

	T* m_pDevData;
	uint* m_pDevLength;
	bool* m_pDevIsDevDirty;
};

#endif