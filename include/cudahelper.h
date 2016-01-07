#ifndef CUDAMATHHELPER_H
#define CUDAMATHHELPER_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define __hd__ __host__ __device__
#define M_EPSILON	1E-9
#define M_INF		1E20

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
#define CUFREE(x) if(x) cudaFree(x);
#endif

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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

#endif

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
	return make_float3(lhs/rhs.x, lhs/rhs.y, lhs/rhs.z);
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
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline __hd__ float3 normalize(const float3& a)
{
	return a / length(a);
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