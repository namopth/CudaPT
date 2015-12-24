#include "cudaMathHelper.cuh"

struct Ray
{
	float3 orig;
	float3 dir;
	__hd__ Ray(float3 o, float3 d) : orig(o), dir(d) {}
};

struct Sphere
{
	float3 center;
	float radius;
	__hd__ Sphere(const float3 c, const float r) : center(c), radius(r) {}
	__hd__ bool intersect(const Ray &r, float3 &hitPoint, float3 &hitNormal)
	{
		float3 co = r.orig - center;
		float a = r.dir*r.dir;
		if (fabs(a) < M_EPSILON)
			return false;
		float b = 2.f * r.dir*co;
		float c = co*co - radius * radius;
		float root = b * b - 4.f * a * c;
		if (root < 0)
			return false;
		root = sqrtf(root);
		float t0 = fmax((-b + root) / 2.f * a, 0.f);
		float t1 = fmax((-b - root) / 2.f * a, 0.f);
		float t = fmin(t0, t1);
		if (t <= 0)
			return false;
		hitPoint = r.orig + r.dir * t;
		hitNormal = normalize(hitPoint - center);
		return true;
	}
};


struct AABBBox
{
	float3 minPoint;
	float3 maxPoint;
	__hd__ AABBBox(float3 min, float3 max)
		:minPoint(min), maxPoint(max)
	{}

	__hd__ bool intersect(const Ray &r, float3 &hitPoint, float3 &hitNormal)
	{
		float3 modDir = r.dir;
		modDir.x = escapeZero(modDir.x, M_EPSILON);
		modDir.y = escapeZero(modDir.y, M_EPSILON);
		modDir.z = escapeZero(modDir.z, M_EPSILON);
		float3 tmin = (minPoint - r.orig) / modDir;
		float3 tmax = (maxPoint - r.orig) / modDir;
		float3 real_min = vecmin(tmin, tmax);
		float3 real_max = vecmax(tmin, tmax);
		float minmax = min(min(real_max.x, real_max.y), real_max.z);
		float maxmin = max(max(real_min.x, real_min.y), real_min.z);
		if (minmax >= maxmin && maxmin > M_EPSILON)
		{
			hitPoint = r.orig + r.dir * maxmin;
			hitNormal = (maxmin == real_min.x) ? make_float3(1.f, 0.f, 0.f) :
				(maxmin == real_min.y) ? make_float3(0.f, 1.f, 0.f) : make_float3(0.f, 0.f, 1.f);
			if (hitNormal*r.dir > 0.f)
				hitNormal = -1 * hitNormal;
			return true;
		}
		return false;
	}

protected:
	__hd__ float escapeZero(const float value, const float epsilon)
	{
		float result = value;
		if (fabs(result) < epsilon)
			result = (result > 0) ? result + epsilon : result - epsilon;
		return result;
	}
	__hd__ float3 vecmin(const float3& lhs, const float3& rhs)
	{
		return make_float3(min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z));
	}
	__hd__ float3 vecmax(const float3& lhs, const float3& rhs)
	{
		return make_float3(max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z));
	}
};

struct Tri
{
	// CCW
	float3 p0;
	float3 p1;
	float3 p2;

	__hd__ Tri(float3 a, float3 b, float3 c)
		: p0(a)
		, p1(b)
		, p2(c)
	{

	}

	__hd__ bool intersect(const Ray &r, float3 &hitPoint, float3 &hitNormal, float &w, float &u, float &v)
	{
		float3 e1 = p1 - p0;
		float3 e2 = p2 - p0;
		if ((e1%e2)*r.dir > 0.f) return false;
		float3 de2 = r.dir%e2;
		float divisor = de2*e1;
		if (fabs(divisor) < M_EPSILON)
			return false;
		float3 t = r.orig - p0;
		float3 te1 = t%e1;
		float rT = (te1*e2) / divisor;
		if (rT < 0.f)
			return false;
		u = de2*t;
		v = te1*r.dir;
		w = 1 - u - v;
		if (u < 0.f || u > 1.f || v > 1.f || v < 0.f || w > 1.f || w < 0.f)
			return false;
		hitPoint = r.orig + rT * r.dir;
		hitNormal = normalize(e1%e2);
		return true;
	}
};

__global__ void render_kernel(float* output, uint width, uint height)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint ind = y * width + x;
	output[ind] = x / width;
	output[ind + 1] = y / height;
	output[ind + 2] = 1.f;
}

void render(float* output, uint width, uint height)
{
	dim3 block(1, 1, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	render_kernel << < grid, block >> > (output, width, height);
}

__global__ void add(int *a, int *b, int *c)
{
	*c = *b + *a;
}

int cuda_test(int a, int b)
{
	int c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add <<< 1, 1 >>>(d_a, d_b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return c;
}