#include "cudahelper.h"
#include "cudarayhelper.h"

#define BLOCK_SIZE 32

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

__global__ void test(const NPCudaRayHelper::Scene::DeviceScene* scene, int *c)
{
	*c = *scene->models[0].meshes[0].vertN;
}

__global__ void test2(int a, int *c)
{
	*c = a;
}

int cuda_test2(NPCudaRayHelper::Scene* scene)
{
	int c;
	int *d_c;
	int size = sizeof(int);
	scene->GenerateDeviceData();
	if (scene->models.size() > 0)
	{
		cudaMalloc((void**)&d_c, size);
		test << < 1, 1 >> >(scene->devScene, d_c);

		cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

		cudaFree(d_c);
		return c;
	}
	return 0;
}

__global__ void pt_kernel(float3 camPos, float3 camDir, float3 camUp, float3 camRight, float fov
	, NPCudaRayHelper::Scene::DeviceScene* scene
	, float width, float height, float* result)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint ind = (y * width + x) * 3;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	float u = (2.f * ((float)x + 0.5f) / width - 1.f) * tan(fov * 0.5f) * width / height;
	float v = (2.f * ((float)y + 0.5f) / height - 1.f) * tan(fov * 0.5f);
	float3 dir = normalize(camRight * u + camUp * v + camDir);
	NPCudaRayHelper::Ray ray(camPos, dir);

	{
		uint modelId, meshId, triId;
		float hitDist;
		float _w, _u, _v;
		if (scene->Intersect(ray, modelId, meshId, triId, hitDist, _w, _u, _v))
		{
			result[ind] = _w;
			result[ind + 1] = _u;
			result[ind + 2] = _v;
		}
		else
		{
			result[ind] = dir.x;
			result[ind + 1] = dir.y;
			result[ind + 2] = dir.z;
		}
	}
}

void cuda_pt(float3 camPos, float3 camDir, float3 camUp, float fov, NPCudaRayHelper::Scene* scene
	,float width, float height, float* result)
{
	float *d_result;
	int size = sizeof(float) * 3 * width * height;
	cudaMalloc((void**)&d_result, size);

	scene->GenerateDeviceData();

	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 grid(width / block.x, height / block.y, 1);

	if (abs(1.f - vecDot(camUp, camDir)) < 1E-5)
		camUp = make_float3(1.f, 0.f, 0.f);
	float3 camRight = normalize(vecCross(camDir, camUp));
	camUp = normalize(vecCross(camRight, camDir));
	pt_kernel << < grid, block >> > (camPos, camDir, camUp, camRight, fov, scene->devScene, width, height, d_result);

	cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
	cudaFree(d_result);
}