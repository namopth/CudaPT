#include "cudaMathHelper.cuh"

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

	add << < 1, 1 >> >(d_a, d_b, d_c);

	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return c;
}