#include <stdlib.h>
#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

extern "C" void add_vector_gpu( float* a, float* b, float *c, int size );
__global__ void VectorAdd( float* arrayA, float* arrayB, float* output )
{
	int idx = threadIdx.x;
	output[idx] = arrayA[idx] + arrayB[idx];
}

extern "C" void add_vector_gpu( float* a, float* b, float *c, int size );
void add_vector_gpu( float* a, float* b, float *c, int size )
{
	int	data_size = size * sizeof(float);

	// part1, allocate data on device
	float	*dev_A,	*dev_B,	*dev_C;
	cudaMalloc( (void**)&dev_A, data_size );
	cudaMalloc( (void**)&dev_B, data_size );
	cudaMalloc( (void**)&dev_C, data_size );

	// part2, copy memory to device
	cudaMemcpy( dev_A, a, data_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_B, b, data_size, cudaMemcpyHostToDevice );

	// part3, run kernel
	VectorAdd<<< 1, size >>>( dev_A, dev_B, dev_C );

	// part4, copy data from device
	cudaMemcpy( c, dev_C, data_size, cudaMemcpyDeviceToHost );

	// part5, release data
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
}
