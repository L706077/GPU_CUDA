//#include "../common/common.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <time.h>

extern "C" void sumArraysOnGPU1(float*d_A, float*d_B, float*d_C, float *h_A, float *h_B, size_t nBytes, float *gpuRef, float *hostRef);
clock_t t1, t2, t3;


__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}



void sumArraysOnGPU1(float*d_A, float*d_B, float*d_C, float *h_A, float *h_B, size_t nBytes, float *gpuRef, float *hostRef)
{

t1=clock();
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;//24
    printf("Vector size %d\n", nElem);


    int iLen = 1024;//512
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);
t2=clock();

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    //sumArraysOnGPU<<<32768, block>>>(d_A, d_B, d_C, nElem);
    //CHECK(cudaDeviceSynchronize());
    //cudaDeviceSynchronize();
t3=clock();
 // check kernel error
    CHECK(cudaGetLastError()) ;

    printf("sumArraysOnGPU1 total Time:  %f sec\n", (double)(t3-t1)/(CLOCKS_PER_SEC));
    printf("sumArraysOnGPU Time elapsed:  %f sec\n", (double)(t3-t2)/(CLOCKS_PER_SEC));

}






