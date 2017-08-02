//#include "../common/common.h"
//#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <stdlib.h>  //srand()
#include <string.h>  //memset()
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */
extern "C" void sumArraysOnGPU1(float*d_A, float*d_B, float*d_C, float *h_A, float *h_B, size_t nBytes, float *gpuRef, float *hostRef);
#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void checkResult1(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}



int main(int argc, char **argv)
{
    clock_t t1, t2, t3, t4, t5, t6, t7;
    printf("%s Starting...\n", argv[0]);

    // set up device
    //int dev = 0;
    //cudaDeviceProp deviceProp;
    //CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    //printf("Using Device %d: %s\n", dev, deviceProp.name);
    //CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;//24
    printf("Vector size %d\n", nElem);
t1=clock();
    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *d_A1, *d_B1, *d_C1;
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    // malloc device memory
    CHECK(cudaMalloc((float**)&d_A1, nBytes));
    CHECK(cudaMalloc((float**)&d_B1, nBytes));
    CHECK(cudaMalloc((float**)&d_C1, nBytes));



//t1=clock();
    initialData(h_A, nElem);
    initialData(h_B, nElem);

t2=clock();  
    printf("initialData Time elapsed %f sec\n", (double)(t2-t1)/(CLOCKS_PER_SEC));
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A1, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B1, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C1, gpuRef, nBytes, cudaMemcpyHostToDevice));



    // add vector at host side for result checks
t3=clock();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
t4=clock();

    printf("out loop sumArraysOnHost Time elapsed:  %f sec\n", (double)(t4-t3)/(CLOCKS_PER_SEC));

   
t5=clock();
    sumArraysOnGPU1(d_A1, d_B1, d_C1, h_A, h_B, nBytes, gpuRef, hostRef);   /////!!!!!!!!!!!!!!!!!!!!!!!!!1
t6=clock();

    printf("out loop sumArraysOnGPU Time elapsed:  %f sec\n",(double)(t6-t5)/(CLOCKS_PER_SEC));

	cudaMemcpy(gpuRef, d_C1, nBytes, cudaMemcpyDeviceToHost);

t7=clock();	
    printf("out loop sumArraysOnGPU total Time:  %f sec\n",(double)(t7-t5)/(CLOCKS_PER_SEC));
	checkResult1(hostRef, gpuRef, nElem);


    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);


    return(0);
}

















