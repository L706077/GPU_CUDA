//#include "../common/common.h"
//#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <stdlib.h>  //srand()
#include <string.h>  //memset()
/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */
extern "C" void sumArraysOnGPU1(float*d_A, float*d_B, float*d_C, float *h_A, float *h_B, size_t nBytes, float *gpuRef, float *hostRef);


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


int main(int argc, char **argv)
{
    clock_t t1, t2, t3, t4, t5, t6;
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

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    //iStart = seconds();
  t1=clock();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    //iElaps = seconds() - iStart;
  t2=clock();  
    printf("initialData Time elapsed %f sec\n", (double)(t2-t1)/(CLOCKS_PER_SEC));
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
  t3=clock();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
  t4=clock();
    printf("sumArraysOnHost Time elapsed %f sec\n", (double)(t4-t3)/(CLOCKS_PER_SEC));

   
//t5=clock();
    sumArraysOnGPU1(d_A, d_B, d_C, h_A, h_B, nBytes, gpuRef, hostRef);

//t6=clock();
    //printf("sumArraysOnGPU Time elapsed %f sec\n",(double)(t6-t5)/(CLOCKS_PER_SEC));


    return(0);
}

















