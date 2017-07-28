//#include "cuPrintf.cu"
#include <stdio.h>
extern "C" void kernel_wrapper(int *a, int *b);

__global__ void kernel(int *a, int *b){
    int tx = threadIdx.x;
//  cuPrintf("tx = %d\n", tx);
    switch( tx ){
    case 0:
        *a = *a + 10;
        break;
    case 1:
        *b = *b + 3;
        break;
    default:
        break;
    }
}

void kernel_wrapper(int *a, int *b){
//  cudaPrintfInit();
    //cuPrintf("Anything...?");
    printf("Anything...?\n");
    int *d_1, *d_2;
    dim3 threads( 2, 1 );
    dim3 blocks( 1, 1 );

    cudaMalloc( (void **)&d_1, sizeof(int) );
    cudaMalloc( (void **)&d_2, sizeof(int) );

    cudaMemcpy( d_1, a, sizeof(int), cudaMemcpyHostToDevice );
    cudaMemcpy( d_2, b, sizeof(int), cudaMemcpyHostToDevice );

    kernel<<< blocks, threads >>>( d_1, d_2 );
    cudaMemcpy( a, d_1, sizeof(int), cudaMemcpyDeviceToHost );
    cudaMemcpy( b, d_2, sizeof(int), cudaMemcpyDeviceToHost );
    printf("Output: a = %d\n", a[0]);
    cudaFree(d_1);
    cudaFree(d_2);

//  cudaPrintfDisplay(stdout, true);
//  cudaPrintfEnd();
}
