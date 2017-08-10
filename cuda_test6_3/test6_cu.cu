#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>  //srand()
#include <iostream>  //cout
#include <string.h>  //memset()
extern "C" void gpuTestAll(float *MatA, float *MatB, float *MatC, int nx, int ny);
// grid 1D block 1D
// grid 2D block 2D
// grid 2D block 1D
// grid 2D block 2D

#define CHECK(status);									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}


__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

// grid 1D block 1D
__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx )
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }


}

// grid 2D block 1D
__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx,
                                  int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

////=================================================
////=================================================




void gpuTestAll(float *d_MatA, float *d_MatB, float *d_MatC, int nx, int ny)
{
    clock_t t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34;
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

t5=clock();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t6=clock();
    printf("sumMatrixOnGPU2D  <<<  (512,512),  (32,32)  >>> elapsed %f sec\n", (double)(t6-t5)/(CLOCKS_PER_SEC));

    // adjust block size
    block.x = 16;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

t7=clock();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t8=clock();
    printf("sumMatrixOnGPU2D  <<<  (1024,512), (16,32)  >>> elapsed %f sec\n", (double)(t8-t7)/(CLOCKS_PER_SEC));

    // adjust block size
    block.y = 16;
    block.x = 32;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

t9=clock();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t10=clock();
    printf("sumMatrixOnGPU2D  <<<  (512,1024), (32,16)  >>> elapsed %f sec\n", (double)(t10-t9)/(CLOCKS_PER_SEC));

    block.y = 16;
    block.x = 16;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

t11=clock();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t12=clock();
    printf("sumMatrixOnGPU2D  <<<  (1024,1024),(16,16)  >>> elapsed %f sec\n", (double)(t12-t11)/(CLOCKS_PER_SEC));

    block.y = 16;
    block.x = 64;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

t13=clock();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t14=clock();
    printf("sumMatrixOnGPU2D  <<<  (256,1024), (64,16)  >>> elapsed %f sec\n", (double)(t14-t13)/(CLOCKS_PER_SEC));

    block.y = 64;
    block.x = 16;
    grid.x  = (nx + block.x - 1) / block.x;
    grid.y  = (ny + block.y - 1) / block.y;

t15=clock();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t16=clock();
    printf("sumMatrixOnGPU2D  <<<  (1024,256), (16,64)  >>> elapsed %f sec\n", (double)(t16-t15)/(CLOCKS_PER_SEC));
printf("\n");

    block.x = 32;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = 1;

t17=clock();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t18=clock();
    printf("sumMatrixOnGPU1D  <<<  (512,1)   , (32,1)   >>> elapsed %f sec\n", (double)(t18-t17)/(CLOCKS_PER_SEC));

    block.x = 64;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = 1;

t19=clock();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t20=clock();
    printf("sumMatrixOnGPU1D  <<<  (256,1)   , (64,1)   >>> elapsed %f sec\n", (double)(t20-t19)/(CLOCKS_PER_SEC));

    block.x = 128;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = 1;

t21=clock();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t22=clock();;
    printf("sumMatrixOnGPU1D  <<<  (128,1)   , (128,1)  >>> elapsed %f sec\n", (double)(t22-t21)/(CLOCKS_PER_SEC));

printf("\n");
    // grid 2D and block 1D
    block.x = 32;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

t23=clock();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t24=clock();
    printf("sumMatrixOnGPUMix <<<  (512,16384),(32,1)   >>> elapsed %f sec\n",(double)(t24-t23)/(CLOCKS_PER_SEC));

    block.x = 64;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

t25=clock();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t26=clock();
    printf("sumMatrixOnGPUMix <<<  (256,16384),(64,1)   >>> elapsed %f sec\n",(double)(t26-t25)/(CLOCKS_PER_SEC));

    block.x = 128;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

t27=clock();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t28=clock();
    printf("sumMatrixOnGPUMix <<<  (128,16384),(128,1)  >>> elapsed %f sec\n",(double)(t28-t27)/(CLOCKS_PER_SEC));

    block.x = 256;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

t29=clock();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t30=clock();
    printf("sumMatrixOnGPUMix <<<  (64,16384), (256,1)  >>> elapsed %f sec\n",(double)(t30-t29)/(CLOCKS_PER_SEC));

    block.x = 512;
    grid.x  = (nx + block.x - 1) / block.x;
    block.y = 1;
    grid.y  = ny;

t31=clock();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
t32=clock();
    printf("sumMatrixOnGPUMix <<<  (32,16384), (512,1)  >>> elapsed %f sec\n",(double)(t32-t31)/(CLOCKS_PER_SEC));




}













