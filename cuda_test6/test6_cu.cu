#include <cuda_runtime.h>

extern "C" void sumMatrixOnGPU1D1(float *MatA, float *MatB, float *MatC, int nx, int ny, int dimx);
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




void sumMatrixOnGPU1D1(float *MatA, float *MatB, float *MatC, int nx, int ny, int dimx)
{
	dim3 block(dimx, 1);
        dim3 grid((nx + block.x - 1) / block.x, 1);
	sumMatrixOnGPU1D<<<grid, block>>>(MatA, MatB, MatC, nx, ny);

}
