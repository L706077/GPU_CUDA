#include <cuda_runtime.h>

extern "C" void sumMatrixOnGPU2D1(float *MatA, float *MatB, float *MatC, int nx, int ny, int dimx);
// grid 1D block 1D
// grid 2D block 2D
// grid 2D block 1D
__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}




void sumMatrixOnGPU2D1(float *MatA, float *MatB, float *MatC, int nx, int ny, int dimx)
{
   	 dim3 block(dimx, 1);
    	 dim3 grid((nx + block.x - 1) / block.x, ny);	

	//sumMatrixOnGPU2D<<<grid, block>>>(MatA, MatB, MatC, nx, ny);
	sumMatrixOnGPUMix<<<grid, block>>>(MatA, MatB, MatC, nx, ny);

}
