#include <cuda_runtime.h>

extern "C" void sumMatrixOnGPU2D2(float *MatA, float *MatB, float *MatC, int nx, int ny, int dimx, int dimy);
// grid 1D block 1D
// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}




void sumMatrixOnGPU2D2(float *MatA, float *MatB, float *MatC, int nx, int ny, int dimx, int dimy)
{
	//dim3 block(dimx, 1);
        //dim3 grid((nx + block.x - 1) / block.x, 1);
        dim3 block(dimx, dimy);
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);	

	sumMatrixOnGPU2D<<<grid, block>>>(MatA, MatB, MatC, nx, ny);

}
