#include <cuda_runtime.h>
#include <stdio.h>
//#include <stdbool.h>
extern "C" void MeanFilterCUDA(unsigned char* h_in, unsigned char* h_out, int nKernelSize, int rows, int cols);

//template <typename T> __global__ void MeanFilterCUDAkernel(T* pInput, T* pOutput, int nKernelSize, int nHeight, int nWidth)
__global__ void MeanFilterCUDAkernel(unsigned char* pInput, unsigned char* pOutput, int nKernelSize, int nHeight, int nWidth)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = blockIdx.y;
    int pos = j*nWidth + i;  //pixel index
    
    if( i>0 && i < nWidth-1 && j > 0 && j < nHeight-1)  //process scope
    {
        float temp1;
        temp1 += pInput[pos]; 
        temp1 += pInput[pos+1]; 
        temp1 += pInput[pos-1]; 
        temp1 += pInput[pos - nWidth]; 
        temp1 += pInput[pos - nWidth + 1]; 
        temp1 += pInput[pos - nWidth - 1]; 
        temp1 += pInput[pos + nWidth]; 
        temp1 += pInput[pos + nWidth + 1]; 
        temp1 += pInput[pos + nWidth - 1];
        pOutput[pos] = (unsigned char)(temp1/nKernelSize);
    }
    else
    {
        pOutput[pos]=pInput[pos];    
    }
}

extern "C" void MeanFilterCUDA(unsigned char* h_in, unsigned char* h_out, int nKernelSize, int rows, int cols){

	printf("rows_kernel: %d \n", rows);
	printf("cols_kernel: %d \n", cols);

	dim3 block(256,1,1);
	dim3 grid((cols+255)/block.x, rows, 1);

	unsigned char* d_in;
	unsigned char* d_out;
	cudaMalloc((void**) &d_in, rows*cols);
	cudaMalloc((void**) &d_out, rows*cols);

	cudaMemcpy(d_in, h_in, rows*cols*sizeof(unsigned char), cudaMemcpyHostToDevice); //input

        MeanFilterCUDAkernel<<< grid, block >>>(d_in, d_out, nKernelSize, rows, cols);
   
	cudaMemcpy(h_out, d_out, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost); //output
	cudaFree(d_in);
	cudaFree(d_out);

}












