#include <cuda_runtime.h>
#include <stdio.h>
//#include <stdbool.h>
extern "C" void gray_parallel(unsigned char* h_in, unsigned char* h_out, int elems, int rows, int cols);


__global__ void kernel1(unsigned char* d_in, unsigned char* d_out, int rows, int cols){

	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	int idy = blockIdx.y; 
	int index = idx+idy*cols;      
	int clr_adr = 3*index;

        if(index<(rows*cols))
	{	
		double gray_val = 0.144*d_in[clr_adr] + 0.587*d_in[clr_adr+1] + 0.299*d_in[clr_adr+2];
		d_out[index] = (unsigned char)gray_val;
	}
}

__global__ void kernel2(unsigned char* d_in, unsigned char* d_out, int rows, int cols){

	int index= threadIdx.x+blockIdx.x*blockDim.x;
	int clr_adr = 3*index;

        if(index<(rows*cols))
	{	
		double gray_val = 0.144*d_in[clr_adr] + 0.587*d_in[clr_adr+1] + 0.299*d_in[clr_adr+2];
		d_out[index] = (unsigned char)gray_val;
	}
}
//   Kernel Calling Function

extern "C" void gray_parallel(unsigned char* h_in, unsigned char* h_out, int elems, int rows, int cols){

	int checkgrid2D=1;
        dim3 block(cols,1,1);
	//dim3 block(64,1,1);
	dim3 grid(cols+block.x-1/block.x, rows, 1);
	
	unsigned char* d_in;
	unsigned char* d_out;
	cudaMalloc((void**) &d_in, elems);
	cudaMalloc((void**) &d_out, rows*cols);
	
	printf("rows_kernel: %d \n", rows);
	printf("cols_kernel: %d \n", cols);
	cudaMemcpy(d_in, h_in, elems*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	if(checkgrid2D==1)
	{
        	kernel1<<<grid,block>>>(d_in, d_out, rows, cols);
		printf("use 2D grid 1D block\n");        
	}
	else
	{        
		kernel2<<<rows,cols>>>(d_in, d_out, rows, cols);
		printf("use 1D grid 1D block\n");      
	}

	cudaMemcpy(h_out, d_out, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);

}




























