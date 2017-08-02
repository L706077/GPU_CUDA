#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

//索引用到的緒構體
struct Index{
        int block, thread;
};
extern "C" void kernel_wrapper(Index *aa, Index *bb);
void OutputSpec( const cudaDeviceProp sDevProp );

const uint3 blockIdx;
const uint3 threadIdx;
const dim3 blockDim;
//核心:把索引寫入裝置記憶體
__global__ void prob_idx(Index id[]){
        int b=blockIdx.x;       //區塊索引
        int t=threadIdx.x;      //執行緒索引
        int n=blockDim.x;       //區塊中包含的執行緒數目
        int x=b*n+t;            //執行緒在陣列中對應的位置

        //每個執行緒寫入自己的區塊和執行緒索引.
        id[x].block=b;
        id[x].thread=t;
};

//主函式
void kernel_wrapper(Index *aa,Index *bb)
{
	int  iDeviceCount = 0;
	cudaGetDeviceCount( &iDeviceCount );
        printf( "Number of GPU: %d\n", iDeviceCount );

	  for( int i = 0; i < iDeviceCount; ++ i )
	  {
    		printf( "=== Device %i ===", i );
    		cudaDeviceProp  sDeviceProp;
    		cudaGetDeviceProperties( &sDeviceProp, i );
    		OutputSpec( sDeviceProp );
  	}




        Index* d=aa;
        Index* h=bb;

        //配置裝置記憶體
        cudaMalloc((void**) &d, 100*sizeof(Index));

        //呼叫裝置核心
        int g=3, b=4, m=g*b;
        prob_idx <<<g,b>>>(d);

        //下載裝置記憶體內容到主機上
        cudaMemcpy(h, d, 100*sizeof(Index), cudaMemcpyDeviceToHost);

        //顯示內容
        for(int i=0; i<m; i++){
            printf("h[%d]={block:%d, thread:%d}\n", i,h[i].block,h[i].thread);
        }

        //釋放裝置記憶體
        cudaFree(d);
 }


void OutputSpec( const cudaDeviceProp sDevProp )
{
  printf( "Device name: %s \n", sDevProp.name );
  printf( "Device memory: %d \n", sDevProp.totalGlobalMem );
  printf( "shared Memory per-block: %d \n", sDevProp.sharedMemPerBlock );
  printf( "Register mMemory per-block: %d \n", sDevProp.regsPerBlock );
  printf( "Warp size: %d \n", sDevProp.warpSize );
  printf( "Memory pitch: %d \n", sDevProp.memPitch );
  printf( "Constant Memory: %d \n", sDevProp.totalConstMem );
  printf( "Max thread per-block: %d \n", sDevProp.maxThreadsPerBlock );
  printf( "Max Blocks per-grid: %d \n", sDevProp.maxBlocksPerGrid );
  printf( "Max thread dim: ( %d, %d, %d ) \n", sDevProp.maxThreadsDim[0], sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2] );
  printf( "Max grid size: ( %d, %d, %d ) \n", sDevProp.maxGridSize[0], sDevProp.maxGridSize[1], sDevProp.maxGridSize[2] );
  printf( "Ver: %d.%d \n", sDevProp.major, sDevProp.minor );
  printf( "Clock: %d \n", sDevProp.clockRate );
  printf( "textureAlignment: %d \n", sDevProp.textureAlignment );
}





