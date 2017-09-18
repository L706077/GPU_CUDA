#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>  //srand()
//#include <stdbool.h>
//#define block 514


extern "C" void smooth_global_outer(float* b, float* a, int n, int loop, int BLOCK);
extern "C" void smooth_shared_outer(float* b, float* a, int n, int loop, int BLOCK);
//(2) 裝置核心(global 版).
__global__ void smooth_global(float* b, float* a, int n){
        int k = blockIdx.x*blockDim.x+threadIdx.x;
        if(k==0){
                b[k]=(2*a[0]+a[1])*0.25;
        }
        else if(k==n-1){
                b[k]=(a[n-2]+2*a[n-1])*0.25;
        }
        else if(k<n){
                b[k]=(a[k-1]+2*a[k]+a[k+1])*0.25;
        }
}
//(3) 裝置核心(shared 版).
__global__ void smooth_shared(float* b, float* a, int n,  int BLOCK){
        int base = blockIdx.x*blockDim.x;
        int t = threadIdx.x;

        //__shared__ float s[BLOCK+2];//宣告共享記憶體.
	extern __shared__ float s[];//宣告共享記憶體.
   		 //載入主要資料 s[1]~s[BLOCK]
    		 // s[0] <-- a[base-1]  (左邊界)
    		 // s[1] <-- a[base]
    		 // s[2] <-- a[base+1]
    		 // s[3] <-- a[base+2]
    		 //      ...
    		 // s[BLOCK]   <-- a[base+BLOCK-1]
    		 // s[BLOCK+1] <-- a[base+BLOCK]  (右邊界)
        if(base+t<n){
                s[t+1]=a[base+t];
        }
        if(t==0){
                //左邊界.
                if(base==0){
                        s[0]=0;
                }
                else{
                        s[0]=a[base-1];  //載入邊界資料 s[0] & s[BLOCK+1] (只用兩個執行緒處理) 
                }
        }       
        if(t==32){                       //*** 使用獨立的 warp 讓 branch 更快 ***
                if(base+BLOCK>=n){       //右邊界.
                        s[n-base+1]=0;
                }
                else{
                        s[BLOCK+1] = a[base+BLOCK];
                }
        }
        __syncthreads();                                   //同步化 (確保共享記憶體已寫入)
        if(base+t<n){
                b[base+t]=(s[t]+2*s[t+1]+s[t+2])*0.25;     //輸出三點加權平均值
        }
};

extern "C" void smooth_global_outer(float* b, float* a, int n, int loop, int BLOCK)
{

        dim3 block(BLOCK, 1, 1);
	dim3 grid(n/BLOCK+1, 1, 1);
        for(int k=0; k<loop; k++)
	{
                smooth_global<<< grid, block >>>(b, a, n);
        }	


}

extern "C" void smooth_shared_outer(float* b, float* a, int n, int loop, int BLOCK)
{

        dim3 block(BLOCK, 1, 1);
	dim3 grid(n/BLOCK+1, 1, 1);
	
	for(int k=0; k<loop; k++){
                smooth_shared<<< grid, block, BLOCK+2 >>>(b, a, n, BLOCK);
        }


}


























