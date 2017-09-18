#include <stdio.h>
#include <time.h>
#include <stdlib.h>  //srand()
#include <iostream>  //cout
#include <string.h>  //memset()
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
//#define BLOCK 512
int BLOCK=512;
using namespace std;
using namespace cv;

extern "C" void smooth_global_outer(float* b, float* a, int n, int loop, int BLOCK);
extern "C" void smooth_shared_outer(float* b, float* a, int n, int loop, int BLOCK);
//(1) 對照組 (host 版).
void smooth_host(float* b, float* a, int n){
        for(int k=1; k<n-1; k++){
                b[k]=(a[k-1]+2*a[k]+a[k+1])*0.25;
        }
        //邊界為0
        b[0]=(2*a[0]+a[1])*0.25;
        b[n-1]=(a[n-2]+2*a[n-1])*0.25;
}



int main( int argc, char** argv )
{	
       clock_t t1, t2, t3, t4, t5, t6;
//t1=clock();
//t2=clock();
//printf("cpu \t\telapsed %f sec\n", (double)(t2-t1)/(CLOCKS_PER_SEC));
//=================================

	int num=10*1000*1000;
        //int loop=130;  //測試迴圈次數 (量時間用)
        int loop=atoi(argv[1]);
	printf("loops = %d \n",loop);

	float* a=new float[num];
        float* b=new float[num];
        float* bg=new float[num];
        float* bs=new float[num];
        float *GA, *GB;
        cudaMalloc((void**) &GA, sizeof(float)*num);
        cudaMalloc((void**) &GB, sizeof(float)*num);

        for(int k=0; k<num; k++)
	{
                a[k]=(float)rand()/RAND_MAX;
                b[k]=bg[k]=bs[k]=0;
        }
        cudaMemcpy(GA, a, sizeof(float)*num, cudaMemcpyHostToDevice);


    //Test(1): smooth_host
    //--------------------------------------------------
        double t_host=(double)clock()/CLOCKS_PER_SEC;

        for(int k=0; k<loop; k++)
	{
                smooth_host(b,a,num);
        }

        t_host=((double)clock()/CLOCKS_PER_SEC-t_host)/loop;


    //Test(2): smooth_global (GRID*BLOCK 必需大於 num).
    //--------------------------------------------------
        double t_global=(double)clock()/CLOCKS_PER_SEC;
        
	cudaThreadSynchronize();

	smooth_global_outer(GB, GA, num, loop, BLOCK);

        cudaThreadSynchronize();

        t_global=((double)clock()/CLOCKS_PER_SEC-t_global)/loop;

        cudaMemcpy(bg, GB, sizeof(float)*num, cudaMemcpyDeviceToHost);


    //Test(3): smooth_shared (GRID*BLOCK 必需大於 num).
    //--------------------------------------------------
        double t_shared=(double)clock()/CLOCKS_PER_SEC;

        cudaThreadSynchronize();

	smooth_shared_outer(GB, GA, num, loop, BLOCK);
        
	cudaThreadSynchronize();

        t_shared=((double)clock()/CLOCKS_PER_SEC-t_shared)/loop;

        cudaMemcpy(bs, GB, sizeof(float)*num, cudaMemcpyDeviceToHost);


     //--------------------------------------------------
        double sum_dg2=0, sum_ds2=0, sum_b2=0;            //比較正確性
        for(int k=0; k<num; k++){
                double dg=bg[k]-b[k];
                double ds=bs[k]-b[k];

                sum_b2+=b[k]*b[k];
                sum_dg2+=dg*dg;
                sum_ds2+=ds*ds;
        }
    //--------------------------------------------------
        printf("vector size: %d \n", num);
        printf("\n");
        printf("Smooth_Host:   %g ms\n", t_host*1000);    //時間.
        printf("Smooth_Global: %g ms\n", t_global*1000);
        printf("Smooth_Shared: %g ms\n", t_shared*1000);
        printf("\n");
        printf("Diff(Smooth_Global): %g \n", sqrt(sum_dg2/sum_b2));
        printf("Diff(Smooth_Shared): %g \n", sqrt(sum_ds2/sum_b2));
        printf("\n");
        cudaFree(GA);                                     //釋放裝置記憶體.
        cudaFree(GB);
        delete [] a;
        delete [] b;
        delete [] bg;
        delete [] bs;
        return 0;



	cudaDeviceReset();
	return 0;
}


//======================









