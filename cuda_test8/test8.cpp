#include <stdio.h>
#include <time.h>
#include <stdlib.h>  //srand()
#include <iostream>  //cout
#include <string.h>  //memset()
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;


extern "C" void MeanFilterCUDA(unsigned char* h_in, unsigned char* h_out, int nKernelSize, int rows, int cols);


int main( int argc, char** argv )
{	
        clock_t t1, t2, t3, t4, t5, t6;
        Mat image0, image, gray;
        image0 = imread("test1.jpg");	
        imshow("test1.jpg",image0);

        cvtColor(image0, gray, CV_BGR2GRAY);
        imshow("gray_origin",gray);
	image=gray.clone();
		printf("channel: %d \n", image.channels());

//         Now trying GPU code
t3=clock(); 
	const int rows = image.rows; //height
		printf("rows: %d \n", rows);
	const int cols = image.cols; //width
		printf("cols: %d \n", cols);
	unsigned char *h_in = image.data;
	unsigned char *h_out = new unsigned char[rows*cols];

        MeanFilterCUDA(h_in, h_out, 9, rows, cols);

t4=clock();	

	Mat gray2 = Mat(rows, cols, CV_8UC1, h_out);
        	printf("gpu \t\telapsed %f sec\n", (double)(t4-t3)/(CLOCKS_PER_SEC));

	imshow("gpu_grey",gray2);
	waitKey(0);
	
	cudaDeviceReset();
	return 0;
}









