#include <stdio.h>
#include <time.h>
#include <stdlib.h>  //srand()
#include <iostream>  //cout
#include <string.h>  //memset()
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

extern "C" void gray_parallel(unsigned char* h_in, unsigned char* h_out, int elems, int rows, int cols);

// My serial implementation, works fine
Mat gray_serial(Mat img){
  int rows = img.rows;
	int cols = img.cols;
	Mat gray(rows, cols, CV_8UC1);
	for(int r=0; r<rows; r++){
		for(int c=0; c<cols; c++){
			Vec3b bgr = img.at<Vec3b>(r,c);
			double gray_val = 0.144*bgr.val[0] + 0.587*bgr.val[1] + 0.299*bgr.val[2];
			gray.at<unsigned char>(r,c) = (unsigned char)gray_val;
		}
	}
	return gray;
}

// One more serial code, just for testing, works fine
Mat gray_test(Mat img){
	cout << "running test" << endl;
	uint rows = img.rows;
	uint cols = img.cols;
	unsigned char* test = img.data;
	unsigned char* op = new unsigned char[rows*cols];
	for (uint i=0; i<rows*cols; i++){
		uint index = 3*i;
		double temp = 0.144*test[index]+0.587*test[index+1]+0.299*test[index+2];
		op[i] = (unsigned char)temp;
	}
	Mat gray = Mat(rows, cols, CV_8UC1, op);
	return gray;
}


int main( int argc, char** argv )
{	
       clock_t t1, t2, t3, t4, t5, t6;
       Mat image;
       image = imread("test1.jpg");	
       imshow("test1.jpg",image);
t1=clock();
	Mat gray = gray_test(image);
	imshow("cpu_grey",gray);
t2=clock();
        printf("cpu \t\telapsed %f sec\n", (double)(t2-t1)/(CLOCKS_PER_SEC));


//         Now trying GPU code
t3=clock(); 
	const int rows = image.rows; //height
	printf("rows: %d \n", rows);
	const int cols = image.cols; //width
	printf("cols: %d \n", cols);
	int elems = rows*cols*3;
	unsigned char *h_in = image.data;
	unsigned char *h_out = new unsigned char[rows*cols];
//t3=clock();
	std::cout<<"h_in"<< &h_in <<std::endl;    
	gray_parallel(h_in, h_out, elems, rows, cols);
t4=clock();	

	Mat gray2 = Mat(rows, cols, CV_8UC1, h_out);
        printf("gpu \t\telapsed %f sec\n", (double)(t4-t3)/(CLOCKS_PER_SEC));

	imshow("gpu_grey",gray2);
	waitKey(0);
	
	cudaDeviceReset();
	return 0;
}









