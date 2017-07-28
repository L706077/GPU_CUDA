#include <stdio.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <cuda_runtime.h>

#include <stdio.h>
//#include <test2.cu>


extern "C" void add_vector_gpu( float* a, float* b, float *c, int size );

void add_vector_cpu( float* a, float* b, float *c, int size )
{
	for( int i = 0; i < size; ++ i )
		c[i] = a[i] + b[i];
}

int main( int argc, char** argv) 
{
	// initial data
	int	data_size = 100;
	float	*dataA = new float[data_size],
			*dataB = new float[data_size],
			*dataC = new float[data_size],
			*dataD = new float[data_size];

	for( int i = 0; i < data_size; ++ i )
	{
		dataA[i] = i;
		dataB[i] = -1 * i;
	}

	// run CPU program
	add_vector_cpu( dataA, dataB, dataC, data_size );
printf( "test111111\n");
	// run GPU program
	add_vector_gpu( dataA, dataB, dataD, data_size );
printf( "test222222\n");
	// compare the result
	for( int i = 0; i < data_size; ++ i )
	{
		if( dataC[i] != dataD[i] )
			printf( "Error!! (%f & %f)\n", dataC[i], dataD[i] );
	}
printf( "over\n");
}


