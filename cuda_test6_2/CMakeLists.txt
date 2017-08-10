cmake_minimum_required(VERSION 2.8.4)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lpthread")
#cmake_minimum_required(VERSION 3.2)
project( main )

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)
#find_package(OpenCV REQUIRED)
# For compilation ...
# Specify target & source files to compile it from

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_30,code=sm_30
	-gencode arch=compute_35,code=sm_35
	-gencode arch=compute_50,code=sm_50
	-gencode arch=compute_52,code=sm_52
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_61,code=sm_61
    #--std=c++11
    )

cuda_add_library(test6_cu_obj test6_cu.cu)
cuda_add_executable(main test6_2D_1D.cpp test6_cu.cu)
#target_link_libraries(main ${CUDA_LIBRARIES} ${OpenCV_LIBS} /usr/local/cuda-8.0/lib64/libcudart.so)
target_link_libraries(main test6_cu_obj)





















