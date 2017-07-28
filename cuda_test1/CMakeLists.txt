#cmake_minimum_required(VERSION 2.8.4)
#project( test2 )
#set(CMAKE_BUILD_TYPE Release)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#include_directories(/usr/local/include) 
#include_directories("/usr/local/cuda/include")
#include_directories("/usr/local/cuda-8.0/include")
#include_directories(${CMAKE_BINARY_DIR})
#include_directories(/usr/include) 
#include_directories(/usr/include/x86_64-linux-gnu) 

cmake_minimum_required(VERSION 2.8.4)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lpthread")
project( main )
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

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
    --std=c++11
    )

cuda_add_library(test1_cu_obj test1_cu.cu)
cuda_add_executable(main test1.cpp test1_cu.cu)
#target_link_libraries(main ${CUDA_LIBRARIES} ${OpenCV_LIBS} /usr/local/cuda-8.0/lib64/libcudart.so)
target_link_libraries(main test1_cu_obj)






























