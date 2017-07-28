#method1:
$ g++ -c test1.cpp
$ nvcc -c test1_cu.cu
$ nvcc -o main test1.o test1_cu.o


#method2:
$ cd build
$ cmake ..
$ sudo make
