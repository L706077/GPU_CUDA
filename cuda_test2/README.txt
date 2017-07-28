#method1:
$ g++ -c test2.cpp
$ nvcc -c test2_cu.cu
$ nvcc -o main test2.o test2_cu.o


#method2:
$ cd build
$ cmake ..
$ sudo make
