# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/test1_cu_obj.dir/test1_cu_obj_generated_test1_cu.cu.o: CMakeFiles/test1_cu_obj.dir/test1_cu_obj_generated_test1_cu.cu.o.depend
CMakeFiles/test1_cu_obj.dir/test1_cu_obj_generated_test1_cu.cu.o: CMakeFiles/test1_cu_obj.dir/test1_cu_obj_generated_test1_cu.cu.o.cmake
CMakeFiles/test1_cu_obj.dir/test1_cu_obj_generated_test1_cu.cu.o: ../test1_cu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/test1_cu_obj.dir/test1_cu_obj_generated_test1_cu.cu.o"
	cd /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/test1_cu_obj.dir && /usr/bin/cmake -E make_directory /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/test1_cu_obj.dir//.
	cd /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/test1_cu_obj.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/test1_cu_obj.dir//./test1_cu_obj_generated_test1_cu.cu.o -D generated_cubin_file:STRING=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/test1_cu_obj.dir//./test1_cu_obj_generated_test1_cu.cu.o.cubin.txt -P /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/test1_cu_obj.dir//test1_cu_obj_generated_test1_cu.cu.o.cmake

CMakeFiles/main.dir/main_generated_test1_cu.cu.o: CMakeFiles/main.dir/main_generated_test1_cu.cu.o.depend
CMakeFiles/main.dir/main_generated_test1_cu.cu.o: CMakeFiles/main.dir/main_generated_test1_cu.cu.o.cmake
CMakeFiles/main.dir/main_generated_test1_cu.cu.o: ../test1_cu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building NVCC (Device) object CMakeFiles/main.dir/main_generated_test1_cu.cu.o"
	cd /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir && /usr/bin/cmake -E make_directory /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir//.
	cd /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir//./main_generated_test1_cu.cu.o -D generated_cubin_file:STRING=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir//./main_generated_test1_cu.cu.o.cubin.txt -P /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir//main_generated_test1_cu.cu.o.cmake

CMakeFiles/main.dir/test1.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/test1.cpp.o: ../test1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/test1.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/test1.cpp.o -c /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/test1.cpp

CMakeFiles/main.dir/test1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/test1.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/test1.cpp > CMakeFiles/main.dir/test1.cpp.i

CMakeFiles/main.dir/test1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/test1.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/test1.cpp -o CMakeFiles/main.dir/test1.cpp.s

CMakeFiles/main.dir/test1.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/test1.cpp.o.requires

CMakeFiles/main.dir/test1.cpp.o.provides: CMakeFiles/main.dir/test1.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/test1.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/test1.cpp.o.provides

CMakeFiles/main.dir/test1.cpp.o.provides.build: CMakeFiles/main.dir/test1.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/test1.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS = \
"/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir/main_generated_test1_cu.cu.o"

main: CMakeFiles/main.dir/test1.cpp.o
main: CMakeFiles/main.dir/main_generated_test1_cu.cu.o
main: CMakeFiles/main.dir/build.make
main: /usr/local/cuda-8.0/lib64/libcudart_static.a
main: /usr/lib/x86_64-linux-gnu/librt.so
main: libtest1_cu_obj.a
main: /usr/local/cuda-8.0/lib64/libcudart_static.a
main: /usr/lib/x86_64-linux-gnu/librt.so
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/test1.cpp.o.requires

.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend: CMakeFiles/test1_cu_obj.dir/test1_cu_obj_generated_test1_cu.cu.o
CMakeFiles/main.dir/depend: CMakeFiles/main.dir/main_generated_test1_cu.cu.o
	cd /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1 /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1 /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build /home/ubuntu/Desktop/gitTUT/GPU_CUDA/cuda_test1/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

