## Introduction

This repository contains the project developed for the GPU-101 course by PiA @ Politecnico di Milano.
The assignment was to implement the Symmetric Gauss Seidel Algorythm on a CUDA capable device. The original algorythm (performed on the cpu) can be found inside "symgs/symgs-csr.c". It's also used in the CUDA implementation to test the accuracy of my solution.
The paper describing the solution and the profiling data can be found inside the "report" folder.

## Usage

### Compilation

Compile inside the symgs folder using 
```
make
```
An NVIDIA CUDA COMPILER DRIVER and a CUDA capable GPU are needed (Check out [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit))

### Execution

Execute using
```
./symgs-gpu-test matrix_file threads_per_block
```
The matrix stored in the .mtx file for this algorythm MUST have non-zero values on the main diagonal. 

The threads_per_block value has to be an integer (between 32 and 1024) and it's used for testing performance. Using 128 threads per block was found to achieve the best performance on my system (Nvidia GTX 1660 SUPER).


