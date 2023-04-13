## Introduction

This repository contains the project developed for the GPU101 PiA course 2023 @ Politecnico di Milano.
The assignment consisted in the implementation of the Symmetric Gauss Seidel Algorithm on a CUDA capable device. The original algorithm (performed on the CPU) can be found inside "symgs/symgs-csr.c". It's also used in the CUDA implementation to test the accuracy of my solution.
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
The matrix stored in the .mtx file MUST have non-zero values on the main diagonal. (The matrix used for the testing can be downloaded [here](https://www.dropbox.com/s/jzn573j0z9ffl7h/kmer_V4a.mtx?dl=0))

The threads_per_block value has to be an integer (between 32 and 1024) and it's used for testing performance. Using 128 threads per block was found to achieve the best performance on my system (Nvidia GTX 1660 SUPER).
