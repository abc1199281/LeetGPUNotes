# Reverse Array

## Problem
- Implement a CUDA program that reverses an array of 32-bit floating point numbers in-place.

## GPU Code
~~~c++
#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N/2) return;
    float tmp = input[idx];
    input[idx] = input[N - idx -1];
    input[N - idx -1] = tmp;
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N/2+ threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
~~~