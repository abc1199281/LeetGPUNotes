# Vector Addition

## Thought
- CPU code first
- Understand the GPU 1D dimension

## CPU Code
~~~c++
// A, B, C are host pointers (i.e. pointers to memory on the CPU)
void solve(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) { // replaced
        C[i] = B[i] + A[i];
    }
}
~~~

## GPU Code
~~~c++
#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N)  // Necessary
        C[id] = B[id] + A[id];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    // Round up
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
~~~