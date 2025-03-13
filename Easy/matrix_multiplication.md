# Matrix multiplication

## Problem
- A: M x N
- B: N x K
- C = A x B (M x K)

## Thought
- CPU code first
- Understand the GPU 2D dimension

## CPU Code

~~~c++
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) { // replaced by kernel
        for (int col = 0; col < K; ++col) { // replaced by kernel
            float Pvalue = 0.0f;
            for (int k = 0; k < N; ++k) {
                Pvalue += A[row * N + k] * B[k * K + col];
            }
            C[row * K + col] = Pvalue;
        }
    }
}
~~~

## GPU Code

~~~c++
#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < K){ // protect the ranges
        float Pvalue = 0.0f;

        for(int k = 0;k<N;k++){
            Pvalue += A[row * N + k] * B[k * K + col];
        }

        C[row * K + col] = Pvalue;
    }

}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    // Remember the concept of threadsPerBlock and blocksPerGrid
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
~~~