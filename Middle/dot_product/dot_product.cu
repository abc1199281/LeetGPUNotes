#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel to compute the histogram
__global__ void dot_kernel(const float* A, const float* B, float* C, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("idx: %d\n", idx);

    // Load elements into shared memory
    float sum = 0;
    if (idx < N) {
        sum = A[idx] * B[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to the output
    if (tid == 0) {
        C[blockIdx.x] = sdata[0];
    }

}

// input, histogram are device pointers
void solve(const float* A, const float* B, float* result, int N) {

    // Configure the grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* d_result; // partial results
    cudaMalloc((void**)&d_result, blocksPerGrid * sizeof(float));

    // Launch the CUDA kernel
    dot_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(A, B, d_result, N);


    // Copy the result back to the host
    std::vector<float> partialResults(blocksPerGrid);
    cudaMemcpy(partialResults.data(), d_result, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform final reduction on the host
    float host_result = 0;
    for (float val : partialResults) {
        host_result += val;
    }
    cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(d_result);
}
void solve_host(const float* A, const float* B, float* result, int N) {

    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_result;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    // Copy input data to the device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    solve(d_A, d_B, d_result, N);

    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}

int main() {
    // Example usage 1
    std::vector<float> A1 = {1, 2, 3, 4};
    std::vector<float> B1 = {5, 6, 7, 8};
    float result1;
    int N1 = A1.size();

    solve_host(A1.data(), B1.data(), &result1, N1);

    std::cout << "Result 1: " << result1 << std::endl;
    return 0;
}