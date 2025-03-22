#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel to compute the histogram
__global__ void histogram_kernel(int* input, int N, int num_bins, int* histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int value = input[idx];
        if (value >= 0 && value < num_bins) {
            atomicAdd(&histogram[value], 1);
        }
    }
}
// Shared memory version of the histogram kernel
// Reducing Atomic Operations
__global__ void histogram_kernel_sm(int* input, int N, int num_bins, int* histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int local_histogram[1024]; // max bins <= 1024
    if(threadIdx.x < num_bins){
        local_histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    if (idx < N) {
        int value = input[idx];
        if (value >= 0 && value < num_bins) {
            atomicAdd(&(local_histogram[value]),1);
        }
    }
    __syncthreads();

    if(idx < num_bins){
        atomicAdd(&(histogram[idx]), local_histogram[idx]);
    }
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    // Allocate memory on the device
    int* d_input;
    int* d_histogram;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_histogram, num_bins * sizeof(int));

    // Copy input data to the device
    cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize the histogram on the device to zero
    cudaMemset(d_histogram, 0, num_bins * sizeof(int));

    // Configure the grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    // histogram_kernel<<<numBlocks, threadsPerBlock>>>(d_input, N, num_bins, d_histogram);
    histogram_kernel_sm<<<numBlocks, threadsPerBlock>>>(d_input, N, num_bins, d_histogram);

    // Copy the result back to the host
    cudaMemcpy(histogram, d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_histogram);
}

int main() {
    // Example usage 1
    std::vector<int> input1 = {0, 1, 2, 1, 0};
    int N1 = input1.size();
    int num_bins1 = 3;
    std::vector<int> histogram1(num_bins1, 0);

    solve(input1.data(), histogram1.data(), N1, num_bins1);

    std::cout << "Histogram 1: ";
    for (int count : histogram1) {
        std::cout << count << " ";
    }
    std::cout << std::endl;

    // Example usage 2
    std::vector<int> input2 = {3, 3, 3, 3};
    int N2 = input2.size();
    int num_bins2 = 5;
    std::vector<int> histogram2(num_bins2, 0);

    solve(input2.data(), histogram2.data(), N2, num_bins2);

    std::cout << "Histogram 2: ";
    for (int count : histogram2) {
        std::cout << count << " ";
    }
    std::cout << std::endl;

    return 0;
}