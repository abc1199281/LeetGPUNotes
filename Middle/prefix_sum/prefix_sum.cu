#include <iostream>

/*
// Complexity: O(n^2)

__global__ void prefixSumKernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) {
            sum += input[j];
        }
        output[i] = sum;
    }
}
*/

/*
- Prefix sum reduction algorithm
- Complexity: O(1) in kernel
    - O(log(256)) -> O(1)
    - Shared memory
    - Parallel reduction
*/

__global__ void prefixSumKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // O(log(256)) -> O(1)
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (tid >= offset) {
            sdata[tid] += sdata[tid - offset];
        }
        __syncthreads();
    }

    if (i < n) {
        output[i] = sdata[tid];
    }
}

void solve(const float* input, float* output, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    prefixSumKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(input, output, n);
    cudaDeviceSynchronize();
}

int main() {
    // Example usage
    float input1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int n1 = sizeof(input1) / sizeof(input1[0]);
    float* d_input1;
    float* d_output1;

    cudaMalloc(&d_input1, n1 * sizeof(float));
    cudaMalloc(&d_output1, n1 * sizeof(float));

    cudaMemcpy(d_input1, input1, n1 * sizeof(input1[0]), cudaMemcpyHostToDevice);

    solve(d_input1, d_output1, n1);

    float output1[n1];
    cudaMemcpy(output1, d_output1, n1 * sizeof(input1[0]), cudaMemcpyDeviceToHost);

    std::cout << "Input: [";
    for(int i = 0; i < n1; ++i)
    {
        std::cout << input1[i];
        if(i < n1 -1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Output: [";
    for (int i = 0; i < n1; ++i) {
        std::cout << output1[i];
        if (i < n1 - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    float input2[] = {5.0f, -2.0f, 3.0f, 1.0f, -4.0f};
    int n2 = sizeof(input2) / sizeof(input2[0]);
    float* d_input2;
    float* d_output2;

    cudaMalloc(&d_input2, n2 * sizeof(float));
    cudaMalloc(&d_output2, n2 * sizeof(float));

    cudaMemcpy(d_input2, input2, n2 * sizeof(input2[0]), cudaMemcpyHostToDevice);

    solve(d_input2, d_output2, n2);

    float output2[n2];
    cudaMemcpy(output2, d_output2, n2 * sizeof(input2[0]), cudaMemcpyDeviceToHost);

    std::cout << "Input: [";
    for(int i = 0; i < n2; ++i)
    {
        std::cout << input2[i];
        if(i < n2 -1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Output: [";
    for (int i = 0; i < n2; ++i) {
        std::cout << output2[i];
        if (i < n2 - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaFree(d_input2);
    cudaFree(d_output2);

    return 0;
}