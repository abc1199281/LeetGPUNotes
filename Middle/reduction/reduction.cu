#include <iostream>


__global__ void reduce(float* input, float* output, int n) {
    /*
    extern: Declares dynamically allocated shared memory, which is visible to all threads in the block.
     */
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, float* output, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, gridSize * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    /*
    blockSize: The number of threads in a block.
    gridSize: The number of blocks in a grid.
    blockSize * sizeof(float): The size of the shared memory.
    */
    reduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);

    float* h_output = new float[gridSize];
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        sum += h_output[i];
    }
    cudaMemcpy(output, &sum, sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    float input1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int n1 = sizeof(input1) / sizeof(float);
    float* d_output1, *output1 = new float;
    cudaMalloc(&d_output1, sizeof(float));
    solve(input1, d_output1, n1);
    cudaMemcpy(output1, d_output1, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Output 1: " << *output1 << std::endl;
    cudaFree(d_output1);
    delete [] output1;

    float input2[] = {-2.5, 1.5, -1.0, 2.0};
    int n2 = sizeof(input2) / sizeof(float);
    float* d_output2, *output2 = new float;
    cudaMalloc(&d_output2, sizeof(float));
    solve(input2, d_output2, n2);
    std::cout << "ee" << std::endl;
    cudaMemcpy(output2, d_output2, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Output 2: " << *output2 << std::endl;
    cudaFree(d_output2);
    delete [] output2;

    return 0;
}
