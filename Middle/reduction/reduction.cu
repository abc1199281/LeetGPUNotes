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

void solve(float* input, int n, float& output) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, gridSize * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    /*
    blockSize: The number of threads in a block.
    gridSize: The number of blocks in a grid.
    blockSize * sizeof(float): The size of the shared memory.
    */
    reduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);

    float* h_output = new float[gridSize];
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        sum += h_output[i];
    }

    output = sum;

    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    float input1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    int n1 = sizeof(input1) / sizeof(float);
    float output1;
    solve(input1, n1, output1);
    std::cout << "Output 1: " << output1 << std::endl;

    float input2[] = {-2.5, 1.5, -1.0, 2.0};
    int n2 = sizeof(input2) / sizeof(float);
    float output2;
    solve(input2, n2, output2);
    std::cout << "Output 2: " << output2 << std::endl;

    return 0;
}