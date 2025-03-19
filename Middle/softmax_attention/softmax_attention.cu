#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ void row_wise_softmax(float* row, int N) {
    float max_val = row[0];
    for (int i = 1; i < N; ++i) {
        if (row[i] > max_val) {
            max_val = row[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < N; ++i) {
        row[i] = expf(row[i] - max_val);
        sum_exp += row[i];
    }

    for (int i = 0; i < N; ++i) {
        row[i] /= sum_exp;
    }
}


__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < d) {
        float* temp_row = new float[N];

        // Compute QK^T
        for (int i = 0; i < N; ++i) {
            temp_row[i] = 0.0f;
            for (int j = 0; j < d; ++j) {
                temp_row[i] += Q[row * d + j] * K[i * d + j];
            }
            temp_row[i] /= sqrtf(static_cast<float>(d));
        }

        // Apply softmax
        row_wise_softmax(temp_row, N);

        // Compute weighted sum with V
        float final_val = 0.0;
        for(int i = 0; i < N; i++){
          final_val += temp_row[i] * V[i*d + col];
        }
        output[row * d + col] = final_val;
        delete[] temp_row;

    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *d_Q, *d_K, *d_V, *d_output;

    // Allocate device memory
    cudaMalloc(&d_Q, M * d * sizeof(float));
    cudaMalloc(&d_K, N * d * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_output, M * d * sizeof(float));

    // Copy input matrices to device
    cudaMemcpy(d_Q, Q, M * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockSize(32, 32);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (d + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    attention_kernel<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_output, M, N, d);

    // Copy result back to host
    cudaMemcpy(output, d_output, M * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();
}


int main() {
    // Example 1
    int M = 2, N = 3, d = 4;
    float Q[2 * 4] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    float K[3 * 4] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    float V[3 * 4] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float output[2 * 4];

    solve(Q, K, V, output, M, N, d);

    printf("Output Example 1:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            printf("%.2f ", output[i * d + j]);
        }
        printf("\n");
    }

    // Example 2
    M = 1, N = 2, d = 2;
    float Q2[1 * 2] = {1.0, 2.0};
    float K2[2 * 2] = {1.0, 0.0, 0.0, 1.0};
    float V2[2 * 2] = {3.0, 4.0, 5.0, 6.0};
    float output2[1 * 2];

    solve(Q2, K2, V2, output2, M, N, d);

    printf("Output Example 2:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < d; ++j) {
            printf("%.2f ", output2[i * d + j]);
        }
        printf("\n");
    }

     // Example 3 (Larger matrices)
    M = 100;
    N = 100;
    d = 64;
    float* Q3 = new float[M * d];
    float* K3 = new float[N * d];
    float* V3 = new float[N * d];
    float* output3 = new float[M * d];

    // Initialize with some values (for demonstration)
    for (int i = 0; i < M * d; ++i) {
        Q3[i] = (i % 10) * 0.1f; // Values between 0.0 and 0.9
    }
    for (int i = 0; i < N * d; ++i) {
        K3[i] = (i % 5) * 0.2f;  // Values between 0.0 and 0.8
        V3[i] = (i % 8) * 0.125f; // Values between 0.0 and 0.875
    }

    solve(Q3, K3, V3, output3, M, N, d);

    printf("Output Example 3 (first 5 rows):\n");
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < d; ++j) {
            printf("%.2f ", output3[i * d + j]);
        }
        printf("\n");
    }

    delete[] Q3;
    delete[] K3;
    delete[] V3;
    delete[] output3;

    return 0;
}