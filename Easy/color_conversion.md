
# Color Conversion

## Problem
- The image is represented as a 1D array of RGBA (Red, Green, Blue, Alpha) values, where each component is an 8-bit unsigned integer (unsigned char).

- Color inversion is performed by subtracting each color component (R, G, B) from 255. The Alpha component should remain unchanged.

## GPU Code

~~~c++
#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int layer = 4;
    if (idx > width * height*layer || idx % layer ==3 ) return;
    image[idx] = 255 - image[idx];
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int layer = 4;
    int blocksPerGrid = (width * height * layer + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
~~~