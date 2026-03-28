%%writefile matrix_mul.cu

#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------
// 1. THE KERNEL (Runs on the GPU)
// ---------------------------------------------------------
// The __global__ identifier tells the compiler this function
// runs on the GPU (device) but is called from the CPU (host).
__global__ void matrixMulKernel(float *A, float *B, float *C, int N) {
    // Calculate the global row and column index for this specific thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure the thread maps to a valid matrix element
    if (row < N && col < N) {
        float sum = 0.0f;

        // Compute the dot product for this row of A and column of B
        for (int k = 0; k < N; ++k) {
            // Arrays are 1D in memory, so 2D indices are flattened: (row * width + col)
            sum += A[row * N + k] * B[k * N + col];
        }

        // Write the final computed value to the resulting matrix C in VRAM
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------
// 2. THE HOST CODE (Runs on the CPU)
// ---------------------------------------------------------
int main() {
    int N = 25600; // Matrix size (256 x 256)
    size_t size = N * N * sizeof(float); // Total bytes per matrix

    // Allocate memory for matrices on the Host (CPU System RAM)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize matrices with some dummy data
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Pointers for the Device (GPU VRAM)
    float *d_A, *d_B, *d_C;

    // Allocate memory on the Device (GPU VRAM)
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from Host (System RAM) to Device (VRAM) across the PCIe Bus
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define the Execution Configuration (How to group the GPU threads)
    // We create 16x16 blocks of threads.
    dim3 threadsPerBlock(16, 16);

    // Calculate how many blocks we need to cover the entire N x N matrix
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the Kernel on the GPU
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for the GPU to finish before the CPU continues
    cudaDeviceSynchronize();

    // Copy the computed result from Device (VRAM) back to Host (System RAM)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify a single element just to check if it worked (1.0 * 2.0 * 256 = 512.0)
    printf("Result at C[0][0] is: %f\n", h_C[0]);

    // Clean up: Free Device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Clean up: Free Host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}