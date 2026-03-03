#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------
// THE KERNEL (Runs on the Apple GPU)
// ---------------------------------------------------------
kernel void dense_layer(const device float* X [[buffer(0)]],
                        const device float* W [[buffer(1)]],
                        const device float* B [[buffer(2)]],
                        device float* Y [[buffer(3)]],
                        constant uint& M [[buffer(4)]], // Batch Size
                        constant uint& K [[buffer(5)]], // Input Features
                        constant uint& N [[buffer(6)]], // Output Features
                        uint2 gid [[thread_position_in_grid]]) 
{
    uint row = gid.y; // Current batch index
    uint col = gid.x; // Current output feature index

    // Boundary check
    if (row < M && col < N) {
        // Initialize with the bias for this specific output neuron
        float sum = B[col]; 
        
        // Compute the dot product of the input row and weight column
        for (uint i = 0; i < K; ++i) {
            sum += X[row * K + i] * W[i * N + col];
        }
        
        // Write the final computed value to the output tensor
        Y[row * N + col] = sum;
    }
}