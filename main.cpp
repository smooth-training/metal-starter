#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "metalcpp/Foundation/Foundation.hpp"
#include "metalcpp/Metal/Metal.hpp"
#include <iostream>

int main() {
    // 1. Memory Management (Required for Metal-cpp)
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    // 2. Get the Default Apple Silicon GPU
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported." << std::endl;
        return -1;
    }

    // 3. Define dimensions for the Dense Layer
    uint32_t M = 2;   // Batch Size
    uint32_t K = 4;   // Input Features
    uint32_t N = 3;   // Output Features (Neurons)

    // 4. Create Unified Memory Buffers
    MTL::Buffer* bufX = device->newBuffer(M * K * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufW = device->newBuffer(K * N * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufB = device->newBuffer(N * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufY = device->newBuffer(M * N * sizeof(float), MTL::ResourceStorageModeShared);

    // Initialize dummy data directly using C++ pointers
    float* X = (float*)bufX->contents();
    float* W = (float*)bufW->contents();
    float* B = (float*)bufB->contents();
    
    for (int i = 0; i < M * K; ++i) X[i] = 1.0f; // Inputs
    for (int i = 0; i < K * N; ++i) W[i] = 0.5f; // Weights
    for (int i = 0; i < N; ++i) B[i] = 0.1f;     // Biases

    // 5. Load the Compiled Metal Library (.metallib)
    NS::Error* error = nullptr;
    NS::String* libraryPath = NS::String::string("./dense.metallib", NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libraryPath, &error);
    if (!library) {
        std::cerr << "Failed to load Metal library." << std::endl;
        return -1;
    }

    // 6. Set up the Pipeline State
    NS::String* functionName = NS::String::string("dense_layer", NS::UTF8StringEncoding);
    MTL::Function* denseFunction = library->newFunction(functionName);
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(denseFunction, &error);

    // 7. Create Command Queue and Buffers
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

    // 8. Bind Data to the Kernel
    computeEncoder->setComputePipelineState(pipelineState);
    computeEncoder->setBuffer(bufX, 0, 0);
    computeEncoder->setBuffer(bufW, 0, 1);
    computeEncoder->setBuffer(bufB, 0, 2);
    computeEncoder->setBuffer(bufY, 0, 3);
    computeEncoder->setBytes(&M, sizeof(uint32_t), 4);
    computeEncoder->setBytes(&K, sizeof(uint32_t), 5);
    computeEncoder->setBytes(&N, sizeof(uint32_t), 6);

    // 9. Dispatch Threads
    MTL::Size gridSize = MTL::Size::Make(N, M, 1); // Output features (X), Batch size (Y)
    
    // Calculate threadgroup size (typically a multiple of thread execution width)
    NS::UInteger threadGroupSizeX = pipelineState->maxTotalThreadsPerThreadgroup();
    if (threadGroupSizeX > N) threadGroupSizeX = N;
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSizeX, 1, 1);

    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    // 10. Execute and Wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // 11. Read Results
    float* Y = (float*)bufY->contents();
    std::cout << "Output Y[0][0]: " << Y[0] << std::endl;

    // 12. Clean Up
    bufX->release(); bufW->release(); bufB->release(); bufY->release();
    denseFunction->release(); library->release(); pipelineState->release();
    commandQueue->release(); device->release(); pool->release();

    return 0;
}