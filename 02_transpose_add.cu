// =============================================================================
// PRACTICE 1 - Transpose-add with a 2D grid          (warm-up, ~40 min)
//
// TASK:
//   The CUDA with C++ program computes C = A + B^T for two NxN integer
//   matrices in parallel. N = 1 << 10. Following functions are expected:
//     - kernel to fill the matrices on independent streams;
//     - kernel to compute C = A + B^T using a 2D grid of 16x16 thread blocks;
//     - host function computing the same result;
//     - host function to verify the correctness of the result;
//     - main function.
//   The total number of blocks must equal warp size * number of SMs
//   (hint: dim3 grid(warpSize, numSMs)). Use a 2D grid-stride loop.
//   Measure running time of the kernel and the host version.
//
// WHAT IT PRACTISES: 2D indexing with dim3, 2D grid-stride loops, streams,
// managed memory + prefetch, timing, cleanup.
// Build: nvcc -O2 -o tadd 02_transpose_add.cu
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define N (1 << 10)
#define TILE 16

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

void prefetch(const void *ptr, size_t bytes, int device, cudaStream_t stream = 0)
{
    static int supported = -1;
    if (supported < 0) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrConcurrentManagedAccess, dev));
    }
    if (!supported) return;
#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = (device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    loc.id   = (device == cudaCpuDeviceId) ? 0 : device;
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0, stream));
#else
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device, stream));
#endif
}

// 2D grid-stride fill: rows advance by gridDim.y*blockDim.y, cols by gridDim.x*blockDim.x
__global__ void fillMatrix(int *m, int n, int seed)
{
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n; row += gridDim.y * blockDim.y)
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x)
            m[row * n + col] = (row * n + col) * seed % 100;
}

__global__ void transposeAdd(const int *a, const int *b, int *c, int n)
{
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < n; row += gridDim.y * blockDim.y)
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x)
            c[row * n + col] = a[row * n + col] + b[col * n + row];   // B^T[row][col] = B[col][row]
}

void transposeAddHost(const int *a, const int *b, int *c, int n)
{
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            c[row * n + col] = a[row * n + col] + b[col * n + row];
}

bool verifyResult(const int *gpu, const int *cpu, int n)
{
    for (int i = 0; i < n * n; ++i)
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at (%d, %d): %d vs %d\n", i / n, i % n, gpu[i], cpu[i]);
            return false;
        }
    return true;
}

int main()
{
    int deviceId, numSMs, warpSize;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,   cudaDevAttrMultiProcessorCount, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize,            deviceId));
    dim3 threads(TILE, TILE);                 // 256 threads = 8 warps per block
    dim3 blocks(warpSize, numSMs);            // warpSize * numSMs blocks in total
    printf("grid (%d x %d) blocks of (%d x %d) threads\n", blocks.x, blocks.y, threads.x, threads.y);

    size_t bytes = (size_t)N * N * sizeof(int);
    int *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    int *cHost = (int *)malloc(bytes);

    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreate(&sA));
    CUDA_CHECK(cudaStreamCreate(&sB));
    prefetch(a, bytes, deviceId, sA);
    prefetch(b, bytes, deviceId, sB);
    prefetch(c, bytes, deviceId);
    fillMatrix<<<blocks, threads, 0, sA>>>(a, N, 3);
    fillMatrix<<<blocks, threads, 0, sB>>>(b, N, 7);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(sA));
    CUDA_CHECK(cudaStreamSynchronize(sB));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    transposeAdd<<<blocks, threads>>>(a, b, c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernelMs;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));

    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    transposeAddHost(a, b, cHost, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double hostMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = verifyResult(c, cHost, N);
    printf("%s | kernel %.4f ms | host %.4f ms\n", ok ? "CORRECT" : "WRONG", kernelMs, hostMs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(sA));
    CUDA_CHECK(cudaStreamDestroy(sB));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    free(cHost);
    return ok ? 0 : 1;
}
