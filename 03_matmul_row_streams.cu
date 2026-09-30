// =============================================================================
// PRACTICE 2 - Matrix multiplication split across streams        (~60 min)
//
// TASK:
//   The CUDA with C++ program multiplies two NxN integer matrices in parallel.
//   N = 1 << 10. The output matrix C is split into NUM_STREAMS = 4 horizontal
//   bands of rows; each band is computed by its own kernel launch on its own
//   stream, and each band is migrated back to the CPU on that same stream
//   right after it is computed. Following functions are expected:
//     - kernel to fill the matrices on independent streams;
//     - kernel that multiplies only rows [rowStart, rowStart + rowCount);
//     - host function to implement the multiplication;
//     - host function to verify the correctness of the result;
//     - main function.
//   Number of blocks = warp size * number of SMs (split evenly between the
//   streams is also acceptable). Use a grid-stride loop. Measure running time
//   of the GPU part and of the host multiplication.
//
// WHAT IT PRACTISES: several non-default streams running concurrently,
// per-stream prefetch (overlapping compute with migration), pointer offsets.
// Build: nvcc -O2 -o mmstreams 03_matmul_row_streams.cu
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define N (1 << 10)
#define THREADS 256
#define NUM_STREAMS 4

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

__global__ void fillMatrix(int *m, int n, int seed)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = index; i < n * n; i += stride)
        m[i] = (i * seed + 1) % 10;
}

// Computes only the band of rows [rowStart, rowStart + rowCount)
__global__ void matMulRows(const int *a, const int *b, int *c, int n, int rowStart, int rowCount)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int local = index; local < rowCount * n; local += stride) {
        int row = rowStart + local / n;
        int col = local % n;
        int sum = 0;
        for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
        c[row * n + col] = sum;
    }
}

void matMulHost(const int *a, const int *b, int *c, int n)
{
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            int sum = 0;
            for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
            c[row * n + col] = sum;
        }
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
    int blocks = warpSize * numSMs;
    int blocksPerStream = (blocks + NUM_STREAMS - 1) / NUM_STREAMS;

    size_t bytes = (size_t)N * N * sizeof(int);
    int *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    int *cHost = (int *)malloc(bytes);

    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Fill A and B on two independent streams
    prefetch(a, bytes, deviceId, streams[0]);
    prefetch(b, bytes, deviceId, streams[1]);
    prefetch(c, bytes, deviceId);
    fillMatrix<<<blocks, THREADS, 0, streams[0]>>>(a, N, 3);
    fillMatrix<<<blocks, THREADS, 0, streams[1]>>>(b, N, 7);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());       // every band needs all of A and B

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int rowsPerStream = N / NUM_STREAMS;       // N is divisible by 4
    for (int s = 0; s < NUM_STREAMS; ++s) {
        int rowStart = s * rowsPerStream;
        matMulRows<<<blocksPerStream, THREADS, 0, streams[s]>>>(a, b, c, N, rowStart, rowsPerStream);
        // Same stream -> runs after this band's kernel, while other bands still compute
        prefetch(c + (size_t)rowStart * N, (size_t)rowsPerStream * N * sizeof(int),
                 cudaCpuDeviceId, streams[s]);
    }
    CUDA_CHECK(cudaGetLastError());
    // With the legacy default stream, this record waits for all blocking streams,
    // so the measured time covers every band's kernel and migration.
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuMs;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    matMulHost(a, b, cHost, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double hostMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = verifyResult(c, cHost, N);
    printf("%s | GPU (4 streams) %.3f ms | host %.3f ms\n", ok ? "CORRECT" : "WRONG", gpuMs, hostMs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    for (int s = 0; s < NUM_STREAMS; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    free(cHost);
    return ok ? 0 : 1;
}
