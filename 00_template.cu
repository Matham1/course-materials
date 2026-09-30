// =============================================================================
// 00_template.cu - exam scaffold. Copy, rename to lastname.cu, fill the TODOs.
// Every midterm-style task = this skeleton + 2-3 problem-specific kernels.
// Build: nvcc -O2 -o prog 00_template.cu
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define N (1 << 10)
#define THREADS 256                       // multiple of 32

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Unified Memory prefetch that works on CUDA 12 and CUDA 13, and is skipped
// on GPUs/OSes without concurrent managed access (e.g. Windows).
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

// ---- Kernels ----------------------------------------------------------------
__global__ void initArray(int *x, int n, int value)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = value;                          // TODO: problem-specific init
}

__global__ void computeKernel(const int *a, const int *b, int *c, int n)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride)
        c[i] = a[i] + b[i];                    // TODO: problem-specific work
}

// ---- Host reference + verification ----------------------------------------
void computeHost(const int *a, const int *b, int *c, int n)
{
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];   // TODO: same math as kernel
}

bool verify(const int *gpu, const int *cpu, int n)
{
    for (int i = 0; i < n; ++i)
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at %d: %d vs %d\n", i, gpu[i], cpu[i]);
            return false;
        }
    return true;
}

int main()
{
    // 1. launch configuration from the device
    int deviceId, numSMs, warpSize;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,   cudaDevAttrMultiProcessorCount, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize,            deviceId));
    int blocks = warpSize * numSMs, threads = THREADS;

    // 2. memory
    const int n = N;
    size_t bytes = (size_t)n * sizeof(int);
    int *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    int *cHost = (int *)malloc(bytes);

    // 3. streams + init on independent streams
    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));
    prefetch(a, bytes, deviceId, s1);
    prefetch(b, bytes, deviceId, s2);
    prefetch(c, bytes, deviceId);
    initArray<<<blocks, threads, 0, s1>>>(a, n, 1);
    initArray<<<blocks, threads, 0, s2>>>(b, n, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaStreamSynchronize(s2));

    // 4. timed kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    computeKernel<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernelMs;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));

    // 5. back to CPU, timed host version, verify
    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    computeHost(a, b, cHost, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double hostMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    bool ok = verify(c, cHost, n);
    printf("%s | kernel %.4f ms | host %.4f ms\n", ok ? "CORRECT" : "WRONG", kernelMs, hostMs);

    // 6. cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    free(cHost);
    return ok ? 0 : 1;
}
