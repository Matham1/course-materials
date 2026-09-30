// =============================================================================
// CSCI 423/523/723 - Midterm 1, Fall 2025 - MODEL SOLUTION
//
// TASK (from the exam sheet):
//   The CUDA with C++ program multiplies two NxN integer matrices in parallel.
//   N = 1 << 6. Required functions:
//     - kernel to fill the matrices on independent streams;
//     - kernel to implement the multiplication;
//     - host function to implement the multiplication;
//     - host function to verify the correctness of the result;
//     - main function.
//   Number of blocks = warp size * number of streaming multiprocessors (SMs).
//   Measure running time of the kernel and host versions of multiplication.
//
// RUBRIC -> where it is earned in this file
//   12% correct implementation ....... fillMatrix, matMulKernel, matMulHost,
//                                      verifyResult, main
//    2% grid-stride loop .............. both kernels (look for "stride")
//    2% optimal blocks / threads ...... main(): blocks = warpSize * numSMs,
//                                      threads = 256 (multiple of 32)
//    2% streams ....................... main(): streamA / streamB for filling
//    2% data migration + clear memory . prefetch() to GPU / CPU, cudaFree,
//                                      cudaStreamDestroy, cudaEventDestroy
//
// Build & run:  nvcc -O2 -o matmul 01_matmul_midterm2025.cu && ./matmul
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define N (1 << 6)          // matrix is N x N
#define THREADS 256         // threads per block: multiple of warp size (32)

// ---- Error checking: always wrap CUDA API calls ----------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---- Prefetch helper (Unified Memory, Week 5 L1) ---------------------------
// Moves managed memory to `device` (GPU id, or cudaCpuDeviceId for the host)
// ahead of time, so kernels / host code do not page-fault on first access.
// CUDA 13 changed the signature, hence the #if. In the exam, write the
// branch that matches the lab's CUDA version.
void prefetch(const void *ptr, size_t bytes, int device, cudaStream_t stream = 0)
{
    static int supported = -1;
    if (supported < 0) {                       // e.g. Windows: no prefetch support
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

// ---- Kernel 1: fill a matrix (launched on its own stream) ------------------
// Grid-stride loop: each thread handles elements i, i+stride, i+2*stride, ...
// so the kernel is correct for ANY grid size and ANY matrix size.
__global__ void fillMatrix(int *m, int n, int seed)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n * n; i += stride) {
        m[i] = (i * seed + 1) % 10;            // small values -> no int overflow
    }
}

// ---- Kernel 2: C = A * B ----------------------------------------------------
// One output element per loop iteration; row/col recovered from the flat index
// (row = idx / n, col = idx % n), exactly like the DLI heat-equation example.
__global__ void matMulKernel(const int *a, const int *b, int *c, int n)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = index; idx < n * n; idx += stride) {
        int row = idx / n;
        int col = idx % n;
        int sum = 0;
        for (int k = 0; k < n; ++k)
            sum += a[row * n + k] * b[k * n + col];
        c[idx] = sum;
    }
}

// ---- Host multiplication (reference) ---------------------------------------
void matMulHost(const int *a, const int *b, int *c, int n)
{
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) {
            int sum = 0;
            for (int k = 0; k < n; ++k)
                sum += a[row * n + k] * b[k * n + col];
            c[row * n + col] = sum;
        }
}

// ---- Host verification ------------------------------------------------------
bool verifyResult(const int *gpu, const int *cpu, int n)
{
    for (int i = 0; i < n * n; ++i) {
        if (gpu[i] != cpu[i]) {
            printf("Mismatch at (%d, %d): GPU %d vs CPU %d\n", i / n, i % n, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

// Small helper to eyeball results when testing with N = 4
void printMatrix(const char *name, const int *m, int n)
{
    printf("%s:\n", name);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) printf("%6d", m[r * n + c]);
        printf("\n");
    }
}

int main()
{
    // ---- 1. Query the device: blocks = warp size * number of SMs ----------
    int deviceId = 0, numSMs = 0, warpSize = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,   cudaDevAttrMultiProcessorCount, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize,            deviceId));
    int blocks  = warpSize * numSMs;
    int threads = THREADS;
    printf("N = %d, SMs = %d, warp = %d -> %d blocks x %d threads\n",
           N, numSMs, warpSize, blocks, threads);

    // ---- 2. Allocate Unified (managed) memory ------------------------------
    size_t bytes = (size_t)N * N * sizeof(int);
    int *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    int *cHost = (int *)malloc(bytes);         // host result: plain host memory

    // ---- 3. Two independent streams for filling A and B --------------------
    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));

    // Data migration: move each matrix to the GPU in the stream that uses it
    prefetch(a, bytes, deviceId, streamA);
    prefetch(b, bytes, deviceId, streamB);
    prefetch(c, bytes, deviceId);

    fillMatrix<<<blocks, threads, 0, streamA>>>(a, N, 3);
    fillMatrix<<<blocks, threads, 0, streamB>>>(b, N, 7);
    CUDA_CHECK(cudaGetLastError());            // catches bad launch configs
    CUDA_CHECK(cudaStreamSynchronize(streamA));
    CUDA_CHECK(cudaStreamSynchronize(streamB));

    // ---- 4. Time the kernel with CUDA events -------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matMulKernel<<<blocks, threads>>>(a, b, c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));    // wait: kernels are asynchronous!
    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));

    // ---- 5. Bring data back to the CPU before host code touches it ---------
    prefetch(a, bytes, cudaCpuDeviceId);
    prefetch(b, bytes, cudaCpuDeviceId);
    prefetch(c, bytes, cudaCpuDeviceId);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 6. Time the host version with std::chrono -------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    matMulHost(a, b, cHost, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double hostMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- 7. Verify and report ----------------------------------------------
    if (N <= 8) { printMatrix("A", a, N); printMatrix("B", b, N); printMatrix("C", c, N); }
    bool ok = verifyResult(c, cHost, N);
    printf("Result: %s\n", ok ? "CORRECT" : "WRONG");
    printf("Kernel time: %.4f ms | Host time: %.4f ms\n", kernelMs, hostMs);

    // ---- 8. Clean up everything that was created ---------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    free(cHost);
    return ok ? 0 : 1;
}
