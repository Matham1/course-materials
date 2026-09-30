// =============================================================================
// PRACTICE 3 - Parallel sum reduction (Harris, Lecture 6)          (~60 min)
//
// TASK:
//   The CUDA with C++ program computes the sum of an integer array of
//   N = 1 << 24 elements in parallel. Following functions are expected:
//     - kernel to fill the array (value = i % 10);
//     - reduction kernel using shared memory: each thread first accumulates
//       several elements with a grid-stride loop (algorithm cascading), then
//       the block does a tree reduction with SEQUENTIAL addressing, and the
//       last warp is unrolled;
//     - host function computing the sum sequentially;
//     - host function to verify the result;
//     - main function.
//   Because CUDA has no global synchronization, reduce in TWO kernel launches
//   (blocks -> partial sums -> one block). Number of blocks = warp size *
//   number of SMs. Measure running time of the GPU and host versions.
//
// WHAT IT PRACTISES: __shared__ memory, __syncthreads, kernel decomposition,
// reduction #3-#7 ideas, volatile warp unrolling.
// Build: nvcc -O2 -o reduce 04_reduction_sum.cu
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define N (1 << 24)
#define THREADS 256                 // power of two, >= 64 (needed by warpReduce)

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

__global__ void fillArray(int *x, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) x[i] = i % 10;
}

// Last-warp unroll (reduction #5). The slides rely on warp lock-step + volatile;
// on Volta and newer GPUs threads of a warp can diverge, so __syncwarp() is
// added between each read and write to keep it correct.
__device__ void warpReduce(volatile int *sdata, int tid)
{
    int v = sdata[tid];
    v += sdata[tid + 32]; __syncwarp(); sdata[tid] = v; __syncwarp();
    v += sdata[tid + 16]; __syncwarp(); sdata[tid] = v; __syncwarp();
    v += sdata[tid + 8];  __syncwarp(); sdata[tid] = v; __syncwarp();
    v += sdata[tid + 4];  __syncwarp(); sdata[tid] = v; __syncwarp();
    v += sdata[tid + 2];  __syncwarp(); sdata[tid] = v; __syncwarp();
    v += sdata[tid + 1];  __syncwarp(); sdata[tid] = v;
}

// One partial sum per block, written to out[blockIdx.x]
__global__ void reduceSum(const int *in, int *out, int n)
{
    __shared__ int sdata[THREADS];
    int tid = threadIdx.x;

    // (a) First add during load + multiple elements per thread (reductions #4, #7).
    //     Stride = 2 * blockDim * gridDim keeps global loads coalesced.
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    int gridSize = blockDim.x * 2 * gridDim.x;
    int sum = 0;
    while (i < n) {
        sum += in[i];
        if (i + blockDim.x < n) sum += in[i + blockDim.x];
        i += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();

    // (b) Tree reduction with sequential addressing (reduction #3): no bank conflicts,
    //     no divergent modulo.
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // (c) Unroll the last warp (reduction #5)
    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

long long sumHost(const int *x, int n)
{
    long long s = 0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s;
}

bool verifyResult(long long gpu, long long cpu)
{
    if (gpu != cpu) { printf("Mismatch: GPU %lld vs CPU %lld\n", gpu, cpu); return false; }
    return true;
}

int main()
{
    int deviceId, numSMs, warpSize;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,   cudaDevAttrMultiProcessorCount, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize,            deviceId));
    int blocks = warpSize * numSMs;

    size_t bytes = (size_t)N * sizeof(int);
    int *x, *partial, *result;
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    CUDA_CHECK(cudaMallocManaged(&partial, blocks * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(int)));

    prefetch(x, bytes, deviceId);
    prefetch(partial, blocks * sizeof(int), deviceId);
    prefetch(result, sizeof(int), deviceId);
    fillArray<<<blocks, THREADS>>>(x, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    reduceSum<<<blocks, THREADS>>>(x, partial, N);           // level 0: many blocks
    reduceSum<<<1, THREADS>>>(partial, result, blocks);      // level 1: one block
    CUDA_CHECK(cudaEventRecord(stop));                       // launch = global sync point
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float kernelMs;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));

    prefetch(x, bytes, cudaCpuDeviceId);
    prefetch(result, sizeof(int), cudaCpuDeviceId);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    long long hostSum = sumHost(x, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double hostMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = verifyResult(*result, hostSum);
    double gbps = bytes / (kernelMs * 1e-3) / 1e9;           // reductions are memory-bound
    printf("%s | sum %d | kernel %.4f ms (%.1f GB/s) | host %.4f ms\n",
           ok ? "CORRECT" : "WRONG", *result, kernelMs, gbps, hostMs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaFree(result));
    return ok ? 0 : 1;
}
