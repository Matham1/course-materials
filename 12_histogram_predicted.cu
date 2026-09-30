// =============================================================================
// PREDICTION #3 - Midterm-style task built on week-4 DLI-03
//                 ("Implementing New Algorithms with CUDA Kernels": histogram,
//                 data race, atomics, privatized histogram, __syncthreads)
//
// PREDICTED TASK:
//   The CUDA with C++ program builds a histogram of N = 1 << 22 temperatures
//   (values 0..99.9 degrees) with NUM_BINS = 10 bins of 10 degrees each.
//   Following functions are expected:
//     - kernel to fill the temperatures on the GPU;
//     - kernel that builds the histogram with atomic operations;
//     - kernel that uses a per-block (privatized) histogram in shared memory,
//       then adds it to the global histogram;
//     - host function computing the histogram;
//     - host function to verify both GPU results;
//     - main function.
//   Blocks = warp size * SMs, grid-stride loops, managed memory + prefetch,
//   timing of every version, clear memory.
//
// NOTE: the DLI slides use cuda::std::atomic_ref<int>(x).fetch_add(1).
//       Plain atomicAdd(&x, 1) does the same and needs no extra header.
// Build: nvcc -arch=sm_86 -o hist 12_histogram_predicted.cu -run
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define N (1 << 22)
#define NUM_BINS 10
#define THREADS 256

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
  if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s (%s)\n", file, line, cudaGetErrorString(err), func);
    exit(1);
  }
}

void prefetchTo(void *ptr, size_t size, int deviceId, bool toGPU)
{
#if CUDART_VERSION >= 13000
  cudaMemLocation loc;
  loc.type = toGPU ? cudaMemLocationTypeDevice : cudaMemLocationTypeHost;
  loc.id   = toGPU ? deviceId : 0;
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, loc, 0, 0));
#else
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, toGPU ? deviceId : cudaCpuDeviceId));
#endif
}

__host__ __device__ int binOf(float t) { return (int)(t / 10.0f); }   // 10-degree bins

__global__ void fillTemperatures(float *t, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    t[i] = (float)((i * 37) % 1000) / 10.0f;                           // 0.0 .. 99.9
}

__global__ void zeroHistogram(int *h)
{
  if (threadIdx.x < NUM_BINS && blockIdx.x == 0) h[threadIdx.x] = 0;
}

// Version A: every thread increments the global histogram atomically.
// Without atomicAdd, hist[b]++ is read-modify-write -> DATA RACE -> wrong counts.
__global__ void histogramAtomic(const float *t, int *hist, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    atomicAdd(&hist[binOf(t[i])], 1);
}

// Version B: privatized histogram in shared memory (one per block).
// Contention drops from "all threads on 10 counters" to "one block's threads",
// then each block adds its 10 counts to the global histogram once.
__global__ void histogramShared(const float *t, int *hist, int n)
{
  __shared__ int local[NUM_BINS];
  if (threadIdx.x < NUM_BINS) local[threadIdx.x] = 0;
  __syncthreads();                                   // counters zeroed before anyone adds

  int index = threadIdx.x + blockIdx.x * blockDim.x, stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    atomicAdd(&local[binOf(t[i])], 1);               // shared-memory atomics: cheap
  __syncthreads();                                   // all adds done before reading local[]

  if (threadIdx.x < NUM_BINS)
    atomicAdd(&hist[threadIdx.x], local[threadIdx.x]);
}

void histogramHost(const float *t, int *hist, int n)
{
  for (int b = 0; b < NUM_BINS; ++b) hist[b] = 0;
  for (int i = 0; i < n; ++i) hist[binOf(t[i])]++;
}

bool verifyResult(const char *name, const int *gpu, const int *cpu)
{
  for (int b = 0; b < NUM_BINS; ++b)
    if (gpu[b] != cpu[b]) {
      printf("%s: bin %d has %d, expected %d\n", name, b, gpu[b], cpu[b]);
      return false;
    }
  return true;
}

int main()
{
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  int blocks = props.warpSize * props.multiProcessorCount;

  size_t tSize = N * sizeof(float), hSize = NUM_BINS * sizeof(int);
  float *t;
  int *histA, *histB;
  CHECK_CUDA_ERROR(cudaMallocManaged(&t, tSize));
  CHECK_CUDA_ERROR(cudaMallocManaged(&histA, hSize));
  CHECK_CUDA_ERROR(cudaMallocManaged(&histB, hSize));
  int histCPU[NUM_BINS];

  prefetchTo(t, tSize, deviceId, true);
  prefetchTo(histA, hSize, deviceId, true);
  prefetchTo(histB, hSize, deviceId, true);

  fillTemperatures<<<blocks, THREADS>>>(t, N);
  zeroHistogram<<<1, 32>>>(histA);
  zeroHistogram<<<1, 32>>>(histB);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  float msA, msB;

  CHECK_CUDA_ERROR(cudaEventRecord(start));
  histogramAtomic<<<blocks, THREADS>>>(t, histA, N);
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&msA, start, stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start));
  histogramShared<<<blocks, THREADS>>>(t, histB, N);
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&msB, start, stop));
  CHECK_CUDA_ERROR(cudaGetLastError());

  prefetchTo(t, tSize, deviceId, false);
  prefetchTo(histA, hSize, deviceId, false);
  prefetchTo(histB, hSize, deviceId, false);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  auto h0 = std::chrono::high_resolution_clock::now();
  histogramHost(t, histCPU, N);
  auto h1 = std::chrono::high_resolution_clock::now();

  bool ok = verifyResult("global atomics", histA, histCPU) &&
            verifyResult("shared memory", histB, histCPU);
  for (int b = 0; b < NUM_BINS; ++b) printf("[%2d-%2d) %d\n", b * 10, b * 10 + 10, histCPU[b]);
  printf("%s | global atomics %.3f ms | shared memory %.3f ms | CPU %.3f ms\n",
         ok ? "CORRECT" : "WRONG", msA, msB,
         std::chrono::duration<double, std::milli>(h1 - h0).count());

  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  CHECK_CUDA_ERROR(cudaFree(t));
  CHECK_CUDA_ERROR(cudaFree(histA));
  CHECK_CUDA_ERROR(cudaFree(histB));
  return ok ? 0 : 1;
}
