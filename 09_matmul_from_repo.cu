// =============================================================================
// Midterm 1 (Fall 2025) solved by ASSEMBLING pieces of the course repository.
// Every block says which repo file it was adapted from ("FROM:").
// Paths are relative to  nu-courses-material/CSCI-423/
//
// Build on the lab machine (flag from week-6/C++_Examples/Example_1/makefile):
//     nvcc -arch=sm_86 -o lastname lastname.cu -run
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define N (1 << 6)        // FROM: week-2/08-matrix-multiply/01-matrix-multiply-2d.cu (#define N 64)

// -----------------------------------------------------------------------------
// Error check macro
// FROM: week-6/C++_Examples/Example_12/error.cu (CHECK_CUDA_ERROR), shortened
// -----------------------------------------------------------------------------
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
  if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s (%s)\n", file, line, cudaGetErrorString(err), func);
    exit(1);
  }
}

// -----------------------------------------------------------------------------
// Prefetch helper. The repo shows BOTH API styles:
//   old: week-5/part-3/01-vector-add/solutions/01-vector-add-prefetch-solution.cu
//   new: week-5/part-2/08-prefetch/solutions/02-vector-add-prefetch-solution-cpu-also.cu
// Keep only the branch that compiles on the lab machine if you prefer.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Kernel 1: fill a matrix.
// FROM: week-5/part-2/07-unit-in-kernel/solutions/01-vector-add-init-in-kernel-solution.cu
//       (initWith, grid-stride), with the values of the matrix-multiply starter:
//       a[row][col] = row,  b[row][col] = col + 2
// value = row * rowFactor + col * colFactor + offset
//   A: rowFactor 1, colFactor 0, offset 0  -> row
//   B: rowFactor 0, colFactor 1, offset 2  -> col + 2
// -----------------------------------------------------------------------------
__global__ void initMatrix(int *m, int rowFactor, int colFactor, int offset)
{
  int index  = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N * N; i += stride) {
    int row = i / N;
    int col = i % N;
    m[i] = row * rowFactor + col * colFactor + offset;
  }
}

// -----------------------------------------------------------------------------
// Kernel 2: matrix multiplication.  NOT in the repo - write it yourself.
// Recipe: take matrixMulCPU below, delete the two outer loops (row, col),
// and compute row/col from the thread's index instead.
// Grid-stride loop FROM: week-2/05-allocate/solutions/03-grid-stride-double-solution.cu
// -----------------------------------------------------------------------------
__global__ void matrixMulGPU(int *a, int *b, int *c)
{
  int index  = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N * N; i += stride) {   // one output cell per iteration
    int row = i / N;
    int col = i % N;
    int val = 0;
    for (int k = 0; k < N; ++k)                     // same inner loop as the CPU version
      val += a[row * N + k] * b[k * N + col];
    c[row * N + col] = val;
  }
}

// -----------------------------------------------------------------------------
// Host multiplication.  FROM: week-2/08-matrix-multiply/01-matrix-multiply-2d.cu (unchanged)
// -----------------------------------------------------------------------------
void matrixMulCPU(int *a, int *b, int *c)
{
  int val = 0;
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col) {
      val = 0;
      for (int k = 0; k < N; ++k)
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

// -----------------------------------------------------------------------------
// Host verification.  FROM: the compare loop at the end of main() in
// week-2/08-matrix-multiply/01-matrix-multiply-2d.cu, moved into a function
// -----------------------------------------------------------------------------
bool verifyResult(int *c_cpu, int *c_gpu)
{
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col)
      if (c_cpu[row * N + col] != c_gpu[row * N + col]) {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        return false;
      }
  return true;
}

int main()
{
  // ---- Launch configuration: blocks = warp size * SMs -----------------------
  // FROM: week-5/part-2/04-device-properties/solutions/01-get-device-properties-solution.cu
  //       week-5/part-2/05-vector-add-SM-blocks/solutions/01-vector-add-SM-blocks-solution.cu
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  int warpSize = props.warpSize;
  int numberOfSMs = props.multiProcessorCount;

  size_t threadsPerBlock = 256;                  // multiple of the warp size
  size_t numberOfBlocks  = warpSize * numberOfSMs;
  printf("SMs: %d, warp: %d -> %zu blocks x %zu threads\n",
         numberOfSMs, warpSize, numberOfBlocks, threadsPerBlock);

  // ---- Memory ---------------------------------------------------------------
  // FROM: week-2/08-matrix-multiply/01-matrix-multiply-2d.cu
  // Change: c_cpu is only used by the CPU, so plain malloc is enough.
  int *a, *b, *c_gpu;
  size_t size = N * N * sizeof(int);
  CHECK_CUDA_ERROR(cudaMallocManaged(&a, size));
  CHECK_CUDA_ERROR(cudaMallocManaged(&b, size));
  CHECK_CUDA_ERROR(cudaMallocManaged(&c_gpu, size));
  int *c_cpu = (int *)malloc(size);

  // ---- Data migration: to the GPU before any kernel -------------------------
  // FROM: week-5/part-2/08-prefetch/solutions/02-vector-add-prefetch-solution-cpu-also.cu
  prefetchTo(a, size, deviceId, true);
  prefetchTo(b, size, deviceId, true);
  prefetchTo(c_gpu, size, deviceId, true);

  // ---- Fill A and B on independent streams ----------------------------------
  // FROM: week-5/part-3/06-stream-init/solutions/01-stream-init-solution.cu
  cudaStream_t stream1, stream2;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

  initMatrix<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(a, 1, 0, 0);  // a = row
  initMatrix<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(b, 0, 1, 2);  // b = col + 2
  CHECK_CUDA_ERROR(cudaGetLastError());          // FROM: week-2/06-errors solution
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());     // both fills must finish first

  // ---- GPU multiplication, timed ---------------------------------------------
  // Timing idea FROM: week-5/part-3/09-nbody/01-nbody.cu (timer + cudaDeviceSynchronize:
  // "launches are async, so wait or the timing is fiction"). std::chrono keeps it one file.
  auto g0 = std::chrono::high_resolution_clock::now();
  matrixMulGPU<<<numberOfBlocks, threadsPerBlock>>>(a, b, c_gpu);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  auto g1 = std::chrono::high_resolution_clock::now();

  // ---- Data migration: back to the CPU before host code reads it ------------
  prefetchTo(a, size, deviceId, false);
  prefetchTo(b, size, deviceId, false);
  prefetchTo(c_gpu, size, deviceId, false);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());     // (the repo file forgets this sync)

  // ---- CPU multiplication, timed ---------------------------------------------
  auto h0 = std::chrono::high_resolution_clock::now();
  matrixMulCPU(a, b, c_cpu);
  auto h1 = std::chrono::high_resolution_clock::now();

  // ---- Verify + report --------------------------------------------------------
  bool ok = verifyResult(c_cpu, c_gpu);
  printf("%s\n", ok ? "Success!" : "FAILED");
  printf("GPU time: %.4f ms\n", std::chrono::duration<double, std::milli>(g1 - g0).count());
  printf("CPU time: %.4f ms\n", std::chrono::duration<double, std::milli>(h1 - h0).count());

  // ---- Clear memory -------------------------------------------------------------
  // FROM: week-5/part-3/06-stream-init/solutions/01-stream-init-solution.cu (end of main)
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
  CHECK_CUDA_ERROR(cudaFree(a));
  CHECK_CUDA_ERROR(cudaFree(b));
  CHECK_CUDA_ERROR(cudaFree(c_gpu));
  free(c_cpu);
  return ok ? 0 : 1;
}
