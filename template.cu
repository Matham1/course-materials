/*
 * template.cu - exam scaffold written in the style of the course repository
 * (nu-courses-material/CSCI-423, week-2 and week-5 solutions).
 *
 * How to use it in the exam:
 *   1. Copy to lastname.cu
 *   2. Replace every "TODO" block with the task's computation
 *   3. Build and run:   nvcc -arch=sm_86 -o lastname lastname.cu -run
 *
 * As written, it adds two vectors (c = a + b) so it compiles and runs out of the box.
 * Every part the midterm rubric asks for is already in place:
 *   fill kernel on independent streams, compute kernel, host version,
 *   host verification, blocks = warp size * SMs, grid-stride loops,
 *   prefetching both ways, timing, error checks, freeing memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

/*
 * Problem size. For an N x N matrix task use:  #define N (1 << 6)
 * and loop over N * N elements in the kernels.
 */

#define N (2<<20)

/*
 * Timer - the Linux part of week-5/part-3/09-nbody/timer.h,
 * copied in because only one .cu file may be submitted.
 */

struct timeval timerStart;

void StartTimer()
{
  gettimeofday(&timerStart, NULL);
}

// time elapsed in ms
double GetTimer()
{
  struct timeval timerStop, timerElapsed;
  gettimeofday(&timerStop, NULL);
  timersub(&timerStop, &timerStart, &timerElapsed);
  return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}

/*
 * Kernel to initialize the data - week-5/part-2/07-unit-in-kernel/solutions
 * Uses a grid-stride loop so it works for any number of blocks.
 */

__global__
void initWith(float num, float *a, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride)
  {
    a[i] = num;                                   // TODO: task-specific initial values
  }
}

/*
 * Kernel for the main computation - grid-stride loop from
 * week-2/05-allocate/solutions/03-grid-stride-double-solution.cu
 *
 * For a matrix task recover the 2D position from the flat index:
 *   int row = i / N;   int col = i % N;
 */

__global__
void computeGPU(float *result, float *a, float *b, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < n; i += stride)
  {
    result[i] = a[i] + b[i];                      // TODO: task-specific computation
  }
}

/*
 * Host version of the same computation (reference for verification).
 */

void computeCPU(float *result, float *a, float *b, int n)
{
  for(int i = 0; i < n; ++i)
  {
    result[i] = a[i] + b[i];                      // TODO: same math as computeGPU
  }
}

/*
 * Host function to verify the GPU result against the CPU result.
 * Integers: compare with ==.  Floats: compare with a small tolerance.
 */

bool checkResults(float *gpu, float *cpu, int n)
{
  for(int i = 0; i < n; i++)
  {
    if(fabs(gpu[i] - cpu[i]) > 0.0005)
    {
      printf("FAIL: element %d - GPU %f does not equal CPU %f\n", i, gpu[i], cpu[i]);
      return false;
    }
  }
  printf("Success! All values calculated correctly.\n");
  return true;
}

int main()
{
  /*
   * Device properties - week-5/part-2/04-device-properties/solutions
   * Number of blocks = warp size * number of SMs.
   */

  int deviceId;
  int numberOfSMs;
  int warpSize;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = warpSize * numberOfSMs;

  printf("Device ID: %d\tNumber of SMs: %d\tWarp size: %d\n", deviceId, numberOfSMs, warpSize);
  printf("Blocks: %zu\tThreads per block: %zu\n", numberOfBlocks, threadsPerBlock);

  /*
   * Memory. GPU data: cudaMallocManaged. CPU-only result: malloc.
   */

  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c_gpu;
  float *c_cpu;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c_gpu, size);
  c_cpu = (float *)malloc(size);

  /*
   * Prefetch to the GPU before the kernels run -
   * week-5/part-2/08-prefetch/solutions/02-vector-add-prefetch-solution-cpu-also.cu
   *
   * CUDA 12 or older? Use instead:   cudaMemPrefetchAsync(a, size, deviceId);
   */

  cudaMemLocation locationGPU;
  locationGPU.type = cudaMemLocationTypeDevice;
  locationGPU.id = deviceId;

  cudaMemPrefetchAsync(a, size, locationGPU, 0, 0);
  cudaMemPrefetchAsync(b, size, locationGPU, 0, 0);
  cudaMemPrefetchAsync(c_gpu, size, locationGPU, 0, 0);

  cudaError_t initErr;
  cudaError_t computeErr;
  cudaError_t asyncErr;

  /*
   * Initialize the data on independent streams -
   * week-5/part-3/06-stream-init/solutions/01-stream-init-solution.cu
   */

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(3, a, N);
  initWith<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(4, b, N);

  initErr = cudaGetLastError();
  if(initErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(initErr));

  /*
   * Both inputs must be ready before the computation starts.
   */

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  /*
   * GPU computation, timed. The synchronize is INSIDE the timed region:
   * launches are async, so wait or the timing is fiction (09-nbody).
   */

  StartTimer();

  computeGPU<<<numberOfBlocks, threadsPerBlock>>>(c_gpu, a, b, N);

  computeErr = cudaGetLastError();
  if(computeErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(computeErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  double gpuTime = GetTimer();

  /*
   * Prefetch back to the CPU before host code reads the data,
   * then synchronize so the transfer is finished.
   *
   * CUDA 12 or older? Use instead:   cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
   */

  cudaMemLocation locationCPU;
  locationCPU.type = cudaMemLocationTypeHost;
  locationCPU.id = 0;

  cudaMemPrefetchAsync(a, size, locationCPU, 0, 0);
  cudaMemPrefetchAsync(b, size, locationCPU, 0, 0);
  cudaMemPrefetchAsync(c_gpu, size, locationCPU, 0, 0);

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  /*
   * CPU computation, timed.
   */

  StartTimer();

  computeCPU(c_cpu, a, b, N);

  double cpuTime = GetTimer();

  /*
   * Verify and report.
   */

  checkResults(c_gpu, c_cpu, N);

  printf("GPU time: %.4f ms\n", gpuTime);
  printf("CPU time: %.4f ms\n", cpuTime);

  /*
   * Destroy streams and free all memory.
   */

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c_gpu);
  free(c_cpu);
}
