// =============================================================================
// PREDICTION #1 - Midterm-style task built on the UNSOLVED repo exercise
//                 week-2/09-heat/01-heat-conduction.cu
//
// PREDICTED TASK (written in last year's format):
//   The CUDA with C++ program simulates 2D heat conduction on an ni x nj float
//   grid (ni = 200, nj = 100) for nstep = 200 time steps, using
//       out[i,j] = in[i,j] + fact * (d2T/dx2 + d2T/dy2),  fact = 8.418e-5.
//   Boundary cells are not updated. Following functions are expected:
//     - kernel to initialise the two temperature grids on independent streams;
//     - kernel to implement one time step;
//     - host function to implement one time step (reference);
//     - host function to verify the result (max error < 0.0005);
//     - main function.
//   Number of blocks = warp size * number of SMs. Measure running time of the
//   kernel and host versions. Use grid-stride loops, streams, optimal data
//   migration, clear memory.
//
// WHAT CHANGES vs. the repo starter:
//   - step_kernel_mod becomes a __global__ kernel: the two for-loops over (i, j)
//     are replaced by ONE grid-stride loop over all cells, skipping boundaries.
//   - the random CPU initialisation becomes an init kernel (random numbers are
//     awkward on the GPU, so a deterministic formula is used instead).
//   - memory becomes cudaMallocManaged + prefetch; the pointer swap stays on the host.
//
// Build: nvcc -arch=sm_86 -o heat 10_heat_conduction_predicted.cu -run
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// FROM the starter: index into a 1D array from 2D space (c = column, r = row)
#define I2D(num, c, r) ((r)*(num)+(c))

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
  if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s (%s)\n", file, line, cudaGetErrorString(err), func);
    exit(1);
  }
}

void prefetchTo(void *ptr, size_t size, int deviceId, bool toGPU, cudaStream_t stream = 0)
{
#if CUDART_VERSION >= 13000
  cudaMemLocation loc;
  loc.type = toGPU ? cudaMemLocationTypeDevice : cudaMemLocationTypeHost;
  loc.id   = toGPU ? deviceId : 0;
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, loc, 0, stream));
#else
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, toGPU ? deviceId : cudaCpuDeviceId, stream));
#endif
}

// Same starting values for GPU and CPU. Used by the init kernel AND the host.
__host__ __device__ float initialTemp(int idx) { return (float)((idx * 37) % 100); }

// ---- Kernel 1: initialise a grid (launched once per grid, each on its own stream)
__global__ void initGrid(float *temp, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int k = index; k < n; k += stride)
    temp[k] = initialTemp(k);
}

// ---- Kernel 2: one time step ----------------------------------------------------
// The starter loops j = 1..nj-2 and i = 1..ni-2. Here each flat index k is turned
// back into (i, j) and boundary cells are skipped - same work, done in parallel.
__global__ void step_kernel_mod(int ni, int nj, float fact, float *temp_in, float *temp_out)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int k = index; k < ni * nj; k += stride) {
    int i = k % ni;          // column  (I2D uses r * num + c, with num = ni)
    int j = k / ni;          // row
    if (i < 1 || i > ni - 2 || j < 1 || j > nj - 2) continue;   // boundary: untouched

    int i00  = I2D(ni, i, j);
    int im10 = I2D(ni, i - 1, j);
    int ip10 = I2D(ni, i + 1, j);
    int i0m1 = I2D(ni, i, j - 1);
    int i0p1 = I2D(ni, i, j + 1);

    float d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
    float d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];
    temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
  }
}

// ---- Host reference: unchanged from the starter ----------------------------------
void step_kernel_ref(int ni, int nj, float fact, float *temp_in, float *temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;
  for (int j = 1; j < nj - 1; j++) {
    for (int i = 1; i < ni - 1; i++) {
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i - 1, j);
      ip10 = I2D(ni, i + 1, j);
      i0m1 = I2D(ni, i, j - 1);
      i0p1 = I2D(ni, i, j + 1);
      d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
      d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];
      temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
    }
  }
}

// ---- Host verification: the starter's max-error check, as a function ------------
bool verifyResult(float *gpu, float *cpu, int n, float *maxErrorOut)
{
  float maxError = 0;
  for (int k = 0; k < n; ++k)
    if (fabsf(gpu[k] - cpu[k]) > maxError) maxError = fabsf(gpu[k] - cpu[k]);
  *maxErrorOut = maxError;
  return maxError <= 0.0005f;
}

int main()
{
  const int nstep = 200;
  const int ni = 200, nj = 100;
  const float tfac = 8.418e-5f;
  const int n = ni * nj;
  size_t size = n * sizeof(float);

  // ---- launch configuration: warp size * SMs --------------------------------------
  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = props.warpSize * props.multiProcessorCount;

  // ---- memory: GPU grids managed, CPU reference grids plain malloc ------------------
  float *temp1, *temp2;
  CHECK_CUDA_ERROR(cudaMallocManaged(&temp1, size));
  CHECK_CUDA_ERROR(cudaMallocManaged(&temp2, size));
  float *temp1_ref = (float *)malloc(size);
  float *temp2_ref = (float *)malloc(size);

  // ---- init both grids on independent streams ---------------------------------------
  cudaStream_t stream1, stream2;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
  prefetchTo(temp1, size, deviceId, true, stream1);
  prefetchTo(temp2, size, deviceId, true, stream2);
  initGrid<<<numberOfBlocks, threadsPerBlock, 0, stream1>>>(temp1, n);
  initGrid<<<numberOfBlocks, threadsPerBlock, 0, stream2>>>(temp2, n);   // boundaries of temp2 must match too
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  for (int k = 0; k < n; ++k) temp1_ref[k] = temp2_ref[k] = initialTemp(k);

  // ---- GPU simulation, timed ----------------------------------------------------------
  auto g0 = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < nstep; istep++) {
    step_kernel_mod<<<numberOfBlocks, threadsPerBlock>>>(ni, nj, tfac, temp1, temp2);
    float *tmp = temp1; temp1 = temp2; temp2 = tmp;   // swap on the host: kernels stay in order
  }
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  auto g1 = std::chrono::high_resolution_clock::now();

  // ---- CPU reference, timed ---------------------------------------------------------
  auto h0 = std::chrono::high_resolution_clock::now();
  for (int istep = 0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);
    float *tmp = temp1_ref; temp1_ref = temp2_ref; temp2_ref = tmp;
  }
  auto h1 = std::chrono::high_resolution_clock::now();

  // ---- back to the CPU, verify ------------------------------------------------------
  prefetchTo(temp1, size, deviceId, false);
  prefetchTo(temp2, size, deviceId, false);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  float maxError;
  bool ok = verifyResult(temp1, temp1_ref, n, &maxError);   // result sits in temp1 after the swaps
  printf("%s: max error %.6f\n", ok ? "CORRECT" : "WRONG", maxError);
  printf("GPU %.3f ms | CPU %.3f ms\n",
         std::chrono::duration<double, std::milli>(g1 - g0).count(),
         std::chrono::duration<double, std::milli>(h1 - h0).count());

  // ---- clean up -----------------------------------------------------------------------
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
  CHECK_CUDA_ERROR(cudaFree(temp1));
  CHECK_CUDA_ERROR(cudaFree(temp2));
  free(temp1_ref);
  free(temp2_ref);
  return ok ? 0 : 1;
}
