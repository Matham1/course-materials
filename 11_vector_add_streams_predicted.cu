// =============================================================================
// PREDICTION #2 - Midterm-style task built on the UNSOLVED repo exercise
//                 week-2/07-vector-add/01-vector-add.cu (CPU-only vector add)
//                 + the Lab 2 topic "accelerate using concurrent streams"
//                 (week-5/part-3/06-stream-init/solutions/03-sliced.cu, 04-pinned.cu)
//
// PREDICTED TASK:
//   The CUDA with C++ program adds two float vectors of N = 2 << 20 elements.
//   The vectors are split into NUM_STREAMS = 4 equal chunks; each chunk is
//   copied to the GPU, added and copied back on its OWN stream, so copies of
//   one chunk overlap with computation of another. Following functions are
//   expected:
//     - host function to initialise the vectors (a = 3, b = 4);
//     - kernel to add the vectors (grid-stride loop);
//     - host function to add the vectors;
//     - host function to verify the result;
//     - main function.
//   Use device memory (cudaMalloc), pinned host memory (cudaMallocHost) and
//   cudaMemcpyAsync. Blocks = warp size * SMs. Measure the GPU and host times.
//
// Build: nvcc -arch=sm_86 -o vadd 11_vector_add_streams_predicted.cu -run
// =============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define NUM_STREAMS 4

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char *const func, const char *const file, const int line)
{
  if (err != cudaSuccess) {
    printf("CUDA error at %s:%d: %s (%s)\n", file, line, cudaGetErrorString(err), func);
    exit(1);
  }
}

// FROM the starter (host init)
void initWith(float num, float *a, int N)
{
  for (int i = 0; i < N; ++i) a[i] = num;
}

// The starter's addVectorsInto, turned into a grid-stride kernel
__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride)
    result[i] = a[i] + b[i];
}

// FROM the starter (host version, unchanged)
void addVectorsHost(float *result, float *a, float *b, int N)
{
  for (int i = 0; i < N; ++i) result[i] = a[i] + b[i];
}

bool verifyResult(float *gpu, float *cpu, int N)
{
  for (int i = 0; i < N; i++)
    if (fabsf(gpu[i] - cpu[i]) > 1e-6f) {
      printf("FAIL: element %d: %f vs %f\n", i, gpu[i], cpu[i]);
      return false;
    }
  return true;
}

int main()
{
  const int N = 2 << 20;
  const int chunkN = N / NUM_STREAMS;              // N divisible by 4
  size_t size = N * sizeof(float);
  size_t chunkSize = chunkN * sizeof(float);

  int deviceId;
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = props.warpSize * props.multiProcessorCount;

  // Pinned host memory: required for truly asynchronous copies (04-pinned.cu)
  float *h_a, *h_b, *h_c, *h_ref;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_a, size));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_b, size));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_c, size));
  h_ref = (float *)malloc(size);                   // CPU-only result: plain malloc

  // Device memory (no Unified Memory -> no page faults, explicit copies)
  float *d_a, *d_b, *d_c;
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
  CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

  initWith(3, h_a, N);
  initWith(4, h_b, N);

  cudaStream_t streams[NUM_STREAMS];
  for (int s = 0; s < NUM_STREAMS; ++s) CHECK_CUDA_ERROR(cudaStreamCreate(&streams[s]));

  // ---- GPU: copy in -> add -> copy out, per chunk, per stream --------------------------
  auto g0 = std::chrono::high_resolution_clock::now();
  for (int s = 0; s < NUM_STREAMS; ++s) {
    int offset = s * chunkN;                       // pointer arithmetic selects the chunk
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + offset, h_a + offset, chunkSize, cudaMemcpyHostToDevice, streams[s]));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b + offset, h_b + offset, chunkSize, cudaMemcpyHostToDevice, streams[s]));
    addVectorsInto<<<numberOfBlocks / NUM_STREAMS, threadsPerBlock, 0, streams[s]>>>(
        d_c + offset, d_a + offset, d_b + offset, chunkN);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c + offset, d_c + offset, chunkSize, cudaMemcpyDeviceToHost, streams[s]));
  }
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());       // wait for every stream
  auto g1 = std::chrono::high_resolution_clock::now();

  // ---- CPU reference ---------------------------------------------------------------------
  auto h0 = std::chrono::high_resolution_clock::now();
  addVectorsHost(h_ref, h_a, h_b, N);
  auto h1 = std::chrono::high_resolution_clock::now();

  bool ok = verifyResult(h_c, h_ref, N);
  printf("%s | GPU (copies + kernel, %d streams) %.3f ms | CPU %.3f ms\n",
         ok ? "CORRECT" : "WRONG", NUM_STREAMS,
         std::chrono::duration<double, std::milli>(g1 - g0).count(),
         std::chrono::duration<double, std::milli>(h1 - h0).count());

  for (int s = 0; s < NUM_STREAMS; ++s) CHECK_CUDA_ERROR(cudaStreamDestroy(streams[s]));
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
  CHECK_CUDA_ERROR(cudaFreeHost(h_a));
  CHECK_CUDA_ERROR(cudaFreeHost(h_b));
  CHECK_CUDA_ERROR(cudaFreeHost(h_c));
  free(h_ref);
  return ok ? 0 : 1;
}
