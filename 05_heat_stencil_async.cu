// =============================================================================
// PRACTICE 4 - 2D heat equation with async snapshots (DLI 1 + DLI 2) (~75 min)
//
// TASK:
//   The CUDA with C++ program simulates heat diffusion on an H x W float grid
//   (H = W = 1 << 10) for STEPS = 200 time steps. Top and bottom rows are
//   heaters (90 degrees), all other cells start at 15 degrees, and boundary
//   cells stay constant. Each inner cell is updated with the stencil
//       next = t + 0.2 * ((left - 2t + right) + (up - 2t + down)).
//   Every SNAPSHOT_EVERY = 50 steps, the current grid must be copied to the
//   host WITHOUT stopping the simulation: copy the grid to a device staging
//   buffer, then transfer it asynchronously to PINNED host memory on a
//   separate copy stream while the compute stream continues.
//   Following functions are expected:
//     - kernel to initialise the grid;
//     - kernel for one simulation step (grid-stride loop);
//     - host function simulating the same steps;
//     - host function to verify the final GPU grid against the host grid;
//     - main function.
//   Use cudaMalloc (device memory, no Unified Memory) and cudaMallocHost.
//   Blocks = warp size * SMs. Measure GPU and host running times.
//
// WHAT IT PRACTISES: stencil indexing, constant boundaries, double buffering
// (pointer swap), cudaMalloc/cudaMallocHost/cudaMemcpyAsync, compute + copy
// streams, avoiding the data race with a staging buffer, float verification.
// Build: nvcc -O2 -o heat 05_heat_stencil_async.cu
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <utility>

#define H (1 << 10)
#define W (1 << 10)
#define STEPS 200
#define SNAPSHOT_EVERY 50
#define THREADS 256

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                   \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Shared by host and device: the same code computes both results (Lecture 6)
__host__ __device__ inline float initialTemp(int row, int height)
{
    return (row == 0 || row == height - 1) ? 90.0f : 15.0f;
}

__host__ __device__ inline float stencil(const float *in, int row, int col, int height, int width)
{
    float t = in[row * width + col];
    if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
        return t;                                          // constant boundary
    float d2x = in[row * width + col - 1] - 2.0f * t + in[row * width + col + 1];
    float d2y = in[(row - 1) * width + col] - 2.0f * t + in[(row + 1) * width + col];
    return t + 0.2f * (d2x + d2y);
}

__global__ void initGrid(float *grid, int height, int width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int id = index; id < height * width; id += stride)
        grid[id] = initialTemp(id / width, height);
}

__global__ void heatStep(const float *in, float *out, int height, int width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int id = index; id < height * width; id += stride)
        out[id] = stencil(in, id / width, id % width, height, width);
}

void simulateHost(float *grid, float *tmp, int height, int width, int steps)
{
    for (int id = 0; id < height * width; ++id) grid[id] = initialTemp(id / width, height);
    for (int s = 0; s < steps; ++s) {
        for (int id = 0; id < height * width; ++id)
            tmp[id] = stencil(grid, id / width, id % width, height, width);
        std::swap(grid, tmp);                  // swaps the local pointers only
    }
    // Result is in the caller's `grid` buffer if steps is even, else in `tmp`.
}

bool verifyResult(const float *gpu, const float *cpu, int n)
{
    for (int i = 0; i < n; ++i)
        if (fabsf(gpu[i] - cpu[i]) > 1e-3f * fmaxf(1.0f, fabsf(cpu[i]))) {
            printf("Mismatch at %d: GPU %f vs CPU %f\n", i, gpu[i], cpu[i]);
            return false;
        }
    return true;
}

double meanOf(const float *x, int n)
{
    double s = 0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s / n;
}

int main()
{
    int deviceId, numSMs, warpSize;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,   cudaDevAttrMultiProcessorCount, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize,            deviceId));
    int blocks = warpSize * numSMs;

    const int cells = H * W;
    size_t bytes = (size_t)cells * sizeof(float);

    // Device memory (Week 5 L2: non-unified memory) + pinned host memory (DLI 2)
    float *dPrev, *dNext, *dBuffer, *hSnapshot;
    CUDA_CHECK(cudaMalloc(&dPrev, bytes));
    CUDA_CHECK(cudaMalloc(&dNext, bytes));
    CUDA_CHECK(cudaMalloc(&dBuffer, bytes));
    CUDA_CHECK(cudaMallocHost(&hSnapshot, bytes));      // pinned -> truly async copies
    float *hGpuFinal = (float *)malloc(bytes);
    float *hCpuA = (float *)malloc(bytes);
    float *hCpuB = (float *)malloc(bytes);

    cudaStream_t computeStream, copyStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&copyStream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, computeStream));

    initGrid<<<blocks, THREADS, 0, computeStream>>>(dPrev, H, W);
    CUDA_CHECK(cudaGetLastError());

    for (int step = 0; step < STEPS; step += SNAPSHOT_EVERY) {
        // 1. Stage the current grid on the compute stream (device-to-device is ~free).
        //    Nobody else writes dBuffer, so the async copy below cannot race.
        CUDA_CHECK(cudaMemcpyAsync(dBuffer, dPrev, bytes, cudaMemcpyDeviceToDevice, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        // 2. Device-to-host copy on the copy stream ...
        CUDA_CHECK(cudaMemcpyAsync(hSnapshot, dBuffer, bytes, cudaMemcpyDeviceToHost, copyStream));

        // 3. ... while the compute stream keeps simulating (double buffering)
        for (int s = 0; s < SNAPSHOT_EVERY; ++s) {
            heatStep<<<blocks, THREADS, 0, computeStream>>>(dPrev, dNext, H, W);
            std::swap(dPrev, dNext);
        }
        CUDA_CHECK(cudaGetLastError());

        // 4. Wait only for the copy, then use the snapshot on the CPU
        CUDA_CHECK(cudaStreamSynchronize(copyStream));
        printf("snapshot at step %3d: mean temperature %.4f\n", step, meanOf(hSnapshot, cells));

        // 5. Finish this batch of compute before the next staging copy
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    CUDA_CHECK(cudaEventRecord(stop, computeStream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuMs;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
    CUDA_CHECK(cudaMemcpy(hGpuFinal, dPrev, bytes, cudaMemcpyDeviceToHost));

    auto t0 = std::chrono::high_resolution_clock::now();
    simulateHost(hCpuA, hCpuB, H, W, STEPS);
    auto t1 = std::chrono::high_resolution_clock::now();
    double hostMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const float *hostFinal = (STEPS % 2 == 0) ? hCpuA : hCpuB;

    bool ok = verifyResult(hGpuFinal, hostFinal, cells);
    printf("%s | GPU %.3f ms | host %.3f ms\n", ok ? "CORRECT" : "WRONG", gpuMs, hostMs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaFree(dPrev));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFreeHost(hSnapshot));
    free(hGpuFinal);
    free(hCpuA);
    free(hCpuB);
    return ok ? 0 : 1;
}
