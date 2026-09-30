// =============================================================================
// PRACTICE 6 - Templates, enums and __host__ __device__ (Lecture 6) (~45 min)
//
// TASK:
//   The CUDA with C++ program applies an element-wise operation
//   c[i] = op(a[i], b[i]) to two arrays of N = 1 << 20 elements, for BOTH int
//   and float arrays, with a single templated implementation.
//   The operation is selected by an enum {ADD, SUB, MUL, MAX} passed to the
//   kernel. Following functions are expected:
//     - one __host__ __device__ function template apply<T>(op, x, y) used by
//       BOTH the kernel and the host reference (no duplicated math);
//     - kernel template to fill the arrays (on independent streams);
//     - kernel template for the element-wise operation (grid-stride loop);
//     - host function template for the same operation and for verification;
//     - main that runs all 4 operations for int and for float, timing each.
//   Blocks = warp size * SMs.
//
// WHAT IT PRACTISES: function templates with __host__ __device__, template
// kernels, enums as kernel parameters, switch in device code, code reuse.
// Build: nvcc -O2 -o tmpl 07_templates_enum.cu
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define N (1 << 20)
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

enum Op { ADD, SUB, MUL, MAX };
const char *opName[] = {"ADD", "SUB", "MUL", "MAX"};

// Compiled for CPU and GPU: the kernel and the host reference share this code
template <typename T>
__host__ __device__ T apply(Op op, T x, T y)
{
    switch (op) {
        case ADD: return x + y;
        case SUB: return x - y;
        case MUL: return x * y;
        default:  return x > y ? x : y;
    }
}

template <typename T>
__global__ void fillKernel(T *x, int n, T scale)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) x[i] = (T)(i % 100) * scale;
}

template <typename T>
__global__ void elementwiseKernel(const T *a, const T *b, T *c, int n, Op op)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) c[i] = apply(op, a[i], b[i]);
}

template <typename T>
void elementwiseHost(const T *a, const T *b, T *c, int n, Op op)
{
    for (int i = 0; i < n; ++i) c[i] = apply(op, a[i], b[i]);
}

template <typename T>
bool verify(const T *gpu, const T *cpu, int n)
{
    for (int i = 0; i < n; ++i)
        if (std::fabs((double)gpu[i] - (double)cpu[i]) > 1e-5 * std::fmax(1.0, std::fabs((double)cpu[i]))) {
            printf("  mismatch at %d: %f vs %f\n", i, (double)gpu[i], (double)cpu[i]);
            return false;
        }
    return true;
}

// Runs all four operations for element type T
template <typename T>
bool runAll(const char *typeName, int blocks, T scaleA, T scaleB)
{
    size_t bytes = (size_t)N * sizeof(T);
    T *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    T *cHost = new T[N];

    cudaStream_t sA, sB;
    CUDA_CHECK(cudaStreamCreate(&sA));
    CUDA_CHECK(cudaStreamCreate(&sB));
    fillKernel<T><<<blocks, THREADS, 0, sA>>>(a, N, scaleA);
    fillKernel<T><<<blocks, THREADS, 0, sB>>>(b, N, scaleB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    bool allOk = true;
    for (int o = ADD; o <= MAX; ++o) {
        Op op = (Op)o;
        CUDA_CHECK(cudaEventRecord(start));
        elementwiseKernel<T><<<blocks, THREADS>>>(a, b, c, N, op);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));
        float kMs;
        CUDA_CHECK(cudaEventElapsedTime(&kMs, start, stop));

        auto t0 = std::chrono::high_resolution_clock::now();
        elementwiseHost(a, b, cHost, N, op);
        auto t1 = std::chrono::high_resolution_clock::now();
        double hMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool ok = verify(c, cHost, N);
        allOk = allOk && ok;
        printf("%-5s %-3s %s | kernel %.4f ms | host %.4f ms\n",
               typeName, opName[o], ok ? "CORRECT" : "WRONG", kMs, hMs);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(sA));
    CUDA_CHECK(cudaStreamDestroy(sB));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    delete[] cHost;
    return allOk;
}

int main()
{
    int deviceId, numSMs, warpSize;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,   cudaDevAttrMultiProcessorCount, deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize,            deviceId));
    int blocks = warpSize * numSMs;

    bool ok = runAll<int>("int", blocks, 3, 2);
    ok = runAll<float>("float", blocks, 0.5f, 1.25f) && ok;
    printf("%s\n", ok ? "ALL CORRECT" : "SOME WRONG");
    return ok ? 0 : 1;
}
