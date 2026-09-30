// =============================================================================
// PRACTICE 5 - Thrust + fancy iterators (DLI Part 1)                (~50 min)
//
// TASK:
//   Using Thrust only (no hand-written kernels), the program works on an
//   H x W grid of temperatures (H = 1 << 10, W = 1 << 12) stored row-major in a
//   thrust::device_vector<float>. Following functions are expected:
//     - initialise the grid on the GPU with thrust::tabulate
//       (temp = 20 + (i * 37 % 100) / 10.0f);
//     - one cooling step new = old + k * (ambient - old), k = 0.5, ambient = 20,
//       with thrust::transform into a second vector;
//     - maxChange(): max |old - new| WITHOUT a temporary array
//       (zip iterator + transform iterator + thrust::reduce);
//     - variance(): of the new grid, without a temporary array;
//     - rowMeans(): mean of each row with ONE thrust::reduce_by_key call, keys
//       generated on the fly (counting + transform iterator), output keys
//       discarded, and the division by W done by a transform output iterator;
//     - host functions computing the same values and verifying them;
//     - main function that times the GPU and host versions.
//
// WHAT IT PRACTISES: execution policies, __host__ __device__ lambdas,
// counting / transform / zip / discard / transform_output iterators,
// reduce_by_key, device_vector vs host_vector + explicit copies.
// Build: nvcc -O2 --extended-lambda -std=c++17 -o tstats 06_thrust_stats.cu
// =============================================================================
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#define H (1 << 10)
#define W (1 << 12)

constexpr float K = 0.5f, AMBIENT = 20.0f;   // constexpr -> usable inside device lambdas

__host__ __device__ inline float initTemp(int i) { return 20.0f + (i * 37 % 100) / 10.0f; }

struct divide_by {
    float d;
    __host__ __device__ float operator()(float x) const { return x / d; }
};

float maxChange(const thrust::device_vector<float> &a, const thrust::device_vector<float> &b)
{
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin()));
    auto diffs = thrust::make_transform_iterator(zipped,
        [] __host__ __device__(thrust::tuple<float, float> t) {
            return fabsf(thrust::get<0>(t) - thrust::get<1>(t));
        });
    // 2N reads, no temporary vector (naive version: 3N reads + N writes)
    return thrust::reduce(thrust::device, diffs, diffs + a.size(), 0.0f, thrust::maximum<float>{});
}

float variance(const thrust::device_vector<float> &x)
{
    // init value 0.0 (double) -> Thrust accumulates in double. Summing millions of
    // floats in float loses precision and fails verification.
    double mean = thrust::reduce(thrust::device, x.begin(), x.end(), 0.0) / x.size();
    auto sq = thrust::make_transform_iterator(x.begin(),
        [mean] __host__ __device__(float v) { double d = v - mean; return d * d; });
    return (float)(thrust::reduce(thrust::device, sq, sq + x.size(), 0.0) / x.size());
}

void rowMeans(const thrust::device_vector<float> &grid, thrust::device_vector<float> &means, int width)
{
    auto rowIds = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
        [width] __host__ __device__(int i) { return i / width; });     // key = row, never stored
#if THRUST_VERSION >= 200300   // CUDA 12.4+ (what the DLI labs use)
    auto out = thrust::make_transform_output_iterator(means.begin(), divide_by{(float)width});
    thrust::reduce_by_key(thrust::device,
                          rowIds, rowIds + grid.size(),      // keys
                          grid.begin(),                      // values
                          thrust::make_discard_iterator(),   // output keys: not needed
                          out);                              // output values: sum / width
#else   // older Thrust cannot put a transform_output_iterator here -> divide in a 2nd pass
    thrust::reduce_by_key(thrust::device, rowIds, rowIds + grid.size(), grid.begin(),
                          thrust::make_discard_iterator(), means.begin());
    thrust::transform(thrust::device, means.begin(), means.end(), means.begin(),
                      divide_by{(float)width});
#endif
}

bool close(double gpu, double cpu, const char *what)
{
    bool ok = std::fabs(gpu - cpu) <= 1e-3 * std::fmax(1.0, std::fabs(cpu));
    if (!ok) printf("%s mismatch: GPU %f vs CPU %f\n", what, gpu, cpu);
    return ok;
}

int main()
{
    const int n = H * W;
    thrust::device_vector<float> oldT(n), newT(n), means(H);

    auto g0 = std::chrono::high_resolution_clock::now();
    thrust::tabulate(thrust::device, oldT.begin(), oldT.end(),
                     [] __host__ __device__(int i) { return initTemp(i); });
    thrust::transform(thrust::device, oldT.begin(), oldT.end(), newT.begin(),
                      [] __host__ __device__(float t) { return t + K * (AMBIENT - t); });
    float gMax = maxChange(oldT, newT);
    float gVar = variance(newT);
    rowMeans(newT, means, W);
    thrust::host_vector<float> hMeans = means;               // explicit device -> host copy
    auto g1 = std::chrono::high_resolution_clock::now();     // Thrust is synchronous

    // ---- host reference ----
    auto c0 = std::chrono::high_resolution_clock::now();
    std::vector<float> o(n), nw(n);
    for (int i = 0; i < n; ++i) { o[i] = initTemp(i); nw[i] = o[i] + K * (AMBIENT - o[i]); }
    double cMax = 0, sum = 0;
    for (int i = 0; i < n; ++i) { cMax = std::fmax(cMax, std::fabs(o[i] - nw[i])); sum += nw[i]; }
    double mean = sum / n, cVar = 0;
    for (int i = 0; i < n; ++i) cVar += (nw[i] - mean) * (nw[i] - mean);
    cVar /= n;
    std::vector<double> cMeans(H, 0.0);
    for (int i = 0; i < n; ++i) cMeans[i / W] += nw[i];
    for (int r = 0; r < H; ++r) cMeans[r] /= W;
    auto c1 = std::chrono::high_resolution_clock::now();

    bool ok = close(gMax, cMax, "max change") && close(gVar, cVar, "variance");
    for (int r = 0; r < H && ok; ++r) ok = close(hMeans[r], cMeans[r], "row mean");

    printf("max change %.4f | variance %.4f | row 0 mean %.4f\n", gMax, gVar, (float)hMeans[0]);
    printf("%s | GPU %.3f ms | host %.3f ms\n", ok ? "CORRECT" : "WRONG",
           std::chrono::duration<double, std::milli>(g1 - g0).count(),
           std::chrono::duration<double, std::milli>(c1 - c0).count());
    return ok ? 0 : 1;   // device_vector frees its memory automatically (RAII)
}
