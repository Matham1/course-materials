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
#define NUM_STREAMS 4
#define N 2 << 20


#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        printf("Cuda error at %s:%d: %s (%s) \n", file, line, cudaGetErrorString(err), func);
        exit(1);
    }
}

__global__ void addVectors(){

}

int main(){
    const int chunkN = N/NUM_STREAMS;
    size_t size = N * sizeof(float);
    size_t chunkSize = 
}