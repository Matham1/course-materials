// =============================================================================
// CSCI 423/523/723 - Midterm 1, Fall 2025 - MODEL SOLUTION
//
// TASK (from the exam sheet):
//   The CUDA with C++ program multiplies two NxN integer matrices in parallel.
//   N = 1 << 6. Required functions:
//     - kernel to fill the matrices on independent streams;
//     - kernel to implement the multiplication;
//     - host function to implement the multiplication;
//     - host function to verify the correctness of the result;
//     - main function.
//   Number of blocks = warp size * number of streaming multiprocessors (SMs).
//   Measure running time of the kernel and host versions of multiplication.
//
// RUBRIC -> where it is earned in this file
//   12% correct implementation ....... fillMatrix, matMulKernel, matMulHost,
//                                      verifyResult, main
//    2% grid-stride loop .............. both kernels (look for "stride")
//    2% optimal blocks / threads ...... main(): blocks = warpSize * numSMs,
//                                      threads = 256 (multiple of 32)
//    2% streams ....................... main(): streamA / streamB for filling
//    2% data migration + clear memory . prefetch() to GPU / CPU, cudaFree,
//                                      cudaStreamDestroy, cudaEventDestroy
//
// Build & run:  nvcc -O2 -o matmul 01_matmul_midterm2025.cu && ./matmul
// =============================================================================

#include <stdio.h>
#include <stdlib.h>

#define N (1 << 6)                       // matrix is N x N
#define THREADS 256 

__global__ void fillMatrix(){

}
__global__ void matMulKernel(){

}
void matMulHost(){

}
void verifyResult(){

}

int main(){
    int n = N;
    size_t bytes = n*sizeof(int);
    
    cudaStream_t s1, s2; 
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
}
