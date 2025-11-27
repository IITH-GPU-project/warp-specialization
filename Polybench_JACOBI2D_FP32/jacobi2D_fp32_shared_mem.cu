/**
 * jacobi2D.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * Optimization Level 3: Tuned Block Size (16x16).
 * - Reduced threads per block from 1024 to 256.
 * - Improves Occupancy and reduces Synchronization latency.
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#define POLYBENCH_TIME 1

#include "jacobi2D_fp32.cuh"
#include "./common/polybench.h"
#include "./common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define TSTEPS 100
#define N 8192

#define RUN_ON_CPU

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE) i*(j+2) + 10) / N;
            B[i][j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
        }
    }
}


void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
    for (int t = 0; t < _PB_TSTEPS; t++)
    {
        for (int i = 1; i < _PB_N - 1; i++)
        {
            for (int j = 1; j < _PB_N - 1; j++)
            {
                B[i][j] = 0.2f * (A[i][j] + A[i][(j-1)] + A[i][(1+j)] + A[(1+i)][j] + A[(i-1)][j]);
            }
        }
        
        for (int i = 1; i < _PB_N-1; i++)
        {
            for (int j = 1; j < _PB_N-1; j++)
            {
                A[i][j] = B[i][j];
            }
        }
    }
}


__global__ void jacobi_shared_kernel(int n, DATA_TYPE* A, DATA_TYPE* B)
{
    
    __shared__ DATA_TYPE cache[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    
    int sx = tx + 1;
    int sy = ty + 1;

    
    if (gx < n && gy < n) {
        cache[sy][sx] = A[gy * n + gx];
    } else {
        cache[sy][sx] = 0.0f;
    }

    
    if (ty == 0) {
        if (gy > 0) cache[0][sx] = A[(gy - 1) * n + gx];
        else        cache[0][sx] = 0.0f; 
    }

    
    if (ty == blockDim.y - 1) {
        if (gy < n - 1) cache[sy + 1][sx] = A[(gy + 1) * n + gx];
        else            cache[sy + 1][sx] = 0.0f; 
    }

    
    if (tx == 0) {
        if (gx > 0) cache[sy][0] = A[gy * n + (gx - 1)];
        else        cache[sy][0] = 0.0f;
    }

    
    if (tx == blockDim.x - 1) {
        if (gx < n - 1) cache[sy][sx + 1] = A[gy * n + (gx + 1)];
        else            cache[sy][sx + 1] = 0.0f;
    }

    
    __syncthreads();

    
    if ((gx >= 1) && (gx < n - 1) && (gy >= 1) && (gy < n - 1))
    {
        
        B[gy * n + gx] = 0.2f * (
            cache[sy][sx] +         
            cache[sy][sx - 1] +     
            cache[sy][sx + 1] +     
            cache[sy + 1][sx] +     
            cache[sy - 1][sx]       
        );   
    }
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(a,N,N,n,n), DATA_TYPE POLYBENCH_2D(a_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(b,N,N,n,n), DATA_TYPE POLYBENCH_2D(b_outputFromGpu,N,N,n,n))
{
    int i, j, fail;
    fail = 0;   

    
    for (i=0; i<n; i++) 
    {
        for (j=0; j<n; j++) 
        {
            if (percentDiff(a[i][j], a_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
            {
                fail++;
            }
        }
    }
    
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void runJacobi2DCUDA(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n))
{
    DATA_TYPE* Agpu;
    DATA_TYPE* Bgpu;

    cudaMalloc(&Agpu, N * N * sizeof(DATA_TYPE));
    cudaMalloc(&Bgpu, N * N * sizeof(DATA_TYPE));
    
    // Copy A to both buffers for Ping-Pong correctness
    cudaMemcpy(Agpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(Bgpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // Use fixed block dimensions defined for Shared Memory
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), (unsigned int)ceil( ((float)N) / ((float)block.y) ));
    
    polybench_start_instruments;

    DATA_TYPE* src = Agpu;
    DATA_TYPE* dst = Bgpu;

    for (int t = 0; t < _PB_TSTEPS; t++)
    {
        // Call the Shared Memory Kernel
        jacobi_shared_kernel<<<grid,block>>>(n, src, dst);
        cudaDeviceSynchronize();

        // Swap
        DATA_TYPE* temp = src;
        src = dst;
        dst = temp;
    }

    printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;
    
    cudaMemcpy(A_outputFromGpu, src, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(B_outputFromGpu, src, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);

    cudaFree(Agpu);
    cudaFree(Bgpu);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
         DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int n = N;
    int tsteps = TSTEPS;

    POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(b,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(b_outputFromGpu,DATA_TYPE,N,N,n,n);

    init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

    printf("DEBUG: Actual N = %d, TSTEPS = %d\n", N, TSTEPS);
    
    runJacobi2DCUDA(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b_outputFromGpu));

    // #ifdef RUN_ON_CPU

    //     polybench_start_instruments;
    //     runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
    //     printf("CPU Time in seconds:\n");
    //     polybench_stop_instruments;
    //     polybench_print_instruments;
    
    //     compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(b_outputFromGpu));

    // #else 
    //     print_array(n, POLYBENCH_ARRAY(a_outputFromGpu));
    // #endif 

    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu);
    POLYBENCH_FREE_ARRAY(b);
    POLYBENCH_FREE_ARRAY(b_outputFromGpu);

    return 0;
}

#include "./common/polybench.c"