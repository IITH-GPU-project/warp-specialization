/**
 * jacobi2D_warp_spec.cu
 * * Warp Specialized Implementation of Jacobi 2D.
 * Separation of Concerns:
 * - DMA Warps: Load (16+2)x(16+2) tile including Halo.
 * - Compute Warps: Calculate 16x16 output.
 */

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define POLYBENCH_TIME 1

#include "jacobi2D_fp32.cuh"
#include "./common/polybench.h"
#include "./common/polybenchUtilFuncts.h"

// Error threshold
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define TSTEPS 100
#define N 8192

#define RUN_ON_CPU


#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16



#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// --- HELPER FUNCTIONS ---
void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = ((DATA_TYPE) i*(j+2) + 10) / N;
            B[i][j] = ((DATA_TYPE) (i-4)*(j-1) + 11) / N;
        }
    }
}

void runJacobi2DCpu(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
    for (int t = 0; t < _PB_TSTEPS; t++) {
        for (int i = 1; i < _PB_N - 1; i++) {
            for (int j = 1; j < _PB_N - 1; j++) {
                B[i][j] = 0.2f * (A[i][j] + A[i][(j-1)] + A[i][(1+j)] + A[(1+i)][j] + A[(i-1)][j]);
            }
        }
        for (int i = 1; i < _PB_N-1; i++) {
            for (int j = 1; j < _PB_N-1; j++) {
                A[i][j] = B[i][j];
            }
        }
    }
}


__global__ void jacobi_warp_specialized(int n, DATA_TYPE* A, DATA_TYPE* B, int dma_warps)
{
    
    __shared__ DATA_TYPE cache[BLOCK_DIM_Y + 2][BLOCK_DIM_X + 2];

    auto block = cg::this_thread_block();

    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;

   
    bool is_dma = (warp_id < dma_warps);
    bool is_compute = (warp_id >= dma_warps);

    
    if (is_dma) {
        int dma_num_threads = dma_warps * 32;
        int total_elements_to_load = (BLOCK_DIM_Y + 2) * (BLOCK_DIM_X + 2); // 18*18 = 324

        
        for (int k = tid; k < total_elements_to_load; k += dma_num_threads) {
            
            int sy = k / (BLOCK_DIM_X + 2);
            int sx = k % (BLOCK_DIM_X + 2);

            
            int gy = (blockIdx.y * BLOCK_DIM_Y) + sy - 1;
            int gx = (blockIdx.x * BLOCK_DIM_X) + sx - 1;

            DATA_TYPE val = 0.0f;

            
            if (gy >= 0 && gy < n && gx >= 0 && gx < n) {
                val = A[gy * n + gx];
            }

            cache[sy][sx] = val;
        }
    }

    
    block.sync();

    
    // COMPUTE
    
    if (is_compute) {
        
        int compute_tid = tid - (dma_warps * 32);
        int compute_num_threads = (blockDim.x * blockDim.y) - (dma_warps * 32);
        int total_elements_to_compute = BLOCK_DIM_Y * BLOCK_DIM_X; // 16*16 = 256

        for (int k = compute_tid; k < total_elements_to_compute; k += compute_num_threads) {
            
            int ly = k / BLOCK_DIM_X;
            int lx = k % BLOCK_DIM_X;

            int gy = blockIdx.y * BLOCK_DIM_Y + ly;
            int gx = blockIdx.x * BLOCK_DIM_X + lx;

            
            if (gy >= 1 && gy < n - 1 && gx >= 1 && gx < n - 1) {
                
                int sy = ly + 1;
                int sx = lx + 1;

                DATA_TYPE result = 0.2f * (
                    cache[sy][sx] +         
                    cache[sy][sx - 1] +     
                    cache[sy][sx + 1] +     
                    cache[sy + 1][sx] +     
                    cache[sy - 1][sx]       
                );

                B[gy * n + gx] = result;
            }
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_2D(a,N,N,n,n), DATA_TYPE POLYBENCH_2D(a_outputFromGpu,N,N,n,n))
{
    int i, j, fail = 0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (percentDiff(a[i][j], a_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void runJacobi2DCUDA(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n), int dma_warps)
{
    DATA_TYPE* Agpu;
    DATA_TYPE* Bgpu;

    CUDA_CHECK(cudaMalloc(&Agpu, N * N * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&Bgpu, N * N * sizeof(DATA_TYPE)));
    
    CUDA_CHECK(cudaMemcpy(Agpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Bgpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice)); // Init B same as A

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y); // 16x16 = 256 threads
    dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), (unsigned int)ceil( ((float)N) / ((float)block.y) ));
    
    printf("\n=== Jacobi 2D Warp Specialized ===\n");
    printf("Block: 16x16 (256 threads)\n");
    printf("DMA Warps: %d (%d threads)\n", dma_warps, dma_warps * 32);
    printf("Compute Warps: %d (%d threads)\n", 8 - dma_warps, 256 - (dma_warps * 32));

    polybench_start_instruments;

    DATA_TYPE* src = Agpu;
    DATA_TYPE* dst = Bgpu;

    for (int t = 0; t < _PB_TSTEPS; t++)
    {
        jacobi_warp_specialized<<<grid,block>>>(n, src, dst, dma_warps);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        // Swap pointers
        DATA_TYPE* temp = src;
        src = dst;
        dst = temp;
    }

    polybench_stop_instruments;
    printf("GPU Time in seconds:\n");
    polybench_print_instruments;
    
    // Copy result back (src holds the latest data due to swap)
    CUDA_CHECK(cudaMemcpy(A_outputFromGpu, src, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost));

    cudaFree(Agpu);
    cudaFree(Bgpu);
}


int main(int argc, char** argv)
{
    int n = N;
    int tsteps = TSTEPS;

    // --- TUNING PARAMETER ---
    // Total warps in block is 8 (256 threads / 32).
    // Valid values for dma_warps_to_test: 1 to 7.
    // Higher values favor memory loading (useful for memory-bound apps).
    int dma_warps_to_test = 4; 
    
    // Sanity check
    if (dma_warps_to_test < 1 || dma_warps_to_test > 7) {
        fprintf(stderr, "Error: dma_warps_to_test must be between 1 and 7.\n");
        return 1;
    }

    POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(b,DATA_TYPE,N,N,n,n);
    POLYBENCH_2D_ARRAY_DECL(a_outputFromGpu,DATA_TYPE,N,N,n,n);

    init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));

    runJacobi2DCUDA(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(a_outputFromGpu), dma_warps_to_test);

    // #ifdef RUN_ON_CPU
    //     // Re-init for CPU comparison
    //     init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
    //     polybench_start_instruments;
    //     runJacobi2DCpu(tsteps, n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(b));
    //     printf("CPU Time in seconds:\n");
    //     polybench_stop_instruments;
    //     polybench_print_instruments;
        
    //     compareResults(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(a_outputFromGpu));
    // #endif 

    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(a_outputFromGpu);
    POLYBENCH_FREE_ARRAY(b);

    return 0;
}

#include "./common/polybench.c"