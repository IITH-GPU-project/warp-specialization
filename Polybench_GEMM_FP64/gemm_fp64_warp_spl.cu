
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define POLYBENCH_TIME 1
#define TILE 16
#define TILE_PAD (TILE + 1)
#define NBUFF 2

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#include "gemm_fp64.cuh"
#include "./common/polybench.h"
#include "./common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define ALPHA 32412.0f
#define BETA 2123.0f
#define RUN_ON_CPU

/**
 * PARAMETERIZED Warp Specialization GEMM
 */
__global__ void gemm_warp_specialized_16x16(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c,int dma_warps) 
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    auto block = cg::this_thread_block();

    int thread_id = ty * blockDim.x + tx;
    int warp_id = thread_id / 32;
    
    bool is_dma = (warp_id < dma_warps);
    bool is_compute = (warp_id >= dma_warps);

    __shared__ DATA_TYPE As[NBUFF][TILE][TILE_PAD];
    __shared__ DATA_TYPE Bs[NBUFF][TILE][TILE_PAD];
    __shared__ DATA_TYPE C_tile[TILE][TILE];

    
    // COMPUTE THREAD INITIALIZATION
    
    if (is_compute) {
        int compute_id = thread_id - (dma_warps * 32);
        int num_compute_threads = (blockDim.x * blockDim.y) - (dma_warps * 32);

        for (int elem_id = compute_id; elem_id < TILE * TILE; elem_id += num_compute_threads) {
            int local_r = elem_id / TILE;
            int local_c = elem_id % TILE;
            int global_r = by * TILE + local_r;
            int global_c = bx * TILE + local_c;
            if (global_r < ni && global_c < nj) {
                C_tile[local_r][local_c] = beta * c[global_r * nj + global_c];
            } else {
                C_tile[local_r][local_c] = (DATA_TYPE)0;
            }
        }
    }

    int numTiles = (nk + TILE - 1) / TILE;

    
    // PREFETCH 
    
    {
        int tile_idx = 0;
        int buf_idx = 0;
        
        if (is_dma) {
            int dma_threads = dma_warps * 32;
            
            for (int elem_id = thread_id; elem_id < TILE * TILE; elem_id += dma_threads) {
                int local_r = elem_id / TILE;
                int local_c = elem_id % TILE;
                int global_r = by * TILE + local_r;
                int global_c = tile_idx * TILE + local_c;
                As[buf_idx][local_r][local_c] = (global_r < ni && global_c < nk) ? a[global_r * nk + global_c] : (DATA_TYPE)0;
            }
            
            for (int elem_id = thread_id; elem_id < TILE * TILE; elem_id += dma_threads) {
                int local_r = elem_id / TILE;
                int local_c = elem_id % TILE;
                int global_r = tile_idx * TILE + local_r;
                int global_c = bx * TILE + local_c;
                Bs[buf_idx][local_r][local_c] = (global_r < nk && global_c < nj) ? b[global_r * nj + global_c] : (DATA_TYPE)0;
            }
        }
        block.sync();
    }

    
    // MAIN LOOP
    
    for (int t = 0; t < numTiles; ++t) {
        int consume_buf = t % 2;
        int produce_buf = (t + 1) % 2;
        
        // PHASE 1: DMA 
        if (is_dma && (t + 1) < numTiles) {
            int next_tile = t + 1;
            int dma_threads = dma_warps * 32;
            
            for (int elem_id = thread_id; elem_id < TILE * TILE; elem_id += dma_threads) {
                int local_r = elem_id / TILE;
                int local_c = elem_id % TILE;
                int global_r = by * TILE + local_r;
                int global_c = next_tile * TILE + local_c;
                As[produce_buf][local_r][local_c] = (global_r < ni && global_c < nk) ? a[global_r * nk + global_c] : (DATA_TYPE)0;
            }
            
            for (int elem_id = thread_id; elem_id < TILE * TILE; elem_id += dma_threads) {
                int local_r = elem_id / TILE;
                int local_c = elem_id % TILE;
                int global_r = next_tile * TILE + local_r;
                int global_c = bx * TILE + local_c;
                Bs[produce_buf][local_r][local_c] = (global_r < nk && global_c < nj) ? b[global_r * nj + global_c] : (DATA_TYPE)0;
            }
        }
        
        // PHASE 2: Compute 
        if (is_compute) {
            int compute_id = thread_id - (dma_warps * 32);
            int num_compute_threads = (blockDim.x * blockDim.y) - (dma_warps * 32);

            
            for (int elem_id = compute_id; elem_id < TILE * TILE; elem_id += num_compute_threads) {
                int local_r = elem_id / TILE;
                int local_c = elem_id % TILE;

                DATA_TYPE sum_reg = C_tile[local_r][local_c]; 

                #pragma unroll
                for (int k = 0; k < TILE; ++k) {
                    DATA_TYPE a_val = As[consume_buf][local_r][k];
                    DATA_TYPE b_val = Bs[consume_buf][k][local_c];
                    sum_reg += alpha * a_val * b_val;
                }
                C_tile[local_r][local_c] = sum_reg;
            }
        }
        
        block.sync();
    }

    
    // Write results
    
    if (is_compute) {
        int compute_id = thread_id - (dma_warps * 32);
        int num_compute_threads = (blockDim.x * blockDim.y) - (dma_warps * 32);

        for (int elem_id = compute_id; elem_id < TILE * TILE; elem_id += num_compute_threads) {
            int local_r = elem_id / TILE;
            int local_c = elem_id % TILE;
            
            int global_r = by * TILE + local_r;
            int global_c = bx * TILE + local_c;

            if (global_r < ni && global_c < nj) {
                c[global_r * nj + global_c] = C_tile[local_r][local_c];
            }
        }
    }
}


// HOST CODE


void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j, k;
    for (i = 0; i < _PB_NI; i++) {
        for (j = 0; j < _PB_NJ; j++) {
            C[i][j] *= beta;
            for (k = 0; k < _PB_NK; ++k) {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

void init(int ni, int nj, int nk, DATA_TYPE *alpha, DATA_TYPE *beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j;
    *alpha = ALPHA; *beta = BETA;
    for (i = 0; i < ni; i++) for (j = 0; j < nk; j++) A[i][j] = ((DATA_TYPE)i * j) / NI;
    for (i = 0; i < nk; i++) for (j = 0; j < nj; j++) B[i][j] = ((DATA_TYPE)i * j) / NI;
    for (i = 0; i < ni; i++) for (j = 0; j < nj; j++) C[i][j] = ((DATA_TYPE)i * j) / NI;
}

void compareResults(int ni, int nj,
                    DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                    DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj))
{
    int i, j, fail = 0;
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) {
                fail++;
            }
        }
    }
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init() {
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE));
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(GPU_DEVICE));
}



void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj), int dma_warps) 
{
    DATA_TYPE *A_gpu, *B_gpu, *C_gpu;

    CUDA_CHECK(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK));
    CUDA_CHECK(cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ));
    CUDA_CHECK(cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ));

    CUDA_CHECK(cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((NJ + TILE - 1) / TILE, (NI + TILE - 1) / TILE);

    printf("\n=== GEMM with Warp Specialization (block 16x16) ===\n");
    printf("Grid: (%d, %d), Block: (%d, %d) = %d threads\n", 
           grid.x, grid.y, block.x, block.y, block.x * block.y);
    printf("Total warps: %d\n", (block.x * block.y) / 32);
    printf("DMA warps: %d (%d threads)\n", dma_warps, dma_warps * 32); 
    printf("Compute warps: %d (%d threads)\n", (block.x * block.y) / 32 - dma_warps, (block.x * block.y) - (dma_warps * 32));
    printf("Shared memory per block: %d bytes\n", (2 * TILE * TILE_PAD * NBUFF + TILE * TILE) * sizeof(DATA_TYPE)); 

    polybench_start_instruments;
    
    gemm_warp_specialized_16x16<<<grid, block>>>(NI, NJ, NK, alpha, beta, A_gpu, B_gpu, C_gpu, dma_warps); 
    
    
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize()); 
    
    printf("GPU Time in seconds:\n");
    polybench_stop_instruments; 
    polybench_print_instruments;
    

    CUDA_CHECK(cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_gpu));
    CUDA_CHECK(cudaFree(B_gpu));
    CUDA_CHECK(cudaFree(C_gpu));
}

// --- UPDATED main function ---
int main(int argc, char *argv[])
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    // --- TUNING PARAMETER for DMA Warps---
    int dma_warps_to_test = 4;
    

    if (dma_warps_to_test < 1 || dma_warps_to_test > 7) {
        fprintf(stderr, "Error: dma_warps_to_test must be between 1 and 7.\n");
        return 1;
    }

    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

    init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    GPU_argv_init();

    printf("Matrix dimensions: NI=%d, NJ=%d, NK=%d\n", NI, NJ, NK);
    printf("Tile size: %dx%d (padded to %dx%d)\n", TILE, TILE, TILE, TILE_PAD);

    gemmCuda(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu), dma_warps_to_test); 

// #ifdef RUN_ON_CPU
//     init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
//     printf("\n=== CPU Reference Implementation ===\n");
//     polybench_start_instruments;
//     gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
//     printf("CPU Time in seconds:\n");
//     polybench_stop_instruments;
//     polybench_print_instruments;
//     printf("\n=== Verification ===\n");
//     compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu)); 
// #endif

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);

    return 0;
}

#include "./common/polybench.c"